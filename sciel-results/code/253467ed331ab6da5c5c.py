"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Strategy:
    # Use vectors with norm squared = 4 (Norm = 2).
    # The constraint is min pairwise dist >= max norm.
    # If all vectors have norm 2, we need dist^2 >= 4.
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # 8 - 2 u.v >= 4 => u.v <= 2.

    # Type A: Weight 1 vectors in {-2, 0, 2}^11.
    # Norm^2 = 4.
    # Count: 11 positions * 2 signs = 22.
    # All 22 are compatible.

    # Type C: Weight 4 in {-1, 0, 1}^11.
    # Norm^2 = 4.
    # W4 vs W4 requires u.v <= 2.
    # W4 vs W1 (Type A) requires u.v <= 2. (Always true for W4, max dot 2).

    # We'll use spectral partitioning + Kempe chains for W4 MIS, then add W1.

    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = (dots > 2).astype(np.int32)
    np.fill_diagonal(conflicts, 0)

    # Spectral MIS using Fiedler vector bisection + Kempe chain refinement

    def spectral_mis(conflict_matrix, time_limit, rng):
        """Find large independent set using spectral partitioning + Kempe chains."""
        n = conflict_matrix.shape[0]
        if n == 0:
            return np.array([], dtype=int)

        # Build sparse adjacency
        from scipy.sparse import csr_matrix
        adj = csr_matrix(conflict_matrix)

        # Compute Fiedler vector (second smallest eigenvalue eigenvector)
        try:
            # Use scipy's sparse eigsh for Fiedler vector
            # We need the eigenvector for the second smallest eigenvalue
            # laplacian gives L = D - A, eigenvalues in [0, 2]
            L = laplacian(adj, normed=False)
            # Get eigenvectors for smallest eigenvalues
            # k=2 gives us lambda_0=0 and lambda_1 (Fiedler)
            vals, vecs = eigsh(L, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            # Fallback to greedy if spectral fails
            return greedy_mis(conflict_matrix, time_limit, rng)

        # Initial partition by sign of Fiedler vector
        # Try both orientations and take the better one
        partition1 = (fiedler_vec >= 0).astype(int)
        partition2 = 1 - partition1

        # Score: number of vertices in set minus number of edges within set
        def score_partition(part):
            mask = part == 1
            count = mask.sum()
            # Count edges within the partition
            sub_conf = conflict_matrix[np.ix_(mask, mask)]
            internal_edges = sub_conf.sum() / 2
            return count - internal_edges * 0.5  # We want to minimize internal conflicts

        s1 = score_partition(partition1)
        s2 = score_partition(partition2)

        if s1 >= s2:
            best_part = partition1.copy()
        else:
            best_part = partition2.copy()

        # Kempe chain refinement
        # For each vertex, consider flipping it if it improves the objective
        # Objective: maximize |S| - (1/2) * internal_edges(S)
        # Or equivalently, minimize internal edges while keeping |S| large

        def current_objective(part):
            mask = part == 1
            count = mask.sum()
            if count == 0:
                return 0
            sub_conf = conflict_matrix[np.ix_(mask, mask)]
            internal_edges = sub_conf.sum() / 2
            return count - internal_edges * 0.5

        best_obj = current_objective(best_part)
        improved = True
        max_iter = 1000

        for it in range(max_iter):
            if time.time() > time_limit:
                break
            improved = False

            # Try flipping each vertex
            # For efficiency, compute the change in objective for each flip
            mask = best_part == 1
            in_set = mask
            out_set = ~mask

            # For each vertex i, compute delta if we flip it
            # If i is in set: removing it decreases |S| by 1, and removes edges from i to other set members
            #   delta_obj = -1 + (edges from i to set) / 2
            # If i is out of set: adding it increases |S| by 1, and adds edges from i to set members
            #   delta_obj = 1 - (edges from i to set) / 2

            # Compute edges from each vertex to the current set
            # conflict_matrix[i, mask].sum() gives edges from i to set
            edges_to_set = np.zeros(n, dtype=np.float64)
            # For in-set vertices: edges to other in-set vertices (excluding self)
            if np.any(in_set):
                in_set_idx = np.where(in_set)[0]
                # Compute for all vertices
                # edges_to_set[i] = sum of conflict_matrix[i, in_set]
                edges_to_set = conflict_matrix[:, in_set_idx].sum(axis=1)

            # For in-set vertices, they have an edge to themselves (0 on diagonal), so fine
            # For out-set vertices, edges_to_set[i] = edges from i to in-set

            # Compute delta for flipping
            delta = np.zeros(n, dtype=np.float64)
            # In-set vertices: delta = -1 + edges_to_set[i] / 2
            #   (removing vertex i: lose 1 from count, gain back edges_to_set[i]/2 from internal edges)
            #   Wait: internal_edges decreases by edges_to_set[i] (edges from i to other set members)
            #   So new_obj = old_obj - 1 + edges_to_set[i] / 2
            delta[in_set] = -1 + edges_to_set[in_set] / 2

            # Out-set vertices: delta = 1 - edges_to_set[i] / 2
            #   (adding vertex i: gain 1 from count, lose edges_to_set[i]/2 from new internal edges)
            delta[out_set] = 1 - edges_to_set[out_set] / 2

            # Find the best flip
            best_delta = delta.max()
            if best_delta > 1e-10:
                # Flip the vertex with best delta
                best_idx = np.argmax(delta)
                best_part[best_idx] = 1 - best_part[best_idx]
                improved = True
                # Recompute edges_to_set for next iteration (will be done in next loop)

            if not improved:
                break

        # Extract the independent set from the best partition
        # The partition gives a set S, but it may not be independent (may have internal edges)
        # We need to extract a true independent set from S
        # Use greedy on the subgraph induced by S

        mask = best_part == 1
        S_indices = np.where(mask)[0]

        if len(S_indices) == 0:
            # Fallback to greedy
            return greedy_mis(conflict_matrix, time_limit, rng)

        # Build sub-conflict matrix for S
        sub_conf = conflict_matrix[np.ix_(S_indices, S_indices)]
        n_sub = len(S_indices)

        if n_sub == 0:
            return np.array([], dtype=int)

        # Greedy MIS on the subgraph
        alive = np.ones(n_sub, dtype=bool)
        selected = []
        degrees = sub_conf.sum(axis=1).astype(np.int32)

        while np.any(alive) and time.time() < time_limit:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected.append(chosen)

            neighbors = np.where(sub_conf[chosen] > 0)[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))

            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= sub_conf[rem][:, to_remove].sum(axis=1)

        # Map back to original indices
        result = S_indices[selected]
        return result

    def greedy_mis(conflict_matrix, time_limit, rng):
        """Greedy minimum-degree MIS as fallback."""
        n = conflict_matrix.shape[0]
        alive = np.ones(n, dtype=bool)
        selected = []
        degrees = conflict_matrix.sum(axis=1).astype(np.int32)

        while np.any(alive) and time.time() < time_limit:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break

            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected.append(chosen)

            neighbors = np.where(conflict_matrix[chosen] > 0)[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))

            alive[to_remove] = False

            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict_matrix[rem][:, to_remove].sum(axis=1)

        return np.array(selected, dtype=int)

    rng = np.random.default_rng(42)
    time_limit = deadline - 15

    # Try multiple spectral seeds/restarts if time permits
    best_result = None
    best_size = -1

    # First attempt with default spectral
    result = spectral_mis(conflicts, time_limit, rng)
    if len(result) > best_size:
        best_size = len(result)
        best_result = result

    # If we have time, try a few more random perturbations of the Fiedler vector
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and best_size < N4:
        for restart in range(3):
            if time.time() > time_limit:
                break
            rng2 = np.random.default_rng(42 + restart * 1000)

            def spectral_mis_perturbed(conflict_matrix, time_limit, rng, perturbation_scale):
                n = conflict_matrix.shape[0]
                if n == 0:
                    return np.array([], dtype=int)

                from scipy.sparse import csr_matrix
                adj = csr_matrix(conflict_matrix)

                try:
                    L = laplacian(adj, normed=False)
                    vals, vecs = eigsh(L, k=2, which='SM')
                    idx = np.argsort(vals)
                    fiedler_vec = vecs[:, idx[1]]
                except Exception:
                    return greedy_mis(conflict_matrix, time_limit, rng)

                # Add random perturbation
                perturbed = fiedler_vec + rng.normal(0, perturbation_scale, size=n)

                partition1 = (perturbed >= 0).astype(int)
                partition2 = 1 - partition1

                def score_partition(part):
                    mask = part == 1
                    count = mask.sum()
                    if count == 0:
                        return 0
                    sub_conf = conflict_matrix[np.ix_(mask, mask)]
                    internal_edges = sub_conf.sum() / 2
                    return count - internal_edges * 0.5

                s1 = score_partition(partition1)
                s2 = score_partition(partition2)

                if s1 >= s2:
                    best_part = partition1.copy()
                else:
                    best_part = partition2.copy()

                def current_objective(part):
                    mask = part == 1
                    count = mask.sum()
                    if count == 0:
                        return 0
                    sub_conf = conflict_matrix[np.ix_(mask, mask)]
                    internal_edges = sub_conf.sum() / 2
                    return count - internal_edges * 0.5

                best_obj = current_objective(best_part)
                improved = True
                max_iter = 500

                for it in range(max_iter):
                    if time.time() > time_limit:
                        break
                    improved = False

                    mask = best_part == 1
                    in_set = mask
                    out_set = ~mask

                    edges_to_set = np.zeros(n, dtype=np.float64)
                    if np.any(in_set):
                        in_set_idx = np.where(in_set)[0]
                        edges_to_set = conflict_matrix[:, in_set_idx].sum(axis=1)

                    delta = np.zeros(n, dtype=np.float64)
                    delta[in_set] = -1 + edges_to_set[in_set] / 2
                    delta[out_set] = 1 - edges_to_set[out_set] / 2

                    best_delta = delta.max()
                    if best_delta > 1e-10:
                        best_idx = np.argmax(delta)
                        best_part[best_idx] = 1 - best_part[best_idx]
                        improved = True

                    if not improved:
                        break

                mask = best_part == 1
                S_indices = np.where(mask)[0]

                if len(S_indices) == 0:
                    return greedy_mis(conflict_matrix, time_limit, rng)

                sub_conf = conflict_matrix[np.ix_(S_indices, S_indices)]
                n_sub = len(S_indices)

                if n_sub == 0:
                    return np.array([], dtype=int)

                alive = np.ones(n_sub, dtype=bool)
                selected = []
                degrees = sub_conf.sum(axis=1).astype(np.int32)

                while np.any(alive) and time.time() < time_limit:
                    alive_indices = np.where(alive)[0]
                    if len(alive_indices) == 0:
                        break

                    min_deg = np.min(degrees[alive_indices])
                    cands = alive_indices[degrees[alive_indices] == min_deg]
                    chosen = cands[rng.integers(len(cands))]

                    selected.append(chosen)

                    neighbors = np.where(sub_conf[chosen] > 0)[0]
                    to_remove = np.unique(np.concatenate(([chosen], neighbors)))

                    alive[to_remove] = False

                    rem = np.where(alive)[0]
                    if len(rem) > 0:
                        degrees[rem] -= sub_conf[rem][:, to_remove].sum(axis=1)

                result = S_indices[selected]
                return result

            result_r = spectral_mis_perturbed(conflicts, time_limit, rng2, 0.1 * (restart + 1))
            if len(result_r) > best_size:
                best_size = len(result_r)
                best_result = result_r

    if best_result is None or len(best_result) == 0:
        # Final fallback
        best_result = greedy_mis(conflicts, time_limit, rng)

    result_v4 = vectors_v4[best_result]

    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
