import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
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

    # Compute dot products
    dots = vectors_v4 @ vectors_v4.T

    # Conflict if dot > 2
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    best_size = 0
    best_selected = None

    # Spectral + Kempe-chain local search
    # The conflict graph: we want max independent set
    # Equivalently: max-cut where we assign each vertex to S (selected) or T (not selected)
    # S is an independent set if no edge has both endpoints in S
    # Objective: maximize |S| subject to S being independent

    def kempe_local_search(assign, conflicts, N, deadline):
        """
        assign: boolean array, True = in independent set
        Try Kempe-chain flips: for a vertex v not in S, if flipping v to S 
        doesn't create a conflict (i.e., no neighbor in S), do it.
        Also try: remove a vertex from S, then add multiple others.
        """
        assign = assign.copy()
        in_set = assign.copy()
        size = in_set.sum()

        # Precompute neighbor lists as bitmasks for speed
        # Actually, let's use the conflict matrix directly

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False

            # Phase 1: Add any vertex that can be added
            not_in_set = ~in_set
            if np.any(not_in_set):
                # For each vertex not in set, check if it conflicts with any in set
                # conflicts[v] & in_set gives neighbors of v that are in set
                can_add = np.zeros(N, dtype=bool)
                for v in np.where(not_in_set)[0]:
                    if conflicts[v] @ in_set == 0:
                        can_add[v] = True
                if np.any(can_add):
                    in_set[can_add] = True
                    size += can_add.sum()
                    improved = True

            # Phase 2: Try removing one vertex and adding others
            if not improved:
                for v in np.where(in_set)[0]:
                    if time.time() > deadline - 2:
                        break
                    # Temporarily remove v
                    in_set[v] = False
                    size -= 1
                    # Try to add as many as possible
                    added = 0
                    for u in np.where(~in_set)[0]:
                        if conflicts[u] @ in_set == 0:
                            in_set[u] = True
                            size += 1
                            added += 1
                    if added >= 2:
                        improved = True
                        break
                    else:
                        # Revert
                        in_set[v] = True
                        size += 1

        return in_set, size

    def spectral_init(conflicts, N, k_eigs, rng):
        """
        Use spectral method to initialize the independent set.
        Compute Laplacian, get eigenvectors, use sign patterns.
        """
        # Laplacian L = D - A where A is adjacency (conflict matrix)
        degrees = conflicts.sum(axis=1)
        L = np.diag(degrees) - conflicts.astype(float)

        # Compute eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except:
            return None

        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        candidates = []

        # Use second smallest eigenvector (Fiedler vector)
        # Assign vertices with positive component to S
        for i in range(1, min(k_eigs, N)):
            vec = eigvecs[:, i]
            # Try both sign assignments
            assign1 = vec > 0
            assign2 = vec < 0

            # Also try thresholding at different levels
            for threshold in [0, np.median(vec), np.percentile(vec, 40), np.percentile(vec, 60)]:
                assign = vec > threshold
                candidates.append(assign)
                assign = vec < threshold
                candidates.append(assign)

        # Random projections
        for _ in range(5):
            proj = rng.standard_normal(N)
            # Project onto the eigenspace
            coeff = eigvecs[:, 1:k_eigs+1].T @ proj
            proj_vec = eigvecs[:, 1:k_eigs+1] @ coeff
            for threshold in [0, np.median(proj_vec)]:
                candidates.append(proj_vec > threshold)
                candidates.append(proj_vec < threshold)

        return candidates

    # Run spectral + local search with multiple seeds
    rng = np.random.default_rng(42)

    # First, try spectral initialization
    if time.time() < deadline - 10:
        candidates = spectral_init(conflicts, N4, 5, rng)
        if candidates:
            for assign in candidates:
                if time.time() > deadline - 5:
                    break
                # Ensure it's a valid independent set (remove conflicts)
                # Greedy cleanup: remove vertices that conflict
                assign_clean = assign.copy()
                changed = True
                while changed:
                    changed = False
                    for v in np.where(assign_clean)[0]:
                        if conflicts[v] @ assign_clean > 0:
                            # Remove the one with higher degree in the induced subgraph
                            neighbors_in_set = np.where(conflicts[v] & assign_clean)[0]
                            if len(neighbors_in_set) > 0:
                                # Remove v
                                assign_clean[v] = False
                                changed = True
                                break

                in_set, size = kempe_local_search(assign_clean, conflicts, N4, deadline)
                if size > best_size:
                    best_size = size
                    best_selected = np.where(in_set)[0].tolist()

    # Also try greedy min-degree as baseline
    if time.time() < deadline - 5:
        alive = np.ones(N4, dtype=bool)
        selected = []
        degrees = conflicts.sum(axis=1).astype(np.int32)

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected

    # Additional local search on the best solution found
    if best_selected is not None and time.time() < deadline - 3:
        in_set = np.zeros(N4, dtype=bool)
        in_set[best_selected] = True
        in_set, size = kempe_local_search(in_set, conflicts, N4, deadline)
        if size > best_size:
            best_size = size
            best_selected = np.where(in_set)[0].tolist()

    # More aggressive local search: try multiple random perturbations
    if best_selected is not None and time.time() < deadline - 5:
        for trial in range(10):
            if time.time() > deadline - 3:
                break
            # Perturb: remove a random subset, re-add greedily
            in_set = np.zeros(N4, dtype=bool)
            in_set[best_selected] = True

            # Remove 10% randomly
            to_remove = rng.choice(best_selected, size=max(1, len(best_selected) // 10), replace=False)
            in_set[to_remove] = False

            # Re-add greedily
            for v in np.where(~in_set)[0]:
                if conflicts[v] @ in_set == 0:
                    in_set[v] = True

            in_set, size = kempe_local_search(in_set, conflicts, N4, deadline)
            if size > best_size:
                best_size = size
                best_selected = np.where(in_set)[0].tolist()

    result_v4 = vectors_v4[best_selected]

    # Combine V4 and V1
    # All V1 are compatible with all V4 (max dot = 2, not > 2)
    # V1 among themselves: can take both +2 and -2 for each position
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
