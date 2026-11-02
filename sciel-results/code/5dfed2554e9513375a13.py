"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Generate Weight 3 and Weight 4 vectors
    candidates = []
    norms_sq = []

    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)

    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)

    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)

    # Compute dot products
    dots = np.dot(candidates, candidates.T)

    # Determine conflicts
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)

    # Precompute neighbors for each node
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())

    # Build adjacency matrix for spectral methods
    adj = np.zeros((n_candidates, n_candidates), dtype=np.float64)
    for i in range(n_candidates):
        for j in neighbors[i]:
            adj[i, j] = 1.0

    # Spectral relaxation: compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj

    best_selected = []

    # Use scipy for sparse eigenvalue computation if available
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        L_sparse = csr_matrix(L)
        # Find k smallest eigenvalues (skip the trivial 0 eigenvalue)
        k_eig = min(50, n_candidates - 1)
        evals, evecs = eigsh(L_sparse, k=k_eig, which='SM')

        # Sort by eigenvalue
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Try multiple eigenvector combinations
        for eig_idx in range(1, min(10, k_eig)):
            if time.time() > deadline - 10.0:
                break

            vec = evecs[:, eig_idx]
            # Threshold at 0 to get a subset
            selected_idx = np.where(vec > 0)[0]

            # Greedy refine: remove conflicts
            available = set(selected_idx.tolist())
            refined = []
            while available:
                if time.time() > deadline - 5.0:
                    break
                # Pick vertex with min conflicts in available
                if len(available) > 100:
                    sample = random.sample(list(available), 100)
                else:
                    sample = list(available)

                best_v = None
                min_deg = float('inf')
                for v in sample:
                    deg = len(neighbors[v] & available)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v

                if best_v is None:
                    break

                refined.append(best_v)
                available.discard(best_v)
                for n in neighbors[best_v]:
                    if n in available:
                        available.discard(n)

            if len(refined) > len(best_selected):
                best_selected = refined

        # Also try the negative side
        for eig_idx in range(1, min(10, k_eig)):
            if time.time() > deadline - 10.0:
                break

            vec = evecs[:, eig_idx]
            selected_idx = np.where(vec < 0)[0]

            available = set(selected_idx.tolist())
            refined = []
            while available:
                if time.time() > deadline - 5.0:
                    break
                if len(available) > 100:
                    sample = random.sample(list(available), 100)
                else:
                    sample = list(available)

                best_v = None
                min_deg = float('inf')
                for v in sample:
                    deg = len(neighbors[v] & available)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v

                if best_v is None:
                    break

                refined.append(best_v)
                available.discard(best_v)
                for n in neighbors[best_v]:
                    if n in available:
                        available.discard(n)

            if len(refined) > len(best_selected):
                best_selected = refined

    except Exception:
        # Fallback to pure greedy if spectral fails
        best_selected = []

    # If spectral didn't find anything good, do pure greedy with random restarts
    if len(best_selected) < 20 or time.time() < deadline - 30.0:
        np.random.seed(42)
        random.seed(42)

        def greedy_min_degree():
            available = set(range(n_candidates))
            selected = []

            while available:
                if time.time() > deadline:
                    break

                if len(available) > 200:
                    sample = random.sample(list(available), 200)
                else:
                    sample = list(available)

                best_v = None
                min_deg = float('inf')
                for v in sample:
                    deg = len(neighbors[v] & available)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v

                if best_v is None:
                    break

                selected.append(best_v)
                available.discard(best_v)
                for n in neighbors[best_v]:
                    if n in available:
                        available.discard(n)

            return selected

        num_trials = 50
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break

            sel = greedy_min_degree()
            if len(sel) > len(best_selected):
                best_selected = sel

    # Local search improvement on best_selected
    if best_selected and time.time() < deadline - 5.0:
        selected_set = set(best_selected)

        # Try to add more vertices
        added = True
        while added and time.time() < deadline - 5.0:
            added = False
            for v in range(n_candidates):
                if v in selected_set:
                    continue
                # Check if v conflicts with any selected vertex
                if not (neighbors[v] & selected_set):
                    selected_set.add(v)
                    added = True

        # Try swaps: remove one selected, add up to two non-selected
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            selected_list = list(selected_set)
            for i, v_remove in enumerate(selected_list):
                if time.time() > deadline - 5.0:
                    break

                # Try removing v_remove and adding two others
                trial_set = selected_set - {v_remove}

                # Find candidates that don't conflict with trial_set
                candidates_to_add = []
                for v_add in range(n_candidates):
                    if v_add in trial_set or v_add == v_remove:
                        continue
                    if not (neighbors[v_add] & trial_set):
                        candidates_to_add.append(v_add)

                # Try pairs of candidates
                for j in range(len(candidates_to_add)):
                    if time.time() > deadline - 5.0:
                        break
                    v1 = candidates_to_add[j]
                    if neighbors[v1] & trial_set:
                        continue
                    new_set1 = trial_set | {v1}

                    for k in range(j + 1, len(candidates_to_add)):
                        if time.time() > deadline - 5.0:
                            break
                        v2 = candidates_to_add[k]
                        if neighbors[v2] & new_set1:
                            continue
                        # Check v1 and v2 don't conflict
                        if v1 in neighbors[v2]:
                            continue

                        new_set = new_set1 | {v2}
                        if len(new_set) > len(selected_set):
                            selected_set = new_set
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

    # Convert to final array
    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)

    return candidates[list(selected_set)]
# EVOLVE-BLOCK-END
