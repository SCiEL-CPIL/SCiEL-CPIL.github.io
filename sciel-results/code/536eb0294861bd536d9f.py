import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Precompute popcount for 11-bit integers
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # Build support masks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Vectorized conflict matrix
    masks_col = support_masks[:, None]
    masks_row = support_masks[None, :]
    inter_matrix = masks_col & masks_row
    popcounts = popcount[inter_matrix]
    conflict = popcounts >= 3

    # Degrees for greedy heuristic
    degrees = conflict.sum(axis=1)

    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    def greedy_init_heuristic():
        """
        Greedy initialization using dynamic minimum degree heuristic with random tie-breaking.
        At each step, select the available vertex with the fewest available neighbors.
        """
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        available = np.ones(N_S, dtype=bool)
        current_size = 0
        # Track number of available neighbors for each vertex
        # A vertex u is a neighbor of v if conflict[u,v] is True
        # available_degree[v] = number of available vertices that conflict with v
        # Initially all are available, so available_degree = degrees
        avail_deg = degrees.copy().astype(np.int32)

        for _ in range(N_S):
            if not available.any():
                break

            avail_indices = np.where(available)[0]
            # Among available vertices, find those with minimum available degree
            min_deg = avail_deg[avail_indices].min()
            candidates = avail_indices[avail_deg[avail_indices] == min_deg]

            if len(candidates) == 0:
                break
            v = py_rng.choice(candidates)

            x[v] = True
            current_size += 1
            # v is now selected; its neighbors become blocked
            neighbors = np.where(conflict[v] & available)[0]
            # For each neighbor, decrease their available degree
            avail_deg[neighbors] -= 1
            blocked |= conflict[v]
            available[neighbors] = False
            available[v] = False

        return x, current_size

    def local_search_1swap(x, current_size, max_iters=5000):
        """
        Iterative 1-swap local search with efficient blocked tracking.
        """
        best_x = x.copy()
        best_size_local = current_size

        # Precompute for faster swap checks
        # conflict_count_with_x[v] = number of selected vertices that conflict with v
        conflict_count = np.zeros(N_S, dtype=np.int32)
        if current_size > 0:
            conflict_count = conflict[x].sum(axis=0).astype(np.int32)

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Vertices not in x and not conflicting with any selected vertex
            add_candidates = np.where((~x) & (conflict_count == 0))[0]

            if len(add_candidates) > 0:
                # Prefer adding vertices that unblock the most other vertices
                # Score: how many currently-blocked vertices would become unblocked
                # For simplicity, just pick randomly from add candidates
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                # Update conflict counts
                conflict_count += conflict[v]
                if current_size > best_size_local:
                    best_size_local = current_size
                    best_x = x.copy()
            else:
                # No add possible. Try swaps.
                remove_candidates = np.where(x)[0]
                if len(remove_candidates) == 0:
                    break

                improved = False
                not_in_x = np.where(~x)[0]
                if len(not_in_x) == 0:
                    break

                # For each candidate v_add not in x, check if it conflicts with
                # at most one selected vertex (which would be the one we remove)
                # n_conflicts[v_add] = conflict_count[v_add]
                # We want vertices where conflict_count[v_add] <= 1
                # If == 0, it's an add (but we already checked those)
                # If == 1, find which selected vertex it conflicts with; if we remove that, it works

                # Try swaps: for v_add with conflict_count == 1, find the unique conflict
                swap_candidates = np.where((~x) & (conflict_count == 1))[0]

                if len(swap_candidates) > 0:
                    # Try random swaps from this set
                    for _ in range(30):
                        if time.time() > deadline - 15:
                            break
                        v_add = py_rng.choice(swap_candidates)
                        # Find the one selected vertex that conflicts with v_add
                        conflicting_selected = np.where(conflict[v_add] & x)[0]
                        if len(conflicting_selected) == 0:
                            # Should not happen since conflict_count==1
                            continue
                        v_rem = conflicting_selected[0]
                        # Perform swap
                        x[v_rem] = False
                        x[v_add] = True
                        # Update conflict counts
                        conflict_count -= conflict[v_rem]
                        conflict_count += conflict[v_add]
                        improved = True
                        break

                if not improved:
                    # Try random swaps more broadly
                    for _ in range(20):
                        if time.time() > deadline - 15:
                            break
                        v_rem = py_rng.choice(remove_candidates)
                        v_add = py_rng.choice(not_in_x)

                        if v_add == v_rem:
                            continue

                        # Check if v_add conflicts with any node in x, excluding v_rem
                        conflicts_with_x = conflict[v_add] & x
                        n_conflicts = conflicts_with_x.sum()

                        if n_conflicts == 0 or (n_conflicts == 1 and conflicts_with_x[v_rem]):
                            x[v_rem] = False
                            x[v_add] = True
                            conflict_count -= conflict[v_rem]
                            conflict_count += conflict[v_add]
                            improved = True
                            break

                if not improved:
                    # Perturb: randomly remove one node
                    if current_size > 0:
                        v_rem = py_rng.choice(remove_candidates)
                        x[v_rem] = False
                        current_size -= 1
                        conflict_count -= conflict[v_rem]
                    else:
                        break

        return best_x, best_size_local

    num_restarts = 500
    iters_per_restart = 3000

    for restart in range(num_restarts):
        if time.time() > deadline - 25:
            break

        x, size = greedy_init_heuristic()
        best_x, best_size_local = local_search_1swap(x, size, iters_per_restart)

        if best_size_local > best_size:
            best_size = best_size_local
            best_support_set = set(np.where(best_x)[0].tolist())

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    # Vectorized greedy fill
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))

    if len(current_set_vectors) > 0:
        current_matrix = np.array(current_set_vectors, dtype=np.int64)
    else:
        current_matrix = np.zeros((0, D), dtype=np.int64)

    rng = np.random.RandomState(42)
    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue

        if len(current_matrix) > 0:
            dots = current_matrix @ v
            if np.any(dots >= 3):
                continue

        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_matrix = np.vstack([current_matrix, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
