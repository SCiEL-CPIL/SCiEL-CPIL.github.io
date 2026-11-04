import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    # Build conflict graph: conflict if intersection size >= 3
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute conflict matrix using vectorized operations
    # For each pair (i, j), count bits in support_masks[i] & support_masks[j]
    # We can do this by expanding and using numpy
    # Actually, let's just use a loop but with a faster bit count
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if bin(inter).count('1') >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Also precompute the adjacency list for faster lookups
    # adj[i] = set of indices that conflict with i
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if conflict[i, j]:
                adj[i].add(j)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Greedy initialization
    def greedy_init():
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    # Simulated Annealing with vectorized operations
    T0 = 1.0
    T_end = 0.005
    num_iterations = 30000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(80):
        if time.time() > deadline - 25:
            break

        x = greedy_init()
        current_size = np.sum(x)
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]

            # Compute blocked nodes
            if len(in_set) > 0:
                blocked = np.zeros(N_S, dtype=bool)
                for v in in_set:
                    blocked |= conflict[v]
            else:
                blocked = np.zeros(N_S, dtype=bool)

            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = in_set

            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break

            move_type = py_rng.random()
            if move_type < 0.45 and len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_current_size:
                    best_current_size = current_size
                    best_x = x.copy()
            elif move_type < 0.75 and len(remove_candidates) > 0:
                v = py_rng.choice(remove_candidates)
                x[v] = False
                current_size -= 1
                # Accept with probability exp(-1/T)
                if py_rng.random() < np.exp(-1.0 / T):
                    pass
                else:
                    x[v] = True
                    current_size += 1
            else:
                # Swap
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_remove = py_rng.choice(remove_candidates)
                    v_add = py_rng.choice(add_candidates)
                    if v_remove != v_add:
                        # Check if v_add conflicts with any in-set node except v_remove
                        in_set_no_remove = [v for v in in_set if v != v_remove]
                        if len(in_set_no_remove) > 0:
                            if np.any(conflict[v_add, in_set_no_remove]):
                                T *= cooling
                                continue
                        x[v_remove] = False
                        x[v_add] = True
                        # Size unchanged, always accept for exploration

            T *= cooling

        if best_current_size > best_size:
            best_size = best_current_size
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

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    current_arr = np.array(current_set_vectors, dtype=np.int64)

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 15:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        dots = current_arr @ v
        if np.any(dots >= 3):
            continue
        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_arr = np.vstack([current_arr, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
