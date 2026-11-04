import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    # Norm squared = 4.
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
    # Norm squared = 4.
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Represent supports as bitmasks for fast intersection check
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute conflict matrix: conflict if intersection size >= 3
    conflict = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            # Count bits using bin count
            if bin(inter).count('1') >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0

    # Simulated Annealing for Maximum Independent Set
    # State: binary vector x of length N_S, x[i]=1 means support i is selected
    # Constraint: if x[i]=1 and x[j]=1, then not conflict[i,j]
    # Objective: maximize sum(x)

    def greedy_init(r):
        order = np.arange(N_S)
        rng.shuffle(order)
        x = np.zeros(N_S, dtype=bool)
        blocked = np.zeros(N_S, dtype=bool)
        for v in order:
            if not blocked[v] and not x[v]:
                x[v] = True
                blocked |= conflict[v]
        return x

    # Evaluate a state (must be valid independent set)
    def eval_state(x):
        return np.sum(x)

    # Generate a random valid move:
    # 1. Add a node not in set that is not blocked by any node in set
    # 2. Remove a node from set
    # 3. Swap: remove one, add one (if compatible)

    def get_candidates(x):
        in_set = np.where(x)[0]
        not_in_set = np.where(~x)[0]
        
        # Blocked nodes: those that conflict with any in-set node
        if len(in_set) > 0:
            blocked = np.zeros(N_S, dtype=bool)
            for v in in_set:
                blocked |= conflict[v]
        else:
            blocked = np.zeros(N_S, dtype=bool)
        
        # Add candidates: not in set and not blocked
        add_candidates = [v for v in not_in_set if not blocked[v]]
        # Remove candidates: in set
        remove_candidates = list(in_set)
        
        return add_candidates, remove_candidates

    # Simulated annealing
    T0 = 1.0
    T_end = 0.01
    num_iterations = 20000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    for restart in range(50):
        if time.time() > deadline - 25:
            break

        x = greedy_init(restart)
        current_size = eval_state(x)
        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 25:
                break

            add_candidates, remove_candidates = get_candidates(x)

            if not add_candidates and not remove_candidates:
                break

            move_type = py_rng.random()
            if move_type < 0.5 and add_candidates:
                # Add a node
                v = py_rng.choice(add_candidates)
                if x[v]:
                    continue
                # Check if v conflicts with any in-set node
                in_set = np.where(x)[0]
                if len(in_set) > 0 and np.any(conflict[v, in_set]):
                    continue
                x[v] = True
                delta = 1
                new_size = current_size + delta
                if new_size > best_current_size:
                    best_current_size = new_size
                    best_x = x.copy()
                current_size = new_size
            elif move_type < 0.8 and remove_candidates:
                # Remove a node
                v = py_rng.choice(remove_candidates)
                if not x[v]:
                    continue
                x[v] = False
                delta = -1
                new_size = current_size + delta
                # Always accept removal (it's a valid state)
                # But we want to maximize, so only accept if we can add back later
                # For SA, accept with probability exp(delta/T)
                if py_rng.random() < np.exp(delta / T):
                    current_size = new_size
                else:
                    x[v] = True  # revert
            else:
                # Swap: remove one, add one
                if not add_candidates or not remove_candidates:
                    continue
                v_remove = py_rng.choice(remove_candidates)
                v_add = py_rng.choice(add_candidates)
                if v_remove == v_add:
                    continue
                if not x[v_remove] or x[v_add]:
                    continue
                # Check if v_add conflicts with any in-set node (excluding v_remove)
                in_set = np.where(x)[0]
                in_set_no_remove = [v for v in in_set if v != v_remove]
                if len(in_set_no_remove) > 0 and np.any(conflict[v_add, in_set_no_remove]):
                    continue
                # Perform swap
                x[v_remove] = False
                x[v_add] = True
                delta = 0  # size unchanged
                # For SA with delta=0, always accept? Or use a different criterion?
                # Since size is same, we accept to explore the space
                current_size = current_size  # unchanged
                # Check if this is a new best (unlikely since size same)

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

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(supports[0] and s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        can_add = True
        for u in current_set_vectors:
            dot = np.dot(v, u)
            if dot >= 3:
                can_add = False
                break
        if can_add:
            current_set_vectors.append(v)
            current_set_vectors_set.add(tuple(v))

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])

    return final_vectors
