import numpy as np
import time
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    D = 11
    W = 8  # Weight of vectors

    # 1. Generate all supports (8-subsets of range(11))
    supports = list(combinations(range(D), W))
    N_sup = len(supports)  # C(11,8) = 165

    # 2. Build incompatibility graph
    # Two supports are incompatible if they intersect in >= 5 elements
    # (because for weight-8 vectors with +/-1 signs, dot product magnitude
    #  can be up to the intersection size, and we need |u.v| <= 4 for
    #  distance >= norm when norm^2 = 8, i.e., |u.v| <= 4)
    # Actually let's re-derive:
    # For vectors with entries in {-1, 1} on support S (size 8), 0 elsewhere:
    #   |u|^2 = 8
    #   u.v = sum_{i in S_u ∩ S_v} u_i v_i
    #   Max |u.v| = |S_u ∩ S_v| (when signs align)
    #   We need |u - v|^2 >= 8 (norm squared)
    #   |u-v|^2 = |u|^2 + |v|^2 - 2 u.v = 16 - 2 u.v
    #   Need 16 - 2 u.v >= 8 => u.v <= 4
    #   And 16 + 2 u.v >= 8 => u.v >= -4
    #   So need |u.v| <= 4
    #   Worst case |u.v| = |S_u ∩ S_v|, so need |S_u ∩ S_v| <= 4
    #   Incompatible if |S_u ∩ S_v| >= 5

    sup_sets = [set(s) for s in supports]

    # Precompute incompatibility matrix
    A = np.zeros((N_sup, N_sup), dtype=np.int8)

    for i in range(N_sup):
        for j in range(i + 1, N_sup):
            inter_size = len(sup_sets[i] & sup_sets[j])
            if inter_size >= 5:
                A[i, j] = 1
                A[j, i] = 1

    # 3. Find Maximum Independent Set using simulated annealing + greedy polish
    rng = np.random.default_rng(123)

    best_size = 0
    best_x = None

    lambda_penalty = 1000.0
    T_start = 5.0
    T_end = 0.005
    n_iterations = 3000
    n_runs = 30

    for run in range(n_runs):
        if time.time() > deadline - 5.0:
            break

        # Greedy initialization
        order = rng.permutation(N_sup)
        x = np.zeros(N_sup, dtype=np.int8)
        for idx in order:
            if not np.any(A[idx] & x):
                x[idx] = 1

        current_size = np.sum(x)

        # SA loop
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)

        for step in range(n_iterations):
            if time.time() > deadline - 5.0:
                break

            i = rng.integers(N_sup)
            conflict_count = np.sum(A[i] * x)

            if x[i] == 0:
                delta_E = -1.0 + lambda_penalty * conflict_count
            else:
                delta_E = 1.0 - lambda_penalty * conflict_count

            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                x[i] = 1 - x[i]

            T *= cooling_rate

        # Cleanup: remove conflicts greedily
        changed = True
        while changed:
            changed = False
            selected = np.where(x == 1)[0]
            for idx in selected:
                if x[idx] == 1 and np.sum(A[idx] * x) > 0:
                    x[idx] = 0
                    changed = True
                    break

        # Fill: add all non-conflicting nodes
        changed = True
        while changed:
            changed = False
            for i in range(N_sup):
                if x[i] == 0 and np.sum(A[i] * x) == 0:
                    x[i] = 1
                    changed = True

        size = np.sum(x)
        if size > best_size:
            best_size = size
            best_x = x.copy()

    if best_x is None:
        return np.array([[1] + [0] * (D - 1)], dtype=np.int8)

    # 4. Generate vectors
    # For each selected support, generate all 2^7 = 128 even-parity sign patterns
    # Even parity: product of signs = 1 (i.e., even number of -1s)
    # This ensures intra-support distance: for two even-parity vectors on same support,
    # Hamming distance is even. Min even Hamming distance >= 2, so |u-v|^2 >= 8.

    vectors = []
    for i in range(N_sup):
        if best_x[i] == 1:
            sup = supports[i]
            # Generate all 2^8 sign patterns, keep only even parity
            for signs in product([1, -1], repeat=W):
                # Count number of -1s
                if sum(1 for s in signs if s == -1) % 2 == 0:
                    v = np.zeros(D, dtype=np.int8)
                    for idx, s in zip(sup, signs):
                        v[idx] = s
                    vectors.append(v)

    if not vectors:
        return np.array([[1] + [0] * (D - 1)], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
