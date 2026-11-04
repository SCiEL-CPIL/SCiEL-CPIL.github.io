import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    # Precompute intersection sizes for all pairs of supports
    # inter[i][j] = len(supports[i] & supports[j])
    inter = np.zeros((N_S, N_S), dtype=np.int8)
    for i in range(N_S):
        s_i = set(supports[i])
        for j in range(i, N_S):
            s_j = set(supports[j])
            inter_len = len(s_i & s_j)
            inter[i, j] = inter_len
            inter[j, i] = inter_len

    # Conflict graph for independent set (intersection >= 3)
    # We want to maximize the number of selected supports such that intersection <= 2.
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(N_S):
            if inter[i, j] >= 3:
                adj[i].add(j)
                adj[j].add(i)

    def greedy_mis(adj, order):
        in_set = [False] * N_S
        available = set(range(N_S))
        for v in order:
            if v in available:
                in_set[v] = True
                # Remove v and its neighbors
                to_remove = {v} | adj[v]
                available -= to_remove
        return in_set

    def local_search_mis(in_set, adj):
        # Try to add nodes
        # If a node is not in set, and none of its neighbors are in set, add it.
        changed = True
        while changed:
            changed = False
            for v in range(N_S):
                if not in_set[v]:
                    # Check if any neighbor is in set
                    conflict = False
                    for u in adj[v]:
                        if in_set[u]:
                            conflict = True
                            break
                    if not conflict:
                        in_set[v] = True
                        changed = True
        return in_set

    best_size = 0
    best_set = []
    rng = random.Random(42)

    num_restarts = 2000

    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Random order
        order = list(range(N_S))
        rng.shuffle(order)

        in_set = greedy_mis(adj, order)
        in_set = local_search_mis(in_set, adj)

        size = sum(in_set)
        if size > best_size:
            best_size = size
            best_set = [i for i, b in enumerate(in_set) if b]

    # Generate vectors for best_set
    vectors_D = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors_D.append(v)

    vectors_D = np.array(vectors_D, dtype=np.int64)

    # Greedy fill with sign checks
    # Generate all other Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)

    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)

    current_vectors = list(vectors_D)
    current_set = set(map(tuple, current_vectors))

    # Shuffle
    indices_all = list(range(len(all_vectors_D)))
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 5:
            break

        v = all_vectors_D[i]
        if tuple(v) in current_set:
            continue

        can_add = True
        # Check compatibility
        # dot(v, u) <= 2 for all u in current_vectors
        for u in current_vectors:
            if np.dot(v, u) > 2:
                can_add = False
                break

        if can_add:
            current_vectors.append(v)
            current_set.add(tuple(v))

    final_vectors_D = np.array(current_vectors, dtype=np.int64)

    final_vectors = np.vstack([vectors_A, final_vectors_D])

    return final_vectors
