import time
import itertools
import numpy as np
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121312131"  # length 9, optimal is 9

    # For n >= 4, use the recursive construction with local search optimization
    # Standard recursive construction:
    # S_n = S_{n-1} with n inserted, forming blocks p + [n] + p for each permutation p of 1..n-1

    def standard_superperm(k):
        if k == 1:
            return [1]
        if k == 2:
            return [1, 2, 1]
        if k == 3:
            # 121312131 -> permutations of 1,2: 12, 21
            # 12 + [3] + 12 = 12312
            # 21 + [3] + 21 = 21321
            # Overlap: 12312 and 21321 -> suffix "12" of first? No.
            # Actually standard: 121312131
            # Let me just compute it properly
            word = [1, 2, 1]
            perms = [(1, 2), (2, 1)]
            new = []
            for p in perms:
                block = list(p) + [3] + list(p)
                overlap = 0
                max_ov = min(len(new), len(block))
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            return new

    # Get the standard word for n-1
    word_prev = standard_superperm(n - 1)

    # Extract the order of permutations of 1..n-1
    perms_order = []
    seen = set()
    for i in range(len(word_prev) - n + 2):
        window = tuple(word_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms_order.append(window)

    N = len(perms_order)
    perm_to_idx = {p: i for i, p in enumerate(perms_order)}

    # Precompute overlap matrix
    # M[i][j] = max k such that perms_order[i][-k:] == perms_order[j][:k]
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        p_i = perms_order[i]
        for j in range(N):
            p_j = perms_order[j]
            max_k = n - 1
            for k in range(max_k, 0, -1):
                if p_i[-k:] == p_j[:k]:
                    M[i][j] = k
                    break

    # Initial path is 0, 1, ..., N-1
    path = list(range(N))

    def calc_score(path):
        score = 0
        for i in range(len(path) - 1):
            score += M[path[i]][path[i + 1]]
        return score

    current_score = calc_score(path)

    # Local Search: 2-opt with first-improvement strategy
    improved = True
    iterations = 0
    while improved and time.time() < deadline - 5.0:
        improved = False
        iterations += 1
        for i in range(N - 1):
            if time.time() > deadline - 5.0:
                break
            u = path[i]
            v = path[i + 1]
            for j in range(i + 1, N):
                if time.time() > deadline - 5.0:
                    break
                x = path[j]

                if j == N - 1:
                    delta = M[u][x] - M[u][v]
                else:
                    y = path[j + 1]
                    delta = M[u][x] + M[v][y] - M[u][v] - M[x][y]

                if delta > 0:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    current_score += delta
                    improved = True
                    break
            if improved:
                break

    # Try additional 3-opt moves if time permits
    if time.time() < deadline - 10.0:
        improved3 = True
        while improved3 and time.time() < deadline - 5.0:
            improved3 = False
            for i in range(N - 2):
                if time.time() > deadline - 5.0:
                    break
                u = path[i]
                v = path[i + 1]
                for j in range(i + 2, N):
                    if time.time() > deadline - 5.0:
                        break
                    w = path[j]
                    # 3-opt: reverse segment path[i+1..j]
                    # This is actually the same as 2-opt for segment reversal
                    # Let's try a different move: swap two non-adjacent elements
                    if j == N - 1:
                        # Swap path[i+1] and path[j]
                        a = path[i + 1]
                        b = path[j]
                        # Remove edges (u,a) and (b, ?) -- this gets complicated
                        # Skip 3-opt for simplicity, 2-opt is sufficient
                        pass
                    else:
                        # Skip for now
                        pass

    # Construct the final word
    first_p = perms_order[path[0]]
    word = list(first_p) + [n] + list(first_p)

    for idx in range(1, N):
        p = perms_order[path[idx]]
        block = list(p) + [n] + list(p)
        ov = M[path[idx - 1]][path[idx]]
        word += block[ov:]

    return "".join(map(str, word))
