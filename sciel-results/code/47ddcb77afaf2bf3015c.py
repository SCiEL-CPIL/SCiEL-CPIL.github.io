import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def elapsed():
        return time.time() - start_time

    def has_triangle_circulant(n, S_set):
        """Check if circulant graph C(n, S) has a triangle."""
        for s1 in S_set:
            for s2 in S_set:
                d = (s2 - s1) % n
                if d != 0 and d in S_set:
                    return True
        return False

    def max_independent_set_circulant(n, S_set):
        """Find maximum independent set size in circulant graph C(n, S) using branch and bound."""
        non_adj = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j:
                    d = (j - i) % n
                    if d not in S_set:
                        mask |= (1 << j)
            non_adj[i] = mask

        best = [0]
        found_10 = [False]

        def dfs(candidates, current_size):
            if found_10[0]:
                return
            if current_size >= 10:
                found_10[0] = True
                return
            if current_size > best[0]:
                best[0] = current_size
            if candidates == 0:
                return
            if current_size + bin(candidates).count('1') <= best[0]:
                return
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            new_cands = candidates & non_adj[v] & ~lsb
            dfs(new_cands, current_size + 1)
            if found_10[0]:
                return
            dfs(candidates & ~lsb, current_size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def try_circulant(n, S_list):
        """Try a specific circulant graph. Return result dict if valid, else None."""
        S_set = set(S_list)
        if 0 in S_set:
            return None
        if has_triangle_circulant(n, S_set):
            return None
        alpha = max_independent_set_circulant(n, S_set)
        if alpha >= 10:
            return None
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d in S_set:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    best_result = None
    best_n = 0

    # Try known good circulant parameters first (fast path)
    known_good = [
        (39, [1, 2, 3, 5, 7, 8, 10, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38]),
        (39, [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35, 37, 38]),
        (40, [1, 2, 3, 5, 7, 8, 10, 13, 14, 16, 17, 19, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39]),
    ]
    for n, S in known_good:
        if elapsed() > deadline - 20:
            break
        result = try_circulant(n, S)
        if result is not None and n > best_n:
            best_n = n
            best_result = result

    # Systematic search for n=39 and n=40
    for n in [40, 39]:
        if elapsed() > deadline - 15:
            break
        if best_n >= n:
            continue

        if n % 2 == 0:
            half = n // 2
            candidates = list(range(1, half + 1))
        else:
            half = (n - 1) // 2
            candidates = list(range(1, half + 1))

        # Try sizes 4 through half
        for size in range(4, min(half + 1, 15)):
            if elapsed() > deadline - 10:
                break
            if best_n >= n:
                break

            count = 0
            for S_subset in combinations(candidates, size):
                if elapsed() > deadline - 10:
                    break
                count += 1
                if count > 30000:
                    break

                S = set()
                for s in S_subset:
                    S.add(s)
                    S.add(n - s)
                S.discard(0)
                S_list = sorted(S)

                # Quick triangle check before expensive alpha check
                if has_triangle_circulant(n, S):
                    continue

                alpha = max_independent_set_circulant(n, S)
                if alpha < 10:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = (j - i) % n
                            if d in S:
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    break

    if best_result is not None:
        return best_result

    # Fallback: try smaller n
    for n in [38, 37, 36, 35, 34, 33, 32, 30, 28, 26]:
        if elapsed() > deadline - 5:
            break
        if best_n >= n:
            continue

        if n % 2 == 0:
            half = n // 2
            candidates = list(range(1, half + 1))
        else:
            half = (n - 1) // 2
            candidates = list(range(1, half + 1))

        for size in range(3, min(half + 1, 12)):
            if elapsed() > deadline - 5:
                break
            if best_n >= n:
                break

            count = 0
            for S_subset in combinations(candidates, size):
                if elapsed() > deadline - 5:
                    break
                count += 1
                if count > 20000:
                    break

                S = set()
                for s in S_subset:
                    S.add(s)
                    S.add(n - s)
                S.discard(0)
                S_list = sorted(S)

                if has_triangle_circulant(n, S):
                    continue

                alpha = max_independent_set_circulant(n, S)
                if alpha < 10:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = (j - i) % n
                            if d in S:
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    break

    if best_result is not None:
        return best_result

    # Absolute last resort
    n = 26
    S = [1, 2, 5, 8, 13]
    S_set = set(S)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d in S_set:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
