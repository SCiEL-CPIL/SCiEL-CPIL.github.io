import time
import sys
import math
from itertools import permutations
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    if n == 1:
        return "1"

    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)

    # Precompute overlap matrix M
    # M[i][j] = max k such that perms[i][n-k:] == perms[j][:k]
    M = [[0] * N for _ in range(N)]

    # Precompute prefix maps for each k
    prefix_maps = [{} for _ in range(n)]
    for k in range(1, n):
        p_map = prefix_maps[k]
        for idx, p in enumerate(perms):
            pre = p[:k]
            if pre not in p_map:
                p_map[pre] = []
            p_map[pre].append(idx)

    # Fill M: for each i, iterate k from n-1 down to 1
    for i in range(N):
        for k in range(n - 1, 0, -1):
            suf = perms[i][n-k:]
            if suf in prefix_maps[k]:
                for j in prefix_maps[k][suf]:
                    if M[i][j] == 0:
                        M[i][j] = k

    # Greedy Nearest Neighbor
    visited = [False] * N
    path = [0]
    visited[0] = True

    for _ in range(N - 1):
        last = path[-1]
        best_j = -1
        best_m = -1
        for j in range(N):
            if not visited[j]:
                m = M[last][j]
                if m > best_m:
                    best_m = m
                    best_j = j
        if best_j == -1:
            break
        path.append(best_j)
        visited[best_j] = True

    # Or-opt Local Search to maximize overlap sum
    def get_overlap_sum(p):
        s = 0
        for t in range(len(p) - 1):
            s += M[p[t]][p[t+1]]
        return s

    current_sum = get_overlap_sum(path)

    improved = True
    iteration = 0
    max_iterations = 50

    while improved and time.time() < deadline - 15:
        improved = False
        iteration += 1
        if iteration > max_iterations:
            break

        for k in range(1, N - 1):
            u = path[k-1]
            v = path[k]
            w = path[k+1]

            gain_remove = M[u][w] - M[u][v] - M[v][w]

            for p in range(N):
                if p == k or p == k - 1:
                    continue

                if p < k:
                    a = path[p-1] if p > 0 else -1
                    b = path[p]
                else:
                    if p == N - 1:
                        a = path[N-1]
                        b = -1
                    else:
                        b = path[p+1]
                        if p - 1 < k:
                            a = path[p-1]
                        else:
                            a = path[p]

                if a == -1 and b == -1:
                    gain_insert = 0
                elif a == -1:
                    gain_insert = M[v][b]
                elif b == -1:
                    gain_insert = M[a][v]
                else:
                    gain_insert = M[a][v] + M[v][b] - M[a][b]

                total_gain = gain_remove + gain_insert

                if total_gain > 0:
                    path.pop(k)
                    if p > k:
                        p -= 1
                    path.insert(p, v)
                    current_sum += total_gain
                    improved = True
                    break

            if improved:
                break

    # Construct initial superpermutation string
    word = list(perms[path[0]])
    for t in range(1, len(path)):
        i = path[t-1]
        j = path[t]
        ov = M[i][j]
        word += perms[j][ov:]

    # Verify initial word is valid
    if len(word) < n:
        return _classic_superperm_str(n)

    all_perms_set = set(perms)
    perm_counts = Counter()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if w in all_perms_set:
            perm_counts[w] += 1

    covered = sum(1 for p in all_perms_set if perm_counts.get(p, 0) > 0)

    if covered < N:
        # Fallback to classic
        return _classic_superperm_str(n)

    # Greedy deletion to shorten the word
    changed = True
    i = 0
    while changed and (time.time() - start_time) < deadline - 5:
        changed = False
        while i < len(word) and (time.time() - start_time) < deadline - 5:
            # Determine which windows are affected by removing position i
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)

            # Check if any permutation would become uncovered
            safe = True
            for s in range(s_min, s_max + 1):
                window = tuple(word[s:s + n])
                if window in all_perms_set and perm_counts[window] == 1:
                    safe = False
                    break

            if safe:
                # Remove position i
                for s in range(s_min, s_max + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms_set:
                        perm_counts[window] -= 1

                word.pop(i)

                # New windows that cross the join point
                new_s_min = max(0, i - n + 1)
                new_s_max = min(i - 1, len(word) - n)

                for s in range(new_s_min, new_s_max + 1):
                    window = tuple(word[s:s + n])
                    if window in all_perms_set:
                        old_count = perm_counts.get(window, 0)
                        perm_counts[window] = old_count + 1

                changed = True
                # Don't increment i; re-examine the same position
            else:
                i += 1

    # Final verification
    final_perms = set()
    for i in range(len(word) - n + 1):
        final_perms.add(tuple(word[i:i + n]))

    if not all_perms_set.issubset(final_perms):
        return _classic_superperm_str(n)

    return "".join(map(str, word))

def _classic_superperm_str(n):
    if n == 1:
        return "1"
    # Simple valid construction: concatenate all permutations
    result = []
    for p in permutations(range(1, n + 1)):
        result.extend(p)
    return "".join(map(str, result))
