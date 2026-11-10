import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))

    word = [1]

    for m in range(2, n + 1):
        # Extract distinct (m-1) permutations from word
        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # For m >= 4, use optimized greedy with multiple restarts + local search
        if N > 5:
            # Check time
            time_remaining = deadline - time.time() - 15  # Leave 15s for construction
            if time_remaining < 5:
                # Not enough time for optimization, use simple order
                ordered_perms = perms
            else:
                # Compute overlap matrix efficiently using numpy
                max_ov = m - 1
                perms_arr = np.array(perms, dtype=np.int8)  # shape (N, m-1)

                # overlap_matrix[i,j] = max k where perms[i][-(k):] == perms[j][:k]
                overlap_matrix = np.zeros((N, N), dtype=np.int8)

                for k in range(max_ov, 0, -1):
                    left = perms_arr[:, m-1-k:]   # shape (N, k)
                    right = perms_arr[:, :k]      # shape (N, k)
                    match = np.all(left[:, None, :] == right[None, :, :], axis=2)
                    overlap_matrix[match & (overlap_matrix == 0)] = k

                ov_list = overlap_matrix.tolist()

                # Precompute sorted neighbors for each node (by overlap descending)
                # neighbors[i] = list of (overlap, j) sorted by overlap desc
                neighbors = []
                for i in range(N):
                    row = ov_list[i]
                    # Sort indices by overlap descending, then by index for determinism
                    sorted_idx = sorted(range(N), key=lambda j: (-row[j], j))
                    neighbors.append(sorted_idx)

                # Determine number of restarts based on time
                if time_remaining < 10:
                    n_restarts = 8
                    n_polish = 1
                elif time_remaining < 25:
                    n_restarts = 30
                    n_polish = 3
                elif time_remaining < 50:
                    n_restarts = 80
                    n_polish = 5
                else:
                    n_restarts = 150
                    n_polish = 8

                best_path = None
                best_score = -1

                for r in range(n_restarts):
                    # Random start
                    start = random.randint(0, N - 1)
                    path = [start]
                    visited = [False] * N
                    visited[start] = True
                    current = start
                    current_score = 0

                    for _ in range(N - 1):
                        # Find next node with max overlap using precomputed sorted list
                        best_next = -1
                        best_ov = -1
                        for j in neighbors[current]:
                            if not visited[j]:
                                ov = ov_list[current][j]
                                if ov > best_ov:
                                    best_ov = ov
                                    best_next = j
                                # Since sorted by overlap desc, we can break early
                                if best_ov >= ov and best_next != -1:
                                    break

                        if best_next == -1:
                            for j in neighbors[current]:
                                if not visited[j]:
                                    best_next = j
                                    break

                        visited[best_next] = True
                        path.append(best_next)
                        current_score += ov_list[current][best_next]
                        current = best_next

                    if current_score > best_score:
                        best_score = current_score
                        best_path = path[:]

                    # Check time
                    if time.time() > deadline - 20:
                        break

                # Apply local search polish to the best path
                if best_path is not None and time.time() < deadline - 10:
                    # 2-opt style local search with O(1) delta computation
                    for polish_iter in range(n_polish):
                        if time.time() > deadline - 8:
                            break
                        improved = True
                        while improved and time.time() < deadline - 8:
                            improved = False
                            # Try 2-opt: reverse segments
                            for i in range(N - 2):
                                if time.time() > deadline - 8:
                                    break
                                for j in range(i + 2, min(i + 12, N)):
                                    # Compute score change for reversing best_path[i..j]
                                    # Delta = sum of new edges - sum of old edges
                                    delta = 0
                                    
                                    if i > 0:
                                        delta += ov_list[best_path[i-1]][best_path[j]] - ov_list[best_path[i-1]][best_path[i]]
                                    
                                    for k in range(i, j):
                                        delta += ov_list[best_path[j-k]][best_path[j-k-1]] - ov_list[best_path[k]][best_path[k+1]]
                                    
                                    if j + 1 < N:
                                        delta += ov_list[best_path[i]][best_path[j+1]] - ov_list[best_path[j]][best_path[j+1]]

                                    if delta > 0:
                                        best_path[i:j+1] = best_path[i:j+1][::-1]
                                        improved = True
                                        break
                                if improved:
                                    continue
                            # Try adjacent swaps
                            for i in range(N - 1):
                                if time.time() > deadline - 8:
                                    break
                                delta = 0
                                if i > 0:
                                    delta += ov_list[best_path[i-1]][best_path[i+1]] - ov_list[best_path[i-1]][best_path[i]]
                                delta += ov_list[best_path[i+1]][best_path[i]] - ov_list[best_path[i]][best_path[i+1]]
                                if i + 1 < N - 1:
                                    delta += ov_list[best_path[i]][best_path[i+2]] - ov_list[best_path[i+1]][best_path[i+2]]

                                if delta > 0:
                                    best_path[i], best_path[i+1] = best_path[i+1], best_path[i]
                                    improved = True
                                    break

                ordered_perms = [perms[i] for i in best_path] if best_path is not None else perms

        else:
            ordered_perms = perms

        # Build the new word
        new = []
        for idx in range(len(ordered_perms)):
            p = ordered_perms[idx]
            block = list(p) + [m] + list(p)

            if idx == 0:
                new += block
            else:
                # Find maximum overlap between end of new and start of block
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, 0, -1):
                    if new[-k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]

        word = new

        if time.time() > deadline:
            break

    return "".join(map(str, word))
