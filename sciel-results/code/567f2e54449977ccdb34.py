import time
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

        if N <= 5:
            ordered_perms = perms
        else:
            if time.time() > deadline - 30:
                ordered_perms = perms
            else:
                # Compute overlap matrix using numpy for speed
                perms_np = np.array(perms, dtype=np.int8)

                max_ov = m - 1
                overlap_matrix = np.zeros((N, N), dtype=np.int8)

                # Vectorized overlap computation: for each k from max_ov down to 1
                for k in range(max_ov, 0, -1):
                    left = perms_np[:, m - 1 - k:]   # shape (N, k)
                    right = perms_np[:, :k]           # shape (N, k)
                    # match[i,j] = all(left[i,t] == right[j,t] for t in range(k))
                    match = np.all(left[:, None, :] == right[None, :, :], axis=2)
                    # Only set where overlap_matrix is still 0 (we want max overlap)
                    overlap_matrix[match & (overlap_matrix == 0)] = k

                ov_list = overlap_matrix.tolist()

                # Determine number of restarts based on time remaining
                time_remaining = deadline - time.time() - 15
                if time_remaining < 5:
                    n_restarts = 10
                elif time_remaining < 15:
                    n_restarts = 50
                elif time_remaining < 30:
                    n_restarts = 150
                else:
                    n_restarts = 400

                best_path = None
                best_score = -1

                for r in range(n_restarts):
                    start = random.randint(0, N - 1)
                    path = [start]
                    visited = [False] * N
                    visited[start] = True
                    current = start
                    current_score = 0

                    for _ in range(N - 1):
                        best_next = -1
                        best_ov = -1
                        ov_row = ov_list[current]
                        for j in range(N):
                            if not visited[j]:
                                ov = ov_row[j]
                                if ov > best_ov:
                                    best_ov = ov
                                    best_next = j

                        if best_next == -1:
                            for j in range(N):
                                if not visited[j]:
                                    best_next = j
                                    break

                        visited[best_next] = True
                        path.append(best_next)
                        current_score += ov_list[current][best_next]
                        current = best_next

                    if current_score > best_score:
                        best_score = current_score
                        best_path = path

                    if time.time() > deadline - 20:
                        break

                # Local search polish: try 2-opt style swaps on the best path
                if best_path is not None and time.time() < deadline - 10:
                    improved = True
                    polish_iter = 0
                    max_polish_iters = 50
                    while improved and polish_iter < max_polish_iters:
                        improved = False
                        polish_iter += 1
                        # Compute current total overlap
                        cur_total = sum(ov_list[best_path[i]][best_path[i+1]] for i in range(len(best_path)-1))
                        
                        # Try random pairs of adjacent segments to swap
                        n_tries = min(200, N * 2)
                        for _ in range(n_tries):
                            if time.time() > deadline - 10:
                                break
                            # Pick two random positions i < j in the path
                            i = random.randint(0, len(best_path) - 2)
                            j = random.randint(i + 1, len(best_path) - 1)
                            # Swap segments: path[i..j] reversed? No, let's do a 2-opt:
                            # Reverse the segment between i and j
                            new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                            # Compute new total overlap
                            new_total = sum(ov_list[new_path[k]][new_path[k+1]] for k in range(len(new_path)-1))
                            if new_total > cur_total:
                                best_path = new_path
                                cur_total = new_total
                                improved = True
                        
                        if time.time() > deadline - 10:
                            break

                if best_path is not None:
                    ordered_perms = [perms[i] for i in best_path]
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
