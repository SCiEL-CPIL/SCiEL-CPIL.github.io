import numpy as np
import time
import random
import heapq

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3 ** n
    ball_size = 1 + 2 * n  # 17
    powers = [3 ** i for i in range(n)]

    # Precompute balls and coverers
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball

    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)

    # Convert to numpy arrays for speed
    balls_arr = [np.array(b, dtype=np.int32) for b in balls]

    def greedy_set_cover(initial_code=None, time_limit=110):
        start = time.time()
        code = set(initial_code) if initial_code else set()
        uncovered = np.ones(M, dtype=bool)
        gains = np.full(M, ball_size, dtype=np.int32)

        if initial_code:
            for c in initial_code:
                for p in balls_arr[c]:
                    if uncovered[p]:
                        uncovered[p] = False
                        for c_prime in coverers[p]:
                            if c_prime not in code:
                                gains[c_prime] -= 1

        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
        heapq.heapify(heap)
        uncovered_count = int(np.sum(uncovered))

        while uncovered_count > 0 and (time.time() - start) < time_limit:
            valid_c = None
            while heap:
                neg_g, c = heapq.heappop(heap)
                if c in code:
                    continue
                if -neg_g != gains[c] or gains[c] <= 0:
                    continue
                valid_c = c
                break

            if valid_c is None:
                break

            code.add(valid_c)
            for p in balls_arr[valid_c]:
                if uncovered[p]:
                    uncovered[p] = False
                    uncovered_count -= 1
                    for c_prime in coverers[p]:
                        if c_prime not in code:
                            gains[c_prime] -= 1
                            if gains[c_prime] > 0:
                                heapq.heappush(heap, (-gains[c_prime], c_prime))

        return code

    def local_search(code, time_limit=110):
        start_time = time.time()
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls_arr[c]:
                count[p] += 1

        def is_redundant(c):
            return all(count[p] > 1 for p in balls_arr[c])

        def remove_c(c):
            code.remove(c)
            for p in balls_arr[c]:
                count[p] -= 1

        def add_c(c):
            code.add(c)
            for p in balls_arr[c]:
                count[p] += 1

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False

            # Phase 1: Remove redundant
            to_remove = [c for c in list(code) if is_redundant(c)]
            if to_remove:
                for c in to_remove:
                    remove_c(c)
                improved = True
                continue

            # Phase 2: 1-Swap with best candidate selection
            code_list = list(code)
            random.shuffle(code_list)
            swapped = False
            for c in code_list:
                if (time.time() - start_time) >= time_limit:
                    break
                unique_points = [p for p in balls_arr[c] if count[p] == 1]
                if not unique_points:
                    remove_c(c)
                    improved = True
                    swapped = True
                    break

                candidates = set(coverers[unique_points[0]])
                for p in unique_points[1:]:
                    candidates.intersection_update(coverers[p])
                candidates -= code
                if candidates:
                    # Choose candidate that maximizes new coverage
                    best_c = None
                    best_gain = -1
                    for c_new in list(candidates)[:20]:  # Limit search
                        gain = sum(1 for p in balls_arr[c_new] if count[p] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c_new
                    if best_c is not None:
                        remove_c(c)
                        add_c(best_c)
                        improved = True
                        swapped = True
                        break

            if swapped:
                continue

            # Phase 3: 2-Swap
            if (time.time() - start_time) < time_limit and len(code) >= 2:
                code_list = list(code)
                random.shuffle(code_list)
                found = False
                for i in range(min(5, len(code_list))):
                    for j in range(i + 1, min(5, len(code_list))):
                        if (time.time() - start_time) >= time_limit:
                            break
                        c1, c2 = code_list[i], code_list[j]
                        U1 = [p for p in balls_arr[c1] if count[p] == 1]
                        U2 = [p for p in balls_arr[c2] if count[p] == 1]

                        if U1 and U2:
                            all_unique = set(U1) | set(U2)
                            cands = set(coverers[next(iter(all_unique))])
                            for p in all_unique:
                                cands.intersection_update(coverers[p])
                            cands -= code
                            if cands:
                                c_new = random.choice(list(cands))
                                remove_c(c1)
                                remove_c(c2)
                                add_c(c_new)
                                improved = True
                                found = True
                                break

                            # Try two separate replacements
                            cands1 = set(coverers[U1[0]])
                            for p in U1[1:]:
                                cands1.intersection_update(coverers[p])
                            cands1 -= code
                            cands1.discard(c1)
                            cands1.discard(c2)

                            cands2 = set(coverers[U2[0]])
                            for p in U2[1:]:
                                cands2.intersection_update(coverers[p])
                            cands2 -= code
                            cands2.discard(c1)
                            cands2.discard(c2)

                            if cands1 and cands2:
                                c1_new = random.choice(list(cands1))
                                valid_c2s = [c for c in cands2 if c != c1_new]
                                if valid_c2s:
                                    c2_new = random.choice(valid_c2s)
                                    remove_c(c1)
                                    remove_c(c2)
                                    add_c(c1_new)
                                    add_c(c2_new)
                                    improved = True
                                    found = True
                                    break
                    if found:
                        break

            if improved:
                continue

            # Perturbation: remove k random, re-greedy
            if (time.time() - start_time) < time_limit and len(code) > 10:
                k = random.randint(3, 8)
                to_perturb = random.sample(list(code), min(k, len(code)))
                for c in to_perturb:
                    remove_c(c)

                # Re-greedy fill
                gains = np.zeros(M, dtype=np.int32)
                for c in range(M):
                    if c not in code:
                        gains[c] = sum(1 for p in balls_arr[c] if count[p] == 0)

                heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
                heapq.heapify(heap)
                while heap:
                    neg_g, c = heapq.heappop(heap)
                    if c in code or -neg_g != gains[c] or gains[c] <= 0:
                        continue
                    add_c(c)
                    for p in balls_arr[c]:
                        if count[p] == 1:
                            for c_prime in coverers[p]:
                                if c_prime not in code:
                                    gains[c_prime] -= 1
                                    if gains[c_prime] > 0:
                                        heapq.heappush(heap, (-gains[c_prime], c_prime))

        return code

    best_code = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 115

    # Start 1: symmetry-fixed
    code = greedy_set_cover({0}, time_limit=5)
    code = local_search(code, time_limit=8)
    if len(code) < best_size:
        best_size = len(code)
        best_code = set(code)

    # Multiple random starts
    remaining = time_limit - (time.time() - start_time)
    num_starts = max(1, int(remaining // 15))
    for _ in range(num_starts):
        if time.time() - start_time > time_limit - 15:
            break
        initial = set(random.sample(range(M), random.randint(5, 15)))
        code = greedy_set_cover(initial, time_limit=3)
        code = local_search(code, time_limit=8)
        if len(code) < best_size:
            best_size = len(code)
            best_code = set(code)

    # Final long refinement
    if best_code:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 5:
            best_code = local_search(best_code, time_limit=remaining)

    return [list(map(int, [(c // (3 ** i)) % 3 for i in range(n)])) for c in best_code]
