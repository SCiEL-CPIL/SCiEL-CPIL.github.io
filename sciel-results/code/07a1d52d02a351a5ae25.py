import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    N = 128
    TARGET_MIN = 128
    TARGET_MAX = 256
    TARGET_SIZE = TARGET_MAX - TARGET_MIN + 1

    def get_cost(sums):
        target_sums = sums[TARGET_MIN:TARGET_MAX + 1]
        gaps = np.sum(target_sums == 0)
        if gaps > 0:
            return 1000000 + gaps * 10000 + np.max(target_sums) * 100
        peak = int(np.max(target_sums))
        # Count how many targets have multiplicity == peak (bottleneck width)
        bottleneck_width = int(np.sum(target_sums == peak))
        sq = int(np.sum(target_sums ** 2))
        return peak * 100000 + bottleneck_width * 1000 + sq

    def apply_add(A, sums, x):
        for a in A:
            s = a + x
            if s <= 256:
                sums[s] += 1
        if 2 * x <= 256:
            sums[2 * x] += 1
        A.add(x)

    def apply_remove(A, sums, x):
        for a in A:
            if a == x:
                continue
            s = a + x
            if s <= 256:
                sums[s] -= 1
        if 2 * x <= 256:
            sums[2 * x] -= 1
        A.remove(x)

    def calculate_sums(A):
        sums = np.zeros(257, dtype=np.int32)
        A_list = list(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if s <= 256:
                    sums[s] += 1
        return sums

    def greedy_init():
        A = set()
        sums = np.zeros(257, dtype=np.int32)

        apply_add(A, sums, 128)
        apply_add(A, sums, 0)

        candidates = [x for x in range(1, 128)]
        random.shuffle(candidates)

        for _ in range(15):
            best_cand = None
            best_cost = float('inf')

            for x in candidates:
                apply_add(A, sums, x)
                c = get_cost(sums)
                apply_remove(A, sums, x)

                if c < best_cost:
                    best_cost = c
                    best_cand = x

            if best_cand is None:
                break

            apply_add(A, sums, best_cand)
            candidates.remove(best_cand)

            if best_cost < 20000:
                break

        return A, sums

    def polish(A, sums, max_iters=1000, time_limit=5.0):
        start_t = time.time()
        current_cost = get_cost(sums)
        improved = True

        while improved and time.time() - start_t < time_limit and max_iters > 0:
            improved = False
            max_iters -= 1

            A_list = list(A)
            random.shuffle(A_list)

            for x in A_list:
                if len(A) <= 1:
                    continue
                if x == 128:
                    continue

                apply_remove(A, sums, x)
                new_cost = get_cost(sums)

                if new_cost <= current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    apply_add(A, sums, x)

            if time.time() - start_t >= time_limit:
                break

            A_list = list(A)
            random.shuffle(A_list)

            for x in A_list:
                if x not in A:
                    continue
                if x == 128:
                    continue

                candidates = [y for y in range(N + 1) if y not in A and y != 128]
                if not candidates:
                    continue

                random.shuffle(candidates)
                for y in candidates[:10]:
                    apply_remove(A, sums, x)
                    apply_add(A, sums, y)

                    new_cost = get_cost(sums)

                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        x = y
                        break
                    else:
                        apply_remove(A, sums, y)
                        apply_add(A, sums, x)

                    if time.time() - start_t >= time_limit:
                        break

        return A, sums, current_cost

    best_A = None
    best_cost = float('inf')
    best_sums = None

    # 1. Greedy Initialization
    A_greedy, sums_greedy = greedy_init()
    cost_greedy = get_cost(sums_greedy)

    if cost_greedy < best_cost:
        best_cost = cost_greedy
        best_A = set(A_greedy)
        best_sums = sums_greedy.copy()

    # Polish the greedy result
    A_polished, sums_polished, cost_polished = polish(set(A_greedy), sums_greedy.copy())
    if cost_polished < best_cost:
        best_cost = cost_polished
        best_A = set(A_polished)
        best_sums = sums_polished.copy()

    # 2. Simulated Annealing / Random Restarts
    num_restarts = 8
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break

        size = random.randint(15, 30)
        A = set(random.sample(range(N + 1), size))

        if 128 not in A:
            A.remove(random.choice(list(A)))
            A.add(128)

        sums = calculate_sums(A)
        current_cost = get_cost(sums)

        if current_cost < best_cost:
            best_cost = current_cost
            best_A = set(A)
            best_sums = sums.copy()

        T = 50.0
        T_min = 0.1
        cooling = 0.99
        steps = 5000

        for step in range(steps):
            if time.time() > deadline - 5:
                break

            move_type = random.random()

            if move_type < 0.1:
                candidates = [x for x in range(N + 1) if x not in A]
                if not candidates:
                    continue
                x = random.choice(candidates)

                apply_add(A, sums, x)
                new_cost = get_cost(sums)

                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, x)

            elif move_type < 0.2:
                if len(A) < 10:
                    continue
                x = random.choice(list(A))
                if x == 128:
                    continue

                apply_remove(A, sums, x)
                new_cost = get_cost(sums)

                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_add(A, sums, x)

            elif move_type < 0.35:
                # 2-swap: remove two elements, add two new ones
                if len(A) < 2:
                    continue
                A_list = list(A)
                non128 = [a for a in A_list if a != 128]
                if len(non128) < 2:
                    continue
                x1, x2 = random.sample(non128, 2)
                candidates = [y for y in range(N + 1) if y not in A and y != 128]
                if len(candidates) < 2:
                    continue
                y1, y2 = random.sample(candidates, 2)

                apply_remove(A, sums, x1)
                apply_remove(A, sums, x2)
                apply_add(A, sums, y1)
                apply_add(A, sums, y2)

                new_cost = get_cost(sums)

                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, y1)
                    apply_remove(A, sums, y2)
                    apply_add(A, sums, x1)
                    apply_add(A, sums, x2)

            else:
                if len(A) == 0:
                    continue
                x = random.choice(list(A))
                if x == 128:
                    others = [a for a in A if a != 128]
                    if not others:
                        continue
                    x = random.choice(others)

                candidates = [y for y in range(N + 1) if y not in A and y != 128]
                if not candidates:
                    continue
                y = random.choice(candidates)

                apply_remove(A, sums, x)
                apply_add(A, sums, y)

                new_cost = get_cost(sums)

                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                        best_sums = sums.copy()
                else:
                    apply_remove(A, sums, y)
                    apply_add(A, sums, x)

            T *= cooling
            if T < T_min:
                break

        # Polish after SA
        if time.time() < deadline - 5:
            A_pol, sums_pol, cost_pol = polish(set(A), sums.copy(), max_iters=200, time_limit=2.0)
            if cost_pol < best_cost:
                best_cost = cost_pol
                best_A = set(A_pol)
                best_sums = sums_pol.copy()

    # Final high-precision polish on the best solution found
    if time.time() < deadline - 2 and best_A is not None:
        A_final, sums_final, cost_final = polish(set(best_A), best_sums.copy(), max_iters=500, time_limit=min(5.0, max(0.5, deadline - time.time() - 1)))
        if cost_final < best_cost:
            best_cost = cost_final
            best_A = set(A_final)
            best_sums = sums_final.copy()

    if best_A is None:
        return [128, 0, 64, 32, 16, 8, 4, 2, 1]

    return sorted(best_A)
