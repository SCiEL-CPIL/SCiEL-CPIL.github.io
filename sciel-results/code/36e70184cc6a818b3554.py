import random
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 84.0  # Leave 6s margin for 90s limit

    N = 256

    # Precompute prime factors for each number 1..N
    prime_factors = [set() for _ in range(N + 1)]
    prime_factors[1] = set()
    for i in range(2, N + 1):
        x = i
        d = 2
        while d * d <= x:
            if x % d == 0:
                prime_factors[i].add(d)
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            prime_factors[i].add(x)

    # Check if number x is coprime to all in set S (using prime factor sets)
    def is_coprime_to_set(x, S):
        if x == 1:
            return True
        pf_x = prime_factors[x]
        for y in S:
            if pf_x & prime_factors[y]:
                return False
        return True

    all_numbers = list(range(1, N + 1))

    # Greedy initialization: sort by value descending, add if coprime
    def greedy_init():
        S = set()
        s_sum = 0
        for x in sorted(all_numbers, reverse=True):
            if is_coprime_to_set(x, S):
                S.add(x)
                s_sum += x
        return S, s_sum

    # Local search with 1-for-1 and 2-for-1 swaps
    def local_search(S, s_sum, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            s_list = list(S)
            candidates_out = sorted(s_list, key=lambda x: x)  # small first for better swaps
            candidates_in = sorted([y for y in all_numbers if y not in S], reverse=True)

            # 1-for-1 swap
            for x in candidates_out:
                if time.time() > deadline:
                    break
                for y in candidates_in:
                    if y <= x:
                        break
                    # Check if y is coprime to S \ {x}
                    valid = True
                    for z in s_list:
                        if z == x:
                            continue
                        if prime_factors[y] & prime_factors[z]:
                            valid = False
                            break
                    if valid:
                        S.discard(x)
                        S.add(y)
                        s_sum += y - x
                        improved = True
                        break
                if improved:
                    break

            # 2-for-1 swap (more expensive, do less frequently)
            if not improved and time.time() < deadline - 2:
                s_list = list(S)
                candidates_in2 = sorted([y for y in all_numbers if y not in S], reverse=True)
                for y in candidates_in2:
                    if time.time() > deadline:
                        break
                    for i in range(len(s_list)):
                        a = s_list[i]
                        for j in range(i + 1, len(s_list)):
                            b = s_list[j]
                            if y <= a + b:
                                continue
                            # Check if y is coprime to S \ {a, b}
                            valid = True
                            for z in s_list:
                                if z == a or z == b:
                                    continue
                                if prime_factors[y] & prime_factors[z]:
                                    valid = False
                                    break
                            if valid:
                                S.discard(a)
                                S.discard(b)
                                S.add(y)
                                s_sum += y - a - b
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        return S, s_sum

    # Start with greedy initialization
    current_set, current_sum = greedy_init()
    best_sum = current_sum
    best_set = set(current_set)

    # Local search from greedy
    current_set, current_sum = local_search(current_set, current_sum, time_limit - 10)
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)

    # Random restarts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 8:
        num_restarts = max(1, int(remaining_time / 5))
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 4:
                break

            # Perturb: start from best, remove 1-3 random elements, then greedy fill
            random_set = set(best_set)
            k = min(len(random_set), 1 + (restart % 3))
            to_remove = random.sample(list(random_set), k)
            for x in to_remove:
                random_set.discard(x)

            random_sum = sum(random_set)

            # Greedy fill: add any coprime number
            for x in sorted(all_numbers, reverse=True):
                if x not in random_set and is_coprime_to_set(x, random_set):
                    random_set.add(x)
                    random_sum += x

            # Local search
            random_set, random_sum = local_search(random_set, random_sum, time_limit - 3)

            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)

    # Final polish: one more local search pass on best
    remaining = time_limit - (time.time() - start_time)
    if remaining > 3:
        best_set, best_sum = local_search(best_set, best_sum, time_limit - 1)

    # Ensure types are correct
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
