import math
import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

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

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(N)
    all_numbers = list(range(1, N + 1))

    # Precompute a bitmask for each number's prime factors for fast intersection checks
    # First, map each prime to an index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    prime_mask = [0] * (N + 1)
    prime_mask[1] = 0
    for i in range(2, N + 1):
        mask = 0
        for p in prime_factors[i]:
            mask |= (1 << prime_to_idx[p])
        prime_mask[i] = mask

    def is_coprime_to_mask(x, used_mask):
        if x == 1:
            return True
        return (prime_mask[x] & used_mask) == 0

    def mask_to_set(used_mask):
        """Reconstruct which primes are used from the mask."""
        result = set()
        for p in primes:
            if used_mask & (1 << prime_to_idx[p]):
                result.add(p)
        return result

    # Product move: replace subset of primes {p1,...,pk} with their product P
    # if P <= N, P is coprime to S\subset, and P > sum(subset)
    def try_improve_product_moves(S, s_sum, used_mask):
        prims_in_set = [p for p in S if p in primes]
        n = len(prims_in_set)
        if n == 0:
            return False, s_sum, used_mask

        # Size 2
        for i in range(n):
            for j in range(i+1, n):
                p1, p2 = prims_in_set[i], prims_in_set[j]
                prod = p1 * p2
                if prod > N:
                    continue
                # Check if prod is coprime to rest of S
                rest_mask = used_mask & ~(prime_mask[p1] | prime_mask[p2])
                if is_coprime_to_mask(prod, rest_mask):
                    gain = prod - (p1 + p2)
                    if gain > 0:
                        S.discard(p1)
                        S.discard(p2)
                        S.add(prod)
                        new_mask = rest_mask | prime_mask[prod]
                        return True, s_sum + gain, new_mask

        # Size 3
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = prims_in_set[i], prims_in_set[j], prims_in_set[k]
                    prod = p1 * p2 * p3
                    if prod > N:
                        continue
                    rest_mask = used_mask & ~(prime_mask[p1] | prime_mask[p2] | prime_mask[p3])
                    if is_coprime_to_mask(prod, rest_mask):
                        gain = prod - (p1 + p2 + p3)
                        if gain > 0:
                            S.discard(p1)
                            S.discard(p2)
                            S.discard(p3)
                            S.add(prod)
                            new_mask = rest_mask | prime_mask[prod]
                            return True, s_sum + gain, new_mask

        # Size 4
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = prims_in_set[i], prims_in_set[j], prims_in_set[k], prims_in_set[l]
                        prod = p1 * p2 * p3 * p4
                        if prod > N:
                            continue
                        rest_mask = used_mask & ~(prime_mask[p1] | prime_mask[p2] | prime_mask[p3] | prime_mask[p4])
                        if is_coprime_to_mask(prod, rest_mask):
                            gain = prod - (p1 + p2 + p3 + p4)
                            if gain > 0:
                                S.discard(p1)
                                S.discard(p2)
                                S.discard(p3)
                                S.discard(p4)
                                S.add(prod)
                                new_mask = rest_mask | prime_mask[prod]
                                return True, s_sum + gain, new_mask

        return False, s_sum, used_mask

    def try_add(S, s_sum, used_mask, x):
        if x in S:
            return S, s_sum, used_mask, False
        if is_coprime_to_mask(x, used_mask):
            S.add(x)
            return S, s_sum + x, used_mask | prime_mask[x], True
        return S, s_sum, used_mask, False

    def try_swap(S, s_sum, used_mask, out_x, in_y):
        if out_x not in S or in_y in S or in_y == out_x:
            return S, s_sum, used_mask, False
        temp_mask = used_mask & ~prime_mask[out_x]
        if is_coprime_to_mask(in_y, temp_mask):
            S.discard(out_x)
            S.add(in_y)
            new_mask = temp_mask | prime_mask[in_y]
            return S, s_sum - out_x + in_y, new_mask, True
        return S, s_sum, used_mask, False

    def local_search(S, s_sum, used_mask, max_time):
        # Phase A: Product moves
        while time.time() - start_time < max_time:
            improved, s_sum, used_mask = try_improve_product_moves(S, s_sum, used_mask)
            if not improved:
                break

        # Phase B: Greedy add - try adding numbers in descending order of value
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for x in sorted(all_numbers, reverse=True):
                if x not in S:
                    S, s_sum, used_mask, did_add = try_add(S, s_sum, used_mask, x)
                    if did_add:
                        improved = True
                        break

        # Phase C: Single swap - try replacing smaller elements with larger ones
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            s_list = list(S)
            candidates_out = sorted(s_list, key=lambda x: x)
            candidates_in = [y for y in all_numbers if y not in S]
            candidates_in.sort(reverse=True)
            for x in candidates_out:
                if time.time() - start_time > max_time - 1:
                    break
                for y in candidates_in:
                    if y <= x:
                        break
                    S, s_sum, used_mask, did_swap = try_swap(S, s_sum, used_mask, x, y)
                    if did_swap:
                        improved = True
                        break
                if improved:
                    break

        # Phase D: Multi-swap (replace 2 with 1, or 3 with 1)
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            s_list = list(S)
            # Replace 2 elements with 1
            for i in range(len(s_list)):
                if time.time() - start_time > max_time - 1:
                    break
                for j in range(i + 1, len(s_list)):
                    x1, x2 = s_list[i], s_list[j]
                    if x1 + x2 >= N:
                        continue
                    temp_mask = used_mask & ~(prime_mask[x1] | prime_mask[x2])
                    for y in all_numbers:
                        if y in S or y <= x1 + x2:
                            continue
                        if is_coprime_to_mask(y, temp_mask):
                            S.discard(x1)
                            S.discard(x2)
                            S.add(y)
                            s_sum += y - x1 - x2
                            used_mask = temp_mask | prime_mask[y]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Replace 3 elements with 1
            if not improved:
                s_list = list(S)
                for i in range(len(s_list)):
                    if time.time() - start_time > max_time - 1:
                        break
                    for j in range(i + 1, len(s_list)):
                        for k in range(j + 1, len(s_list)):
                            x1, x2, x3 = s_list[i], s_list[j], s_list[k]
                            if x1 + x2 + x3 >= N:
                                continue
                            temp_mask = used_mask & ~(prime_mask[x1] | prime_mask[x2] | prime_mask[x3])
                            for y in all_numbers:
                                if y in S or y <= x1 + x2 + x3:
                                    continue
                                if is_coprime_to_mask(y, temp_mask):
                                    S.discard(x1)
                                    S.discard(x2)
                                    S.discard(x3)
                                    S.add(y)
                                    s_sum += y - x1 - x2 - x3
                                    used_mask = temp_mask | prime_mask[y]
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return S, s_sum, used_mask

    # Initial solution: start with large numbers that have few prime factors
    # Sort numbers by (number of prime factors, -value) to prioritize large numbers with few factors
    def num_prime_factors(x):
        return len(prime_factors[x]) if x > 1 else 0

    sorted_numbers = sorted(all_numbers, key=lambda x: (num_prime_factors(x), -x))

    current_set = set()
    current_sum = 0
    current_mask = 0

    for x in sorted_numbers:
        if is_coprime_to_mask(x, current_mask):
            current_set.add(x)
            current_sum += x
            current_mask |= prime_mask[x]

    # Run local search on initial solution
    current_set, current_sum, current_mask = local_search(current_set, current_sum, current_mask, time_limit - 5)

    best_sum = current_sum
    best_set = set(current_set)
    best_mask = current_mask

    # Random restarts with local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        num_restarts = max(1, int(remaining_time / 5))
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit - 5:
                break

            # Create a perturbed solution
            random_set = set()
            random_sum = 0
            random_mask = 0

            # Start with a random subset, biased toward larger numbers
            shuffled_numbers = all_numbers[:]
            random.shuffle(shuffled_numbers)
            # Bias: try larger numbers first
            shuffled_numbers.sort(key=lambda x: (-x, random.random()))

            for x in shuffled_numbers:
                if is_coprime_to_mask(x, random_mask):
                    random_set.add(x)
                    random_sum += x
                    random_mask |= prime_mask[x]

            # Run local search
            random_set, random_sum, random_mask = local_search(random_set, random_sum, random_mask, time_limit - 3)

            # Update best
            if random_sum > best_sum:
                best_sum = random_sum
                best_set = set(random_set)
                best_mask = random_mask

    current_set = best_set

    # Final validation and type check
    result = list(current_set)
    result = [int(x) for x in result]
    return result
