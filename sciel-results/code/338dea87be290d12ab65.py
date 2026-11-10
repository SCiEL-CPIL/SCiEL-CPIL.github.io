import time
import random
import bisect

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin

    # Pre-compute primes
    primes = [x for x in range(2, 129) if all(x % i for i in range(2, int(x**0.5) + 1))]

    def is_square_free(n):
        i = 2
        while i * i <= n:
            if n % (i * i) == 0:
                return False
            i += 1
        return True

    # Categorize numbers
    all_nums = list(range(2, 129))
    primes_set = set(primes)
    non_sqfree = [x for x in all_nums if x not in primes_set and not is_square_free(x)]
    sqfree_comp = [x for x in all_nums if x not in primes_set and is_square_free(x)]

    def count_distinct_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    def get_heuristic_score(x):
        if not is_square_free(x):
            return 0
        # Prefer numbers with fewer distinct prime factors (less collision risk)
        return -count_distinct_prime_factors(x)

    candidates = non_sqfree + sqfree_comp
    candidates.sort(key=lambda x: (get_heuristic_score(x), x))

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True

    def is_valid_insertion(S, x):
        idx = bisect.bisect_left(S, x)
        new_S = S[:idx] + [x] + S[idx:]
        return is_valid(new_S)

    # 1. Initialize with Primes
    S = primes[:]
    S.sort()

    best_S = S[:]
    best_len = len(S)

    # 2. Greedy Insertion of Candidates
    for x in candidates:
        if time.time() - start_time > limit:
            break
        if is_valid_insertion(S, x):
            idx = bisect.bisect_left(S, x)
            S.insert(idx, x)
            if len(S) > best_len:
                best_S = S[:]
                best_len = len(S)

    # 3. Local Search with various moves
    stuck_count = 0
    max_stuck = 50

    while time.time() - start_time < limit:
        improved = False

        # 1-out/1-in swap move (NEW)
        sf_in_S = [x for x in S if x not in primes_set and is_square_free(x)]
        if not sf_in_S:
            sf_in_S = [x for x in S if x not in primes_set]

        if sf_in_S:
            y = random.choice(sf_in_S)
            S_without_y = [v for v in S if v != y]

            missing = [x for x in all_nums if x not in S_without_y and x != y]
            missing.sort(key=lambda x: (get_heuristic_score(x), x))

            for x in missing:
                if is_valid_insertion(S_without_y, x):
                    idx = bisect.bisect_left(S_without_y, x)
                    new_S = S_without_y[:idx] + [x] + S_without_y[idx:]
                    if len(new_S) > len(S):
                        S = new_S
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        improved = True
                        break

        # 1-out/2-in move
        if not improved and sf_in_S:
            y = random.choice(sf_in_S)
            S_without_y = [v for v in S if v != y]

            missing = [x for x in all_nums if x not in S_without_y and x != y]
            missing.sort(key=lambda x: (get_heuristic_score(x), x))

            added_count = 0
            new_S = S_without_y[:]
            for x in missing:
                if added_count >= 2:
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added_count += 1

            if added_count >= 2:
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_len:
                        best_S = S[:]
                        best_len = len(S)
                    improved = True

        # 2-out/3-in move (try removing two square-free composites)
        if not improved and len(sf_in_S) >= 2:
            y1, y2 = random.sample(sf_in_S, 2)
            S_without = [v for v in S if v != y1 and v != y2]

            missing = [x for x in all_nums if x not in S_without and x != y1 and x != y2]
            missing.sort(key=lambda x: (get_heuristic_score(x), x))

            added_count = 0
            new_S = S_without[:]
            for x in missing:
                if added_count >= 3:
                    break
                if is_valid_insertion(new_S, x):
                    idx = bisect.bisect_left(new_S, x)
                    new_S.insert(idx, x)
                    added_count += 1

            if added_count >= 3:
                if is_valid(new_S):
                    S = new_S
                    if len(S) > best_len:
                        best_S = S[:]
                        best_len = len(S)
                    improved = True

        # Try random 1-in insertion
        if not improved:
            missing = [x for x in all_nums if x not in S]
            if missing:
                random.shuffle(missing)
                for x in missing:
                    if time.time() - start_time > limit:
                        break
                    if is_valid_insertion(S, x):
                        idx = bisect.bisect_left(S, x)
                        S.insert(idx, x)
                        if len(S) > best_len:
                            best_S = S[:]
                            best_len = len(S)
                        improved = True
                        break

        # Track stuck state and restart if needed (NEW)
        if not improved:
            stuck_count += 1
            if stuck_count >= max_stuck:
                # Restart from best solution with random perturbation
                S = best_S[:]
                # Randomly remove 1-2 non-prime elements
                non_prime_in_S = [x for x in S if x not in primes_set]
                if non_prime_in_S:
                    remove_count = min(random.randint(1, 2), len(non_prime_in_S))
                    to_remove = random.sample(non_prime_in_S, remove_count)
                    S = [x for x in S if x not in to_remove]
                stuck_count = 0
        else:
            stuck_count = 0

    return best_S
