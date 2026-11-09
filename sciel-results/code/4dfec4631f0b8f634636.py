"""Improved program: fast conflict detection via precomputed bad-triple table,
more diverse greedy orderings, and stronger local search (1-for-1 swaps, 2-for-1,
multi-pass add)."""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 128

    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            v = _lcm(i, j)
            lcm_table[i][j] = v
            lcm_table[j][i] = v

    # Precompute bad-triple table: bad[a][b] is a set of c (c != a, b) such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad[a].add(c)
                    bad[b].add(c)
                    bad[c].add(a)
                    bad[c].add(b)

    def is_conflict(x, selected):
        """Check if x conflicts with any pair in selected."""
        bx = bad[x]
        for i in range(len(selected)):
            a = selected[i]
            if a in bx:
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    if b in bx:
                        return True
        return False

    # Precompute useful properties for ordering
    def prime_exponents(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors

    exp_profiles = {}
    largest_pf = {}
    omega_count = {}
    radical = {}
    num_divs = {}

    for x in range(1, n + 1):
        pe = prime_exponents(x)
        exp_profiles[x] = pe
        if pe:
            largest_pf[x] = max(pe.keys())
            omega_count[x] = len(pe)
            r = 1
            for p in pe:
                r *= p
            radical[x] = r
        else:
            largest_pf[x] = 1
            omega_count[x] = 0
            radical[x] = 1

        # Number of divisors
        cnt = 0
        d = 1
        while d * d <= x:
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
            d += 1
        num_divs[x] = cnt

    # Generate diverse orderings
    import random
    random.seed(12345)

    orderings = []

    # 1. Descending by largest prime factor
    orderings.append(sorted(range(1, n + 1), key=lambda x: largest_pf[x], reverse=True))

    # 2. Ascending by omega (fewer prime factors)
    orderings.append(sorted(range(1, n + 1), key=lambda x: omega_count[x]))

    # 3. Descending by value
    orderings.append(list(range(n, 0, -1)))

    # 4. Ascending by value
    orderings.append(list(range(1, n + 1)))

    # 5. Ascending by number of divisors
    orderings.append(sorted(range(1, n + 1), key=lambda x: num_divs[x]))

    # 6. Descending by radical
    orderings.append(sorted(range(1, n + 1), key=lambda x: radical[x], reverse=True))

    # 7. Primes first, then composites
    def is_prime(x):
        if x < 2:
            return False
        d = 2
        while d * d <= x:
            if x % d == 0:
                return False
            d += 1
        return True
    primes = [x for x in range(1, n + 1) if is_prime(x)]
    non_primes = [x for x in range(1, n + 1) if not is_prime(x)]
    orderings.append(primes + non_primes)

    # 8. Prime powers first
    def is_prime_power(x):
        if x < 2:
            return False
        for p in range(2, int(x ** 0.5) + 1):
            if x % p == 0:
                while x % p == 0:
                    x //= p
                return x == 1
        return True
    prime_powers = [x for x in range(1, n + 1) if is_prime_power(x)]
    others = [x for x in range(1, n + 1) if not is_prime_power(x)]
    orderings.append(prime_powers + others)

    # 9. Descending by max exponent in factorization
    def max_exp(x):
        pe = exp_profiles[x]
        if not pe:
            return 0
        return max(pe.values())
    orderings.append(sorted(range(1, n + 1), key=max_exp, reverse=True))

    # 10-30: Randomized orderings with various biases
    for trial in range(21):
        perm = list(range(1, n + 1))
        noise = {x: random.random() for x in range(1, n + 1)}
        perm.sort(key=lambda x: (-largest_pf[x] + noise[x] * 10, random.random()))
        orderings.append(perm)

    for trial in range(10):
        perm = list(range(1, n + 1))
        noise = {x: random.random() for x in range(1, n + 1)}
        perm.sort(key=lambda x: (omega_count[x] + noise[x] * 5, random.random()))
        orderings.append(perm)

    for trial in range(10):
        perm = list(range(1, n + 1))
        noise = {x: random.random() for x in range(1, n + 1)}
        perm.sort(key=lambda x: (-radical[x] + noise[x] * 30, random.random()))
        orderings.append(perm)

    for trial in range(10):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)

    # Greedy construction
    best_result = None
    best_size = 0

    for ordering in orderings:
        if time.time() > deadline:
            break

        selected = []
        for x in ordering:
            if time.time() > deadline:
                break
            if not is_conflict(x, selected):
                selected.append(x)

        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
            if best_size >= n:
                break

    # Local search: multi-pass add
    if best_result and time.time() < deadline - 3.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if not is_conflict(x, current):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    # Local search: 1-for-1 swaps (try replacing one element with another to enable additions)
    if best_result and time.time() < deadline - 5.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                removed = current[i]
                temp = current[:i] + current[i + 1:]
                temp_set = set(temp)
                # Try to add one element
                for x in range(1, n + 1):
                    if x in temp_set or x == removed:
                        continue
                    if not is_conflict(x, temp):
                        temp.append(x)
                        temp_set.add(x)
                        # Now try to add more
                        added_more = True
                        while added_more and time.time() < deadline - 5.0:
                            added_more = False
                            for y in range(1, n + 1):
                                if y in temp_set:
                                    continue
                                if not is_conflict(y, temp):
                                    temp.append(y)
                                    temp_set.add(y)
                                    added_more = True
                                    break
                        if len(temp) > len(current):
                            current = temp
                            current_set = set(current)
                            improved = True
                        break
                if improved:
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    # Local search: 2-for-1 swaps
    if best_result and time.time() < deadline - 5.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                removed = current[i]
                temp = current[:i] + current[i + 1:]
                temp_set = set(temp)
                added = []
                for x in range(1, n + 1):
                    if x in temp_set or x == removed:
                        continue
                    if not is_conflict(x, temp):
                        temp.append(x)
                        temp_set.add(x)
                        added.append(x)
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    current = temp
                    current_set = set(current)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    # Final multi-pass add after all local search
    if best_result and time.time() < deadline - 1.0:
        current = list(best_result)
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for x in range(1, n + 1):
                if x in current_set:
                    continue
                if not is_conflict(x, current):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    break
        if len(current) > best_size:
            best_size = len(current)
            best_result = current

    if not best_result:
        best_result = [1]

    return best_result
