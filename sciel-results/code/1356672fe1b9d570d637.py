"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 128
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    def lcm_lookup(a, b):
        return lcm_table[a][b]

    def is_conflict(x, selected):
        for i in range(len(selected)):
            a = selected[i]
            lab = lcm_lookup(a, x)
            for j in range(i + 1, len(selected)):
                b = selected[j]
                lac = lcm_lookup(a, b)
                lbc = lcm_lookup(b, x)
                if lab == lac == lbc:
                    return True
        return False

    def prime_exponents(x):
        """Return dict of prime -> exponent for x."""
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

    def omega(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    def largest_prime_factor(x):
        max_p = 1
        d = 2
        while d * d <= x:
            while x % d == 0:
                max_p = d
                x //= d
            d += 1
        if x > 1:
            max_p = x
        return max_p

    # Precompute prime exponent profiles
    exp_profiles = {}
    for x in range(1, n + 1):
        exp_profiles[x] = prime_exponents(x)

    def profile_distance(a, b):
        """Count primes where max exponent differs (rough conflict indicator)."""
        pa = exp_profiles[a]
        pb = exp_profiles[b]
        all_primes = set(pa.keys()) | set(pb.keys())
        dist = 0
        for p in all_primes:
            ea = pa.get(p, 0)
            eb = pb.get(p, 0)
            if max(ea, eb) != ea and max(ea, eb) != eb:
                pass  # always true
            # The key: if ea == eb and ea > 0, they share that exponent
            if ea == eb and ea > 0:
                dist += 1
        return dist

    # Better ordering: prioritize numbers with large prime factors (fewer shared exponents)
    def conflict_potential(x, selected):
        """Estimate how many conflicts x would create with selected."""
        count = 0
        px = exp_profiles[x]
        for a in selected:
            pa = exp_profiles[a]
            # Check if there exists b in selected such that (a,b,x) is bad
            # Simplified: count pairs where x shares exponent pattern
            for b in selected:
                if b == a:
                    continue
                pb = exp_profiles[b]
                all_p = set(px.keys()) | set(pa.keys()) | set(pb.keys())
                bad = False
                for p in all_p:
                    ex = px.get(p, 0)
                    ea = pa.get(p, 0)
                    eb = pb.get(p, 0)
                    mx = max(ex, ea, eb)
                    # Count how many equal mx
                    cnt = 0
                    if ex == mx:
                        cnt += 1
                    if ea == mx:
                        cnt += 1
                    if eb == mx:
                        cnt += 1
                    if cnt >= 2:
                        # At least two have the max exponent - potential conflict
                        # But need exactly: at least two equal to mx and the third <= mx
                        # This is always true if cnt >= 2
                        bad = True
                        break
                if bad:
                    count += 1
        return count

    best_result = None
    best_size = 0

    random.seed(42)

    # Generate many orderings
    orderings = []

    # 1. Descending by largest prime factor (numbers with big primes are safer)
    orderings.append(sorted(range(1, n + 1), key=largest_prime_factor, reverse=True))

    # 2. Ascending by omega (fewer prime factors = safer)
    orderings.append(sorted(range(1, n + 1), key=omega))

    # 3. Descending by value (larger numbers tend to have larger prime factors)
    orderings.append(list(range(n, 0, -1)))

    # 4. Ascending by value
    orderings.append(list(range(1, n + 1)))

    # 5. Sort by number of divisors (fewer divisors = more "prime-like")
    def num_divisors(x):
        cnt = 0
        for d in range(1, int(x**0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        return cnt
    orderings.append(sorted(range(1, n + 1), key=num_divisors))

    # 6. Sort by radicial (product of distinct primes)
    def radicial(x):
        r = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                r *= d
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            r *= x
        return r
    orderings.append(sorted(range(1, n + 1), key=radicial, reverse=True))

    # 7. Primes and prime powers first
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

    # 8. Numbers with exactly one prime factor (prime powers) first
    def is_prime_power(x):
        if x < 2:
            return False
        for p in range(2, int(x**0.5) + 1):
            if x % p == 0:
                while x % p == 0:
                    x //= p
                return x == 1
        return True  # x itself is prime
    prime_powers = [x for x in range(1, n + 1) if is_prime_power(x)]
    others = [x for x in range(1, n + 1) if not is_prime_power(x)]
    orderings.append(prime_powers + others)

    # 9. Sort by "exponent spread" - numbers with high exponents in one prime are safer
    def exponent_spread(x):
        pe = exp_profiles[x]
        if not pe:
            return 0
        exponents = list(pe.values())
        return max(exponents) - min(exponents) if len(exponents) > 1 else max(exponents)
    orderings.append(sorted(range(1, n + 1), key=exponent_spread, reverse=True))

    # 10-59: Random orderings
    for _ in range(50):
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        orderings.append(perm)

    # 60-69: Orderings biased toward numbers with large prime factors
    for _ in range(10):
        weights = [largest_prime_factor(x) ** 2 for x in range(1, n + 1)]
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        # Sort by a mix of largest prime factor and random
        orderings.append(sorted(perm, key=lambda x: -largest_prime_factor(x) + random.random() * 10))

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

    # Local search: try to add more elements
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

    # Local search with swaps: try replacing one element with two
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
                temp = current[:i] + current[i+1:]
                temp_set = set(temp)
                # Try to add two elements
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

    if not best_result:
        best_result = [1]

    return best_result
# EVOLVE-BLOCK-END
