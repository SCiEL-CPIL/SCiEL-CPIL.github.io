import math
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    moduli = list(range(2, 129))

    # Precompute gcds
    gcd_val = [[0] * 129 for _ in range(129)]
    for i in range(2, 129):
        for j in range(i, 129):
            gcd_val[i][j] = math.gcd(i, j)
            gcd_val[j][i] = gcd_val[i][j]

    # Precompute prime factors for each modulus (for potential use in heuristic ordering)
    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    # Heuristic ordering: sort by number of prime factors (descending) and then by value
    # Moduli with more prime factors have more conflicts, so placing them first might be harder?
    # Or placing them last?
    # Let's try sorting by "difficulty" - moduli with many divisors are more constrained.
    # Actually, let's just use the existing heuristic orders which worked well.

    def try_add(m, current_set):
        """
        current_set: list of (r, m) tuples.
        Returns a valid residue r for m if possible, else None.
        """
        if not current_set:
            return 0

        for r in range(m):
            valid = True
            for r_j, m_j in current_set:
                g = gcd_val[m][m_j]
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def greedy(order):
        current_set = []
        for m in order:
            r = try_add(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set

    best_set = []
    best_size = 0

    # Heuristic orders
    # 1. Reverse sorted (largest first)
    # 2. Sorted (smallest first)
    # 3. Sort by number of divisors (descending) - more constrained first?
    # 4. Sort by number of divisors (ascending) - less constrained first?
    # 5. Sort by prime factor count (descending)
    # 6. Sort by prime factor count (ascending)

    def count_divisors(n):
        return sum(1 for i in range(2, n + 1) if n % i == 0)

    def count_prime_factors(n):
        return len(prime_factors(n))

    heuristic_orders = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        sorted(moduli, key=lambda x: -count_divisors(x)),
        sorted(moduli, key=lambda x: count_divisors(x)),
        sorted(moduli, key=lambda x: -count_prime_factors(x)),
        sorted(moduli, key=lambda x: count_prime_factors(x)),
    ]

    # Run heuristic orders first
    for order in heuristic_orders:
        if time.time() > deadline:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts with random residue selection
    def try_add_random(m, current_set, rng):
        if not current_set:
            return 0
        valid_rs = []
        for r in range(m):
            valid = True
            for r_j, m_j in current_set:
                g = gcd_val[m][m_j]
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                valid_rs.append(r)
        if valid_rs:
            return rng.choice(valid_rs)
        return None

    def greedy_random(order, rng):
        current_set = []
        for m in order:
            r = try_add_random(m, current_set, rng)
            if r is not None:
                current_set.append((r, m))
        return current_set

    # Random restarts
    num_restarts = 20000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        random_order = moduli.copy()
        random.shuffle(random_order)

        # Alternate between deterministic and random residue selection
        if i % 2 == 0:
            s = greedy(random_order)
        else:
            s = greedy_random(random_order, random)

        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local search: try to insert skipped moduli by re-assigning residues
    # This is a simple local search that tries to add one modulus at a time
    # by checking if there's a valid assignment for it given the current set.
    # If not, it tries to swap residues of some existing moduli to make room.
    # For simplicity, we'll just try to insert and if it fails, move on.
    # A more sophisticated approach would involve backtracking on a subset of moduli.

    # Let's implement a simple local search that tries to improve the set
    # by removing one modulus and adding two others if possible.
    # But this might be complex. Let's stick to the greedy + random restarts.

    result = [[r, m] for r, m in best_set]

    if not result:
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return result
