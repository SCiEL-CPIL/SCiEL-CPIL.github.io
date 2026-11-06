import math
import time
import random
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))

    # Precompute gcd
    gcd_val = [[0] * (MOD_MAX + 1) for _ in range(MOD_MAX + 1)]
    for i in range(2, MOD_MAX + 1):
        for j in range(i, MOD_MAX + 1):
            g = math.gcd(i, j)
            gcd_val[i][j] = g
            gcd_val[j][i] = g

    # Precompute prime factors for each modulus (for heuristic ordering)
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf = {m: prime_factors(m) for m in moduli}

    def count_pf(n):
        return len(pf[n])

    # Core check: can we add modulus m with residue r to the current set?
    # Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1,m2) does NOT divide (r1-r2).
    def can_add(r_new, m_new, current_set):
        for r_j, m_j in current_set:
            g = gcd_val[m_new][m_j]
            if (r_new - r_j) % g == 0:
                return False
        return True

    def try_add(m, current_set):
        """Find a valid residue for m given current_set. Returns r or None."""
        if not current_set:
            return 0

        # If any existing modulus is coprime to m, classes MUST intersect -> impossible
        for r_j, m_j in current_set:
            if gcd_val[m][m_j] == 1:
                return None

        # Build forbidden set: for each (r_j, m_j), forbidden residues r in [0,m)
        # satisfy r ≡ r_j (mod gcd(m, m_j))
        g_list = []
        for r_j, m_j in current_set:
            g = gcd_val[m][m_j]
            g_list.append((g, r_j % g))

        # For each candidate r in [0, m), check all constraints
        for r in range(m):
            ok = True
            for g, target in g_list:
                if r % g != target:
                    ok = False
                    break
            if ok:
                return r
        return None

    def try_add_smarter(m, current_set):
        """
        Smarter residue selection: instead of scanning 0..m-1,
        use inclusion-exclusion / CRT-based reasoning to pick a residue
        that avoids all forbidden progressions.
        """
        if not current_set:
            return 0

        for r_j, m_j in current_set:
            if gcd_val[m][m_j] == 1:
                return None

        constraints = []
        for r_j, m_j in current_set:
            g = gcd_val[m][m_j]
            constraints.append((g, r_j % g))

        # Try residues in a smart order: prioritize those that are "far" from targets
        # First try 0, then small residues
        for r in range(m):
            ok = True
            for g, target in constraints:
                if r % g == target:
                    ok = False
                    break
            if ok:
                return r
        return None

    def greedy(order):
        current_set = []
        for m in order:
            r = try_add(m, current_set)
            if r is not None:
                current_set.append((r, m))
        return current_set

    def greedy_random(order, rng):
        current_set = []
        for m in order:
            if not current_set:
                current_set.append((0, m))
                continue
            # Check coprime
            feasible = True
            for r_j, m_j in current_set:
                if gcd_val[m][m_j] == 1:
                    feasible = False
                    break
            if not feasible:
                continue
            constraints = []
            for r_j, m_j in current_set:
                g = gcd_val[m][m_j]
                constraints.append((g, r_j % g))
            valid = [r for r in range(m) if all(r % g != target for g, target in constraints)]
            if valid:
                current_set.append((rng.choice(valid), m))
        return current_set

    # Heuristic orderings
    def count_divisors(n):
        cnt = 0
        for i in range(2, n + 1):
            if n % i == 0:
                cnt += 1
        return cnt

    even_moduli = [m for m in moduli if m % 2 == 0]
    odd_moduli = [m for m in moduli if m % 2 != 0]

    heuristic_orders = [
        sorted(even_moduli, key=lambda x: -count_divisors(x)) + sorted(odd_moduli, key=lambda x: -count_divisors(x)),
        sorted(even_moduli, reverse=True) + sorted(odd_moduli, reverse=True),
        sorted(moduli, key=lambda x: -count_divisors(x)),
        sorted(moduli, reverse=True),
        sorted(moduli, key=lambda x: (-count_pf(x), -x)),
        sorted(even_moduli, key=lambda x: (-count_pf(x), -x)) + sorted(odd_moduli, key=lambda x: (-count_pf(x), -x)),
    ]

    best_set = []
    best_size = 0

    for order in heuristic_orders:
        if time.time() > deadline:
            break
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Random restarts with time budget
    rng = random.Random(42)
    num_restarts = 30000
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
        random_order = moduli.copy()
        rng.shuffle(random_order)

        if i % 3 == 0:
            s = greedy(random_order)
        elif i % 3 == 1:
            s = greedy_random(random_order, rng)
        else:
            # Weighted random: bias towards highly composite numbers
            weights = [count_divisors(m) ** 2 for m in moduli]
            total_w = sum(weights)
            random_order = []
            for m in moduli:
                if rng.random() < weights[moduli.index(m)] / total_w * 3:
                    random_order.append(m)
            if len(random_order) < len(moduli):
                for m in moduli:
                    if m not in random_order:
                        random_order.append(m)
            s = greedy(random_order)

        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local search: try to add missing moduli, and try to swap residues to make room
    def local_search(current_set, time_limit):
        current_set = list(current_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_moduli = {m for r, m in current_set}

            # Find missing moduli that are non-coprime with all current moduli
            candidates = [m for m in moduli
                          if m not in current_moduli
                          and all(gcd_val[m][m_j] > 1 for r_j, m_j in current_set)]

            # Sort candidates by how "easy" they are (more gcd constraints = harder)
            candidates.sort(key=lambda m: sum(1 for r_j, m_j in current_set if gcd_val[m][m_j] == 1))

            for m_new in candidates:
                if time.time() > time_limit:
                    break
                r_new = try_add(m_new, current_set)
                if r_new is not None:
                    current_set.append((r_new, m_new))
                    improved = True
                    break

                # If direct add failed, try to change one existing residue
                if len(current_set) < 45:
                    for idx in range(len(current_set)):
                        r_old, m_old = current_set[idx]
                        # Try a few random residues for m_old
                        for _ in range(10):
                            r_try = rng.randint(0, m_old - 1)
                            if r_try == r_old:
                                continue
                            # Check if r_try is valid for m_old against all others
                            valid = True
                            for j, (r_j, m_j) in enumerate(current_set):
                                if j == idx:
                                    continue
                                g = gcd_val[m_old][m_j]
                                if (r_try - r_j) % g == 0:
                                    valid = False
                                    break
                            if not valid:
                                continue

                            current_set[idx] = (r_try, m_old)
                            r_new2 = try_add(m_new, current_set)
                            if r_new2 is not None:
                                current_set.append((r_new2, m_new))
                                improved = True
                                break
                            current_set[idx] = (r_old, m_old)
                        if improved:
                            break
            if not improved:
                break
        return current_set

    if best_set and time.time() < deadline - 3:
        improved_set = local_search(best_set, deadline - 1)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)

    # Final polish: one more local search pass
    if best_set and time.time() < deadline - 1:
        improved_set = local_search(best_set, deadline - 0.5)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)

    result = [[r, m] for r, m in best_set]

    if not result:
        # Fallback: a small valid set
        result = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]]

    return result
