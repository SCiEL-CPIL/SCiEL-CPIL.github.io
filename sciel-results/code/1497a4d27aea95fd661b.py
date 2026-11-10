import time
import math
import random

def _is_disjoint(r1, m1, r2, m2, gcd_table=None):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    if gcd_table is not None:
        g = gcd_table[m1][m2]
    else:
        g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    MAX_MOD = 128

    moduli_list = list(range(2, MAX_MOD + 1))

    # Precompute gcd table for moduli 0..128
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    div_counts = {m: num_divisors(m) for m in moduli_list}

    def try_add_fast(chosen, m, gcd_table):
        """Try to find a valid residue for modulus m given current chosen set.
        Returns the first valid residue found in sequential order, or None."""
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                g = gcd_table[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None

    def greedy_construct(order, gcd_table):
        """Greedy: add moduli in given order, picking first valid residue."""
        chosen = []
        for m in order:
            r = try_add_fast(chosen, m, gcd_table)
            if r is not None:
                chosen.append((r, m))
        return chosen

    best_count = 0
    best_result = None

    # Define several orderings
    order_asc = sorted(moduli_list)
    order_desc = sorted(moduli_list, reverse=True)
    order_div_asc = sorted(moduli_list, key=lambda x: div_counts[x])
    order_div_desc = sorted(moduli_list, key=lambda x: div_counts[x], reverse=True)
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
    primes = [m for m in moduli_list if is_prime(m)]
    composites = [m for m in moduli_list if not is_prime(m)]
    order_prime_first = primes + composites

    orderings = [order_asc, order_desc, order_div_asc, order_div_desc, order_prime_first]

    # Run a limited number of restarts with random shuffles
    num_restarts = 100
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        if it < len(orderings):
            order = orderings[it][:]
        else:
            order = moduli_list[:]
            random.shuffle(order)
        
        chosen = greedy_construct(order, gcd_table)
        
        # Simple backfill: try to add any missing moduli
        used_moduli = set(m for _, m in chosen)
        unused = [m for m in moduli_list if m not in used_moduli]
        # Sort unused by heuristic: fewer divisors first
        unused.sort(key=lambda m: div_counts[m])
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for m in unused[:]:
                if m in used_moduli:
                    continue
                r = try_add_fast(chosen, m, gcd_table)
                if r is not None:
                    chosen.append((r, m))
                    used_moduli.add(m)
                    unused.remove(m)
                    improved = True
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
            
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback

    return best_result
