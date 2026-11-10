"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

MAX_MOD = 128
DEADLINE_MARGIN = 5.0  # seconds before the 90s limit

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _try_add(residues, moduli, r, m, existing):
    """Check if (r, m) can be added to the existing set."""
    for i in range(len(existing)):
        er, em = existing[i]
        if not _is_disjoint(r, m, er, em):
            return False
    return True

def _greedy_construct(ordering, time_limit):
    """
    Greedy construction: iterate through moduli in given order,
    for each modulus try residues in some order and pick the first that works.
    """
    start_time = time.time()
    selected = []  # list of (residue, modulus)
    
    for m in ordering:
        if time.time() - start_time > time_limit:
            break
        # Try residues in order 0, 1, ..., m-1
        # But we can be smarter: try residues that are "far" from existing ones
        # For now, just try 0 to m-1
        for r in range(m):
            if _try_add(None, None, r, m, selected):
                selected.append((r, m))
                break
    
    return selected

def _local_search_1_for_2(selected, time_limit):
    """
    Try to improve by removing one element and adding two.
    """
    start_time = time.time()
    current = list(selected)
    current_size = len(current)
    
    # Precompute which moduli are used
    used_moduli = set(m for _, m in current)
    available_moduli = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        # Try removing each element and adding two new ones
        for i in range(len(current)):
            if time.time() - start_time > time_limit:
                break
            
            removed = current[i]
            remaining = current[:i] + current[i+1:]
            removed_mod = removed[1]
            
            # Update available moduli
            avail = [m for m in range(2, MAX_MOD + 1) if m not in set(mm for _, mm in remaining)]
            
            # Try all pairs of available moduli
            for j in range(len(avail)):
                if time.time() - start_time > time_limit:
                    break
                m1 = avail[j]
                found_first = None
                for r1 in range(m1):
                    if _try_add(None, None, r1, m1, remaining):
                        found_first = (r1, m1)
                        break
                if found_first is None:
                    continue
                
                # Now try to add a second modulus
                for k in range(j + 1, len(avail)):
                    m2 = avail[k]
                    for r2 in range(m2):
                        if _try_add(None, None, r2, m2, remaining + [found_first]):
                            new_current = remaining + [found_first, (r2, m2)]
                            if len(new_current) > current_size:
                                current = new_current
                                current_size = len(current)
                                improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return current

def _randomized_greedy(orderings, time_limit):
    """
    Try multiple orderings and keep the best.
    """
    start_time = time.time()
    best = []
    
    for ordering in orderings:
        if time.time() - start_time > time_limit:
            break
        result = _greedy_construct(ordering, time_limit - (time.time() - start_time))
        if len(result) > len(best):
            best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = 90.0 - DEADLINE_MARGIN  # 85 seconds
    
    # Generate multiple orderings
    orderings = []
    
    # 1. Natural order
    orderings.append(list(range(2, MAX_MOD + 1)))
    
    # 2. Reverse order
    orderings.append(list(range(MAX_MOD, 1, -1)))
    
    # 3. Prime moduli first
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in range(2, MAX_MOD + 1) if is_prime(m)]
    composites = [m for m in range(2, MAX_MOD + 1) if not is_prime(m)]
    orderings.append(primes + composites)
    
    # 4. By number of prime factors (ascending)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    by_factors = sorted(range(2, MAX_MOD + 1), key=lambda x: (num_prime_factors(x), x))
    orderings.append(by_factors)
    
    # 5. Random shuffles
    random.seed(42)
    for _ in range(20):
        shuffled = list(range(2, MAX_MOD + 1))
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Run greedy with multiple orderings
    elapsed = time.time() - start_time
    remaining_time = deadline - elapsed
    if remaining_time <= 0:
        # Fallback: simple deterministic construction
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    best = _randomized_greedy(orderings, remaining_time * 0.5)
    
    # Local search
    elapsed = time.time() - start_time
    remaining_time = deadline - elapsed
    if remaining_time > 0 and len(best) > 0:
        improved = _local_search_1_for_2(best, remaining_time * 0.8)
        if len(improved) > len(best):
            best = improved
    
    # Convert to list of [residue, modulus] pairs
    result = [[r, m] for r, m in best]
    
    # Ensure nonempty
    if not result:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return result

# EVOLVE-BLOCK-END
