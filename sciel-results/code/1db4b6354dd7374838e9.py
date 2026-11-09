"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _validate(S):
    """Fast validation: for each a in S, check no distinct b,c in S have a|b*c."""
    S_list = sorted(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                if S_list[j] * S_list[k] % a == 0:
                    return False
    return True

def _validate_with_b(S, b):
    """Check if adding b to S keeps it valid. Only check triples involving b."""
    S_list = sorted(S | {b})
    n = len(S_list)
    # For each a in S_list, check if there exist distinct b',c' in S_list (not both being b)
    # such that a | b'*c'. We only need to check where b participates or where a==b.
    # Actually, we need to check all triples in S_list, but triples not involving b were already valid.
    # So we only need to check triples that include b.
    # Triple (x,y,z) includes b means one of x,y,z is b.
    # Case 1: b is the divisor: b | x*y for distinct x,y in S (x!=y, x!=b, y!=b)
    # Case 2: b is one of the multiplicands: a | b*x for distinct a,x in S (a!=b, x!=b, a!=x)
    # Case 3: b is the other multiplicand: same as case 2 by symmetry
    
    # Case 1: b divides x*y
    for i in range(n):
        x = S_list[i]
        if x == b:
            continue
        for j in range(i + 1, n):
            y = S_list[j]
            if y == b:
                continue
            if x * y % b == 0:
                return False
    
    # Case 2: a divides b*x where a != b, x != b, a != x
    for i in range(n):
        a = S_list[i]
        if a == b:
            continue
        for j in range(n):
            x = S_list[j]
            if x == b or x == a:
                continue
            if b * x % a == 0:
                return False
    
    return True

def _validate_remove_add(S, remove_set, add_set):
    """Check validity after removing remove_set and adding add_set."""
    new_S = (S - remove_set) | add_set
    new_list = sorted(new_S)
    n = len(new_list)
    for i in range(n):
        a = new_list[i]
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                if new_list[j] * new_list[k] % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 85
    rng = random.Random(12345)
    
    # Start with primes
    primes = [a for a in range(2, 257) if all(a % d for d in range(2, isqrt(a) + 1))]
    best = list(primes)
    best_size = len(best)
    
    # Strategy: try to add composites one by one, removing conflicting primes
    # For each composite c, find which primes p in current set conflict (p | c*x for some x in set,
    # or c | p*x for some x, or p | c*y where y is another added element)
    
    S = set(primes)
    all_numbers = list(range(2, 257))
    composites = [a for a in all_numbers if a not in S]
    
    # Sort composites by "attractiveness" - prefer those with fewer prime factors
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
    
    composites.sort(key=lambda x: (num_prime_factors(x), x))
    
    for c in composites:
        if time.time() >= deadline:
            break
        if c in S:
            continue
        # Try adding c
        if _validate_with_b(S, c):
            S.add(c)
            best_size = len(S)
    
    best = sorted(S)
    
    # Local search: 1-for-1 swaps
    while time.time() < deadline:
        improved = False
        S_list = sorted(S)
        candidates = sorted(set(range(2, 257)) - S)
        
        for a in S_list:
            if time.time() >= deadline:
                break
            for b in candidates:
                if time.time() >= deadline:
                    break
                if _validate_remove_add(S, {a}, {b}):
                    S.remove(a)
                    S.add(b)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    best = sorted(S)
    best_size = len(best)
    
    # 1-for-2 swaps (remove 1, add 2)
    while time.time() < deadline:
        improved = False
        S_list = sorted(S)
        candidates = sorted(set(range(2, 257)) - S)
        
        for a in S_list:
            if time.time() >= deadline:
                break
            for i in range(len(candidates)):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, len(candidates)):
                    if time.time() >= deadline:
                        break
                    b, c = candidates[i], candidates[j]
                    if _validate_remove_add(S, {a}, {b, c}):
                        S.remove(a)
                        S.add(b)
                        S.add(c)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    best = sorted(S)
    best_size = len(best)
    
    # Random perturbation + refill
    for _ in range(20):
        if time.time() >= deadline:
            break
        S_list = sorted(S)
        if len(S_list) < 3:
            break
        remove_count = rng.randint(2, 4)
        to_remove = set(rng.sample(S_list, min(remove_count, len(S_list))))
        trial_S = S - to_remove
        
        # Greedily refill
        candidates = sorted(set(range(2, 257)) - trial_S)
        for b in candidates:
            if time.time() >= deadline:
                break
            if _validate_with_b(trial_S, b):
                trial_S.add(b)
        
        if len(trial_S) > best_size:
            S = trial_S
            best = sorted(S)
            best_size = len(S)
    
    best = sorted(S)
    
    if not _validate(best):
        best = list(primes)
    
    return list(best)
# EVOLVE-BLOCK-END
