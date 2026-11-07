"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

DEADLINE = 85.0  # safety margin before 90s kill

def _prime_factors(n):
    """Return set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    N = 256
    
    # Precompute prime factors for all numbers 1..256
    pf = {}
    for i in range(1, N + 1):
        if i == 1:
            pf[i] = frozenset()
        else:
            pf[i] = frozenset(_prime_factors(i))
    
    # Greedy: process numbers in descending order, add if coprime to all selected
    # Start with 1 (allowed once, coprime to everything)
    selected = [1]
    selected_factors = set()
    
    for a in range(N, 1, -1):
        if time.time() - start_time > DEADLINE:
            break
        # Check if a is coprime to all selected numbers
        ok = True
        for f in pf[a]:
            if f in selected_factors:
                ok = False
                break
        if ok:
            selected.append(a)
            selected_factors.update(pf[a])
    
    current_sum = sum(selected)
    
    # Local search: try to swap out a smaller number for a larger unused one
    # Or try to add numbers that become available after removing some
    # Strategy: repeatedly try to improve by removing a small number and adding larger ones
    
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        selected_set = set(selected)
        selected_factors = set()
        for x in selected:
            selected_factors.update(pf[x])
        
        # Try removing each selected number (except 1) and see if we can add more
        for rem in list(selected):
            if rem == 1:
                continue
            if time.time() - start_time > DEADLINE:
                break
            
            # Recompute factors without rem
            new_factors = set()
            for x in selected:
                if x != rem:
                    new_factors.update(pf[x])
            
            # Try to add unused numbers in descending order
            added = []
            for a in range(N, 1, -1):
                if a in selected_set and a != rem:
                    continue
                if a == rem:
                    continue
                ok = True
                for f in pf[a]:
                    if f in new_factors:
                        ok = False
                        break
                if ok:
                    added.append(a)
                    new_factors.update(pf[a])
            
            new_sum = sum(x for x in selected if x != rem) + sum(added)
            if new_sum > current_sum:
                selected = [x for x in selected if x != rem] + added
                current_sum = new_sum
                improved = True
                break
    
    # Final: try adding any remaining numbers that fit
    if time.time() - start_time < DEADLINE:
        selected_set = set(selected)
        selected_factors = set()
        for x in selected:
            selected_factors.update(pf[x])
        
        for a in range(N, 1, -1):
            if a in selected_set:
                continue
            ok = True
            for f in pf[a]:
                if f in selected_factors:
                    ok = False
                    break
            if ok:
                selected.append(a)
                selected_factors.update(pf[a])
                current_sum = sum(selected)
    
    # Ensure all are Python int
    result = [int(x) for x in selected]
    return result
# EVOLVE-BLOCK-END
