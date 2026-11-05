import time
from math import gcd
from fractions import Fraction

def construct():
    deadline = time.time() + 115  # safety margin
    
    # Precompute prime factors for 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            for m in range(p, 512, p):
                prime_factors[m].add(p)
    
    # Value function
    def value(a):
        return 1.0 / (512 - a)
    
    # Greedy: largest a first
    selected = []
    selected_set = set()
    used_primes = set()
    
    for a in range(511, 0, -1):
        if time.time() > deadline - 5:
            break
        if a == 1:
            # 1 is coprime to everything
            selected.append(a)
            selected_set.add(a)
            continue
        # Check if a's prime factors are disjoint from used_primes
        if prime_factors[a].isdisjoint(used_primes):
            selected.append(a)
            selected_set.add(a)
            used_primes |= prime_factors[a]
    
    def total_value(sel):
        return sum(value(a) for a in sel)
    
    best = list(selected)
    best_val = total_value(best)
    
    # Local search: try removing one selected and greedily refilling
    # We'll do several passes
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(best)):
            a = best[i]
            # Remove a
            remaining = best[:i] + best[i+1:]
            rem_set = set(remaining)
            rem_used = set()
            for b in remaining:
                rem_used |= prime_factors[b]
            
            # Greedily refill: try adding unselected numbers in descending order
            new_sel = list(remaining)
            new_used = set(rem_used)
            for c in range(511, 0, -1):
                if c in rem_set:
                    continue
                if c == 1:
                    if 1 not in rem_set:
                        new_sel.append(c)
                    continue
                if prime_factors[c].isdisjoint(new_used):
                    new_sel.append(c)
                    new_used |= prime_factors[c]
            
            new_val = total_value(new_sel)
            if new_val > best_val + 1e-12:
                best = new_sel
                best_val = new_val
                improved = True
                break  # restart the outer loop
        
        if not improved:
            break
    
    return best
