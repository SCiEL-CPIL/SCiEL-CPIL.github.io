"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 145  # safety margin for 150s wall-clock limit
    
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
    
    def value(a):
        return 1.0 / (512 - a)
    
    def total_value(sel):
        return sum(value(a) for a in sel)
    
    # Greedy: largest a first, using prime factor disjointness for fast check
    def greedy_build(used_primes, forbidden_set=None):
        if forbidden_set is None:
            forbidden_set = set()
        selected = []
        used = set(used_primes)
        for a in range(511, 0, -1):
            if a in forbidden_set:
                continue
            if a == 1:
                selected.append(a)
                continue
            if prime_factors[a].isdisjoint(used):
                selected.append(a)
                used |= prime_factors[a]
        return selected, used
    
    selected, used_primes = greedy_build(set(), set())
    best = list(selected)
    best_val = total_value(best)
    
    # Local search: try removing one selected and greedily refilling
    for iteration in range(500):
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
    
    # Additional local search: try swapping two selected for two unselected
    # This is more expensive but can find better solutions
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        n = len(best)
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                a, b = best[i], best[j]
                remaining = [x for k, x in enumerate(best) if k != i and k != j]
                rem_set = set(remaining)
                rem_used = set()
                for c in remaining:
                    rem_used |= prime_factors[c]
                
                # Try to add two numbers that are coprime to remaining and each other
                best_pair = None
                best_pair_val = -1
                candidates = [c for c in range(511, 0, -1) if c not in rem_set and c != 1]
                
                # Limit candidates for speed
                for idx1, c1 in enumerate(candidates[:50]):
                    if prime_factors[c1] and not prime_factors[c1].isdisjoint(rem_used):
                        continue
                    for c2 in candidates[idx1+1:idx1+50]:
                        if c2 == c1:
                            continue
                        if not prime_factors[c1].isdisjoint(prime_factors[c2]):
                            continue
                        if prime_factors[c2] and not prime_factors[c2].isdisjoint(rem_used):
                            continue
                        # Check c1 and c2 are coprime to each other and to remaining
                        if gcd(c1, c2) != 1:
                            continue
                        val = value(c1) + value(c2)
                        if val > best_pair_val:
                            best_pair_val = val
                            best_pair = (c1, c2)
                
                if best_pair:
                    c1, c2 = best_pair
                    new_sel = remaining + [c1, c2]
                    new_val = total_value(new_sel)
                    old_val = value(a) + value(b)
                    if new_val > best_val + 1e-12:
                        best = new_sel
                        best_val = new_val
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best

# EVOLVE-BLOCK-END
