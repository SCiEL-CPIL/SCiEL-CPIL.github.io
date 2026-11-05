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
    
    # Precompute value for each a
    val = [0.0] * 512
    for a in range(1, 512):
        val[a] = 1.0 / (512 - a)
    
    # Greedy: largest a first
    selected = []
    selected_set = set()
    used_primes = set()
    
    for a in range(511, 0, -1):
        if time.time() > deadline - 5:
            break
        if a == 1:
            selected.append(a)
            selected_set.add(a)
            continue
        if prime_factors[a].isdisjoint(used_primes):
            selected.append(a)
            selected_set.add(a)
            used_primes |= prime_factors[a]
    
    best = list(selected)
    best_val = sum(val[a] for a in best)
    
    def try_refill(remaining_list, rem_set, rem_used):
        """Greedily add unselected numbers in descending order of a."""
        new_sel = list(remaining_list)
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
        return new_sel
    
    # Local search with 1-for-refill and 2-for-refill moves
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        improved = False
        n = len(best)
        
        # 1-for-refill: remove one, refill
        for i in range(n):
            if time.time() > deadline - 5:
                break
            a = best[i]
            remaining = best[:i] + best[i+1:]
            rem_set = set(remaining)
            rem_used = set()
            for b in remaining:
                rem_used |= prime_factors[b]
            
            new_sel = try_refill(remaining, rem_set, rem_used)
            new_val = sum(val[x] for x in new_sel)
            if new_val > best_val + 1e-12:
                best = new_sel
                best_val = new_val
                improved = True
                break
        
        if improved:
            continue
        
        # 2-for-refill: remove two, refill
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 10:
                    break
                a1 = best[i]
                a2 = best[j]
                remaining = best[:i] + best[i+1:j] + best[j+1:]
                rem_set = set(remaining)
                rem_used = set()
                for b in remaining:
                    rem_used |= prime_factors[b]
                
                new_sel = try_refill(remaining, rem_set, rem_used)
                new_val = sum(val[x] for x in new_sel)
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
