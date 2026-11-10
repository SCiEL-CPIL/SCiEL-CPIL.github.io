import time
import random
from math import gcd

def construct():
    deadline = time.time() + 115
    
    # Precompute prime factors for each a in 1..511
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
    
    pf = [set() for _ in range(512)]
    for a in range(1, 512):
        pf[a] = prime_factors(a)
    
    def score(a):
        return 1.0 / (512 - a)
    
    def greedy(order):
        selected = set()
        used_primes = set()
        for a in order:
            if pf[a] & used_primes:
                continue
            selected.add(a)
            used_primes |= pf[a]
        return selected, used_primes, sum(score(a) for a in selected)
    
    def local_search(selected, used_primes, best_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try additions
            for a in range(511, 0, -1):
                if a in selected:
                    continue
                if pf[a] & used_primes:
                    continue
                selected.add(a)
                used_primes |= pf[a]
                best_score += score(a)
                improved = True
                break
            
            if time.time() >= deadline:
                break
            
            # Try swaps
            selected_list = list(selected)
            for i, a_i in enumerate(selected_list):
                if time.time() >= deadline:
                    break
                freed = pf[a_i]
                available_check = used_primes - freed
                for a_j in range(511, 0, -1):
                    if a_j in selected:
                        continue
                    if pf[a_j] & available_check:
                        continue
                    delta = score(a_j) - score(a_i)
                    if delta > 1e-12:
                        selected.remove(a_i)
                        used_primes -= freed
                        selected.add(a_j)
                        used_primes |= pf[a_j]
                        best_score += delta
                        improved = True
                        break
                if improved:
                    break
        
        return selected, used_primes, best_score
    
    # Try multiple greedy initializations
    best_selected = set()
    best_used = set()
    best_score = 0.0
    
    orders = [
        list(range(511, 0, -1)),
        list(range(1, 512)),
    ]
    
    # Add a few random orders
    random.seed(42)
    for _ in range(5):
        order = list(range(1, 512))
        random.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() >= deadline:
            break
        selected, used_primes, sc = greedy(order)
        selected, used_primes, sc = local_search(selected, used_primes, sc)
        if sc > best_score:
            best_score = sc
            best_selected = selected.copy()
            best_used = used_primes.copy()
    
    return list(best_selected)
