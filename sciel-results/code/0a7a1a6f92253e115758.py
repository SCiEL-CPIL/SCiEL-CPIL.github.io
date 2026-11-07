"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

P = {'n': 40, 'maximum_size': 18}

CANDIDATES = list(range(2, P['n'] + 1))
N_CANDS = len(CANDIDATES)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True

    def can_add(v, products):
        """Check if adding value v to products creates no collision."""
        for p in products:
            if (p * v) in products:
                return False
        return True

    def compute_products(S):
        """Compute all subset products for a set of indices."""
        products = {1}
        for idx in S:
            val = CANDIDATES[idx]
            products = products | {p * val for p in products}
        return products

    def greedy_solve(order):
        S = set()
        products = {1}
        for idx in order:
            val = CANDIDATES[idx]
            if can_add(val, products):
                S.add(idx)
                products = products | {p * val for p in products}
        return S, products

    # Try multiple orderings for greedy
    best_size = 0
    best_set = set()
    
    primes = [i for i in range(N_CANDS) if is_prime(CANDIDATES[i])]
    composites = [i for i in range(N_CANDS) if not is_prime(CANDIDATES[i])]
    
    def num_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Sort by number of prime factors (fewer is better for independence)
    order_by_pf = sorted(range(N_CANDS), key=lambda i: num_prime_factors(CANDIDATES[i]))
    
    # Sort by value (smaller values tend to create more collisions)
    order_by_val = sorted(range(N_CANDS), key=lambda i: CANDIDATES[i])
    
    # Sort by "independence score" - primes first, then by value
    order_by_indep = sorted(range(N_CANDS), key=lambda i: (num_prime_factors(CANDIDATES[i]), CANDIDATES[i]))
    
    orderings = [
        primes + composites,
        order_by_pf,
        order_by_val,
        order_by_indep
    ]
    
    # Add some random orderings
    for _ in range(5):
        orderings.append(sorted(range(N_CANDS), key=lambda i: random.random()))
    
    for order in orderings:
        S, products = greedy_solve(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Improved Branch and Bound with better pruning
    def bb_search(order, current_set, products, pos, depth=0):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if pos == len(order):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return
        
        # Upper bound: current + remaining
        ub = len(current_set) + (len(order) - pos)
        if ub <= best_size:
            return
        
        idx = order[pos]
        val = CANDIDATES[idx]
        
        # Try to include first (usually more promising)
        if can_add(val, products):
            new_set = current_set | {idx}
            new_products = products | {p * val for p in products}
            bb_search(order, new_set, new_products, pos + 1, depth + 1)
        
        # Try to exclude
        bb_search(order, current_set, products, pos + 1, depth + 1)

    # Run B&B with different orderings
    if best_size < P['maximum_size']:
        for order in [order_by_indep, order_by_pf, primes + composites]:
            if time.time() > deadline - 10:
                break
            bb_search(order, set(), {1}, 0)
    
    # Local search refinement with multiple strategies
    if best_size < P['maximum_size']:
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            S = best_set.copy()
            products = compute_products(S)
            
            # Strategy 1: Greedy add
            added = True
            while added and time.time() < deadline - 5:
                added = False
                for idx in range(N_CANDS):
                    if idx not in S:
                        val = CANDIDATES[idx]
                        if can_add(val, products):
                            S.add(idx)
                            products = products | {p * val for p in products}
                            added = True
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
            
            # Strategy 2: 1-for-1 swaps
            if len(S) < P['maximum_size']:
                for remove_idx in list(S):
                    if time.time() > deadline - 5:
                        break
                    S2 = S - {remove_idx}
                    products2 = compute_products(S2)
                    
                    for add_idx in range(N_CANDS):
                        if add_idx not in S2:
                            val = CANDIDATES[add_idx]
                            if can_add(val, products2):
                                S3 = S2 | {add_idx}
                                if len(S3) > best_size:
                                    best_size = len(S3)
                                    best_set = S3.copy()
            
            # Strategy 3: 2-for-3 swaps (more aggressive)
            if len(S) < P['maximum_size'] - 1:
                for remove_pair in combinations(list(S), 2):
                    if time.time() > deadline - 5:
                        break
                    S2 = S - set(remove_pair)
                    products2 = compute_products(S2)
                    
                    for add1 in range(N_CANDS):
                        if add1 not in S2:
                            val1 = CANDIDATES[add1]
                            if can_add(val1, products2):
                                S3 = S2 | {add1}
                                products3 = products2 | {p * val1 for p in products2}
                                
                                for add2 in range(N_CANDS):
                                    if add2 not in S3 and add2 != add1:
                                        val2 = CANDIDATES[add2]
                                        if can_add(val2, products3):
                                            S4 = S3 | {add2}
                                            if len(S4) > best_size:
                                                best_size = len(S4)
                                                best_set = S4.copy()

    # Final verification and cleanup
    final_set = set()
    products = {1}
    for idx in sorted(best_set):
        val = CANDIDATES[idx]
        if can_add(val, products):
            final_set.add(idx)
            products = products | {p * val for p in products}
    
    return [CANDIDATES[i] for i in sorted(final_set)]

# EVOLVE-BLOCK-END
