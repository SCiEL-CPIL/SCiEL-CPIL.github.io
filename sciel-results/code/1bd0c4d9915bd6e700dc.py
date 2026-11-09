# EVOLVE-BLOCK-START
from math import prod, log
import itertools

_LIMIT = 10**18
_SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def _score_from_exp(base_primes, exps):
    """Score = sum_i p_i * prod_{j != i}(e_j+1); returns (score, product)."""
    counts = [e + 1 for e in exps]
    total = prod(counts)
    score = 0
    product = 1
    for p, c in zip(base_primes, counts):
        score += p * (total // c)
        product *= p ** (c - 1)
    return score, product

def _best_structure():
    best_score = 0
    best_primes = None
    best_exps = None
    
    # Generate a larger list of primes for swapping
    primes_pool = _SMALL
    
    for k in range(8, 17):
        if k > len(primes_pool):
            break
        # Try different exponent patterns
        # Mostly 1s, maybe one 2
        patterns = []
        # All 1s
        patterns.append([1]*k)
        # One 2, rest 1s
        if k >= 2:
            for i in range(k):
                p = [1]*k
                p[i] = 2
                patterns.append(p)
        
        for exps in patterns:
            # Check if smallest k primes with these exponents fit
            # Sort exps descending to assign to smallest primes? 
            # No, to minimize product, assign largest exponents to smallest primes.
            sorted_exps = sorted(exps, reverse=True)
            smallest_primes = primes_pool[:k]
            # Calculate product with smallest primes
            prod_check = 1
            for p, e in zip(smallest_primes, sorted_exps):
                prod_check *= p ** e
            if prod_check > _LIMIT:
                continue
            
            # Start with smallest primes assigned to sorted exps
            # We need to track which prime has which exponent
            # To maximize score, we should assign larger primes to larger coefficients?
            # Coefficient C_i depends on exps.
            # If exps are all 1, C_i are all equal.
            # If one exp is 2, C for that prime is half of others.
            # So we want the prime with exp 2 to be as small as possible?
            # Score term for exp 2 prime: p * (2^k / 2) = p * 2^{k-1}
            # Score term for exp 1 prime: p * (2^k / 2) = p * 2^{k-1}
            # Wait.
            # total = prod(e_j+1).
            # If one e=2, others 1: total = 3 * 2^{k-1}.
            # C for e=2: total / 3 = 2^{k-1}.
            # C for e=1: total / 2 = 3 * 2^{k-2}.
            # So C for e=1 is 1.5 times C for e=2.
            # We want to assign larger primes to larger C.
            # So assign larger primes to e=1 slots, smaller to e=2 slot.
            
            # Strategy:
            # 1. Sort primes ascending.
            # 2. Assign smallest prime to largest exponent (to save product budget).
            # 3. Then greedily swap primes to increase score.
            
            # Initial assignment:
            # Sort indices by exponent descending.
            # Assign smallest primes to largest exponents.
            indices = list(range(k))
            indices.sort(key=lambda i: exps[i], reverse=True)
            
            # initial_primes[i] is the prime assigned to slot i
            initial_primes = [0]*k
            for rank, idx in enumerate(indices):
                initial_primes[idx] = primes_pool[rank]
                
            current_primes = initial_primes[:]
            current_exps = exps[:]
            
            # Calculate current score and product
            curr_score, curr_prod = _score_from_exp(current_primes, current_exps)
            
            # Greedy improvement
            improved = True
            while improved:
                improved = False
                # Try to swap any prime with a larger one
                # We want to increase score.
                # Score increase: C_i * (p_new - p_old)
                # Product change: * (p_new / p_old)^e_i
                # Constraint: curr_prod * (p_new / p_old)^e_i <= _LIMIT
                
                for i in range(k):
                    p_old = current_primes[i]
                    e_i = current_exps[i]
                    C_i = 1
                    for j in range(k):
                        if i != j:
                            C_i *= (current_exps[j] + 1)
                    
                    # Find largest prime q in pool, q > p_old, q not in current_primes
                    # such that curr_prod * (q / p_old)**e_i <= _LIMIT
                    # And score increases (always if q > p_old)
                    
                    # To be efficient, iterate candidates from largest down
                    # But we need to check distinctness.
                    
                    best_q = None
                    best_gain = 0
                    for q in primes_pool:
                        if q <= p_old:
                            continue
                        if q in current_primes:
                            continue
                        # Check product constraint
                        # curr_prod * q**e_i / p_old**e_i <= _LIMIT
                        # curr_prod * q**e_i <= _LIMIT * p_old**e_i
                        if curr_prod * (q ** e_i) <= _LIMIT * (p_old ** e_i):
                            gain = C_i * (q - p_old)
                            if gain > best_gain:
                                best_gain = gain
                                best_q = q
                                
                    if best_q:
                        # Perform swap
                        curr_prod = curr_prod * (best_q ** e_i) // (p_old ** e_i)
                        curr_score += best_gain
                        current_primes[i] = best_q
                        improved = True
                        # Restart loop as other swaps might be possible
                        break
                        
            if curr_score > best_score:
                best_score = curr_score
                best_primes = current_primes[:]
                best_exps = current_exps[:]
                
    return best_score, best_primes, best_exps

def _build_n(primes, exps):
    n = 1
    for p, e in zip(primes, exps):
        n *= p ** e
    return n

def construct():
    score, primes, exps = _best_structure()
    n = _build_n(primes, exps)
    assert 2 <= n <= _LIMIT
    return n
# EVOLVE-BLOCK-END
