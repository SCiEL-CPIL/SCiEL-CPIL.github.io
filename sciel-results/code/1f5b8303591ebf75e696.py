"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _spf_sieve(limit):
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _pf_set(n, spf):
    s = set()
    while n > 1:
        p = spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

def _build_sum_pf_table(limit, spf):
    """Precompute prime factor sets for all integers 2..limit."""
    table = [set() for _ in range(limit + 1)]
    for i in range(2, limit + 1):
        table[i] = _pf_set(i, spf)
    return table

def _score_full(arr, sum_pf_table):
    """Count distinct prime factors of all pairwise sums of arr."""
    ps = set()
    n = len(arr)
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            ps |= sum_pf_table[ai + arr[j]]
    return len(ps)

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin
    
    MAX_SUM = 20000
    spf = _spf_sieve(MAX_SUM)
    sum_pf_table = _build_sum_pf_table(MAX_SUM, spf)
    
    best_set = list(range(1, 13))
    best_score = _score_full(best_set, sum_pf_table)
    
    # Strategy: Search over moduli M.
    # Let a_i = M * k_i.
    # Sum = M * (k_i + k_j).
    # Primes(Sum) = Primes(M) U Primes(k_i + k_j).
    # We want to minimize |Primes(M) U Union_{i<j} Primes(k_i + k_j)|.
    
    # Candidate moduli: small numbers with few prime factors.
    # If M has few primes, it's a good base.
    candidate_moduli = []
    for M in range(2, 1001):
        pfM = _pf_set(M, spf)
        if len(pfM) <= 3:  # Allow up to 3 distinct primes in M
            candidate_moduli.append((M, len(pfM), pfM))
    
    # Sort by number of prime factors, then by M value
    candidate_moduli.sort(key=lambda x: (x[1], x[0]))
    
    random.seed(42)
    
    # Precompute prime factor sets for small sums of k's
    # Max k is 10000 // M. For M=2, max k=5000. Sum max 10000.
    # We already have sum_pf_table up to 20000, which covers k_i+k_j for k<=10000.
    
    def score_k_set(k_vals, pfM, sum_pf_table):
        """
        Score a set of multipliers k_vals given the base modulus prime set pfM.
        Returns the size of the union of pfM and prime factors of all k_i+k_j.
        """
        ps = set(pfM)
        n = len(k_vals)
        for i in range(n):
            ki = k_vals[i]
            for j in range(i + 1, n):
                s = ki + k_vals[j]
                if s <= len(sum_pf_table):
                    ps |= sum_pf_table[s]
        return len(ps)

    def local_search_k(k_vals, M, pfM, sum_pf_table, max_k, time_limit):
        """
        Local search to optimize k_vals for a fixed M.
        """
        current_k = k_vals[:]
        current_score = score_k_set(current_k, pfM, sum_pf_table)
        best_k = current_k[:]
        best_score_local = current_score
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try swapping one element
            for idx in range(12):
                if time.time() > time_limit:
                    break
                old_k = current_k[idx]
                # Try random replacements
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    new_k = random.randint(1, max_k)
                    if new_k == old_k or new_k in current_k:
                        continue
                    
                    new_k_vals = current_k[:]
                    new_k_vals[idx] = new_k
                    new_k_vals.sort()
                    
                    new_score = score_k_set(new_k_vals, pfM, sum_pf_table)
                    if new_score < current_score:
                        current_k = new_k_vals
                        current_score = new_score
                        improved = True
                        if current_score < best_score_local:
                            best_score_local = current_score
                            best_k = current_k[:]
                        break
        return best_k, best_score_local

    processed_moduli = 0
    for M, npfM, pfM in candidate_moduli:
        if time.time() > deadline - 5:
            break
        if processed_moduli >= 200: # Limit moduli to try
            break
            
        max_k = 10000 // M
        if max_k < 12:
            continue
            
        processed_moduli += 1
        
        # Lower bound check: if npfM >= best_score, we can't beat it even with 0 new primes
        if npfM >= best_score:
            continue
            
        # Initial guess for k: 1..12
        # Also try small smooth numbers
        init_k = list(range(1, 13))
        
        # Run local search
        # Allocate time based on promise (smaller npfM is more promising)
        time_alloc = 2.0 if npfM <= 2 else 0.5
        local_limit = time.time() + time_alloc
        
        best_k, local_score = local_search_k(init_k, M, pfM, sum_pf_table, max_k, local_limit)
        
        # Convert back to actual values
        if local_score < best_score:
            arr = [k * M for k in best_k]
            # Verify
            if len(set(arr)) == 12 and max(arr) <= 10000:
                # Re-score full to be sure (should be same)
                full_s = _score_full(arr, sum_pf_table)
                if full_s < best_score:
                    best_score = full_s
                    best_set = arr[:]
        
        # If we found a very good set (e.g., score == npfM), we might stop early
        # but let's keep going to see if we can get even lower (unlikely to go below npfM)
        if best_score == 2: # Theoretical minimum is likely 2 or 3
             # Can we do better than 2? No, min primes is 1 (power of 2), but sums of distinct positives > 2.
             # If we hit 2, it's likely optimal or near optimal.
             pass

    # Final polish on the best set found using direct value local search
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                old = best_set[i]
                for _ in range(100):
                    if time.time() > deadline - 2:
                        break
                    new_val = random.randint(1, 10000)
                    if new_val == old or new_val in best_set:
                        continue
                    # Quick score update
                    ns = _score_full(best_set, sum_pf_table) # This is slow, use delta if possible
                    # For simplicity in this structural change, we rely on the M-based search being strong.
                    # If we have time, do a few random full swaps.
                    test_set = best_set[:]
                    test_set[i] = new_val
                    test_set.sort()
                    ns = _score_full(test_set, sum_pf_table)
                    if ns < best_score:
                        best_score = ns
                        best_set = test_set
                        improved = True
                        break

    return best_set
# EVOLVE-BLOCK-END
