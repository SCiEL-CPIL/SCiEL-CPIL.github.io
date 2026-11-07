"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    max_m = 10**18
    
    # Use a smaller set of primes to make meet-in-the-middle feasible
    # Split into two groups for MITM
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    mid = len(primes) // 2
    left_primes = primes[:mid]
    right_primes = primes[mid:]
    
    def time_check():
        return time.time() > deadline
    
    # Generate all (k, sigma_k) pairs for a set of primes
    def generate_combinations(prime_list, max_k=10**9):
        results = [(1, 1)]
        for p in prime_list:
            new_results = []
            p_pow = 1
            sigma_p = 1
            for e in range(30):
                for k, sigma_k in results:
                    new_k = k * p_pow
                    if new_k > max_k:
                        break
                    new_sigma = sigma_k * sigma_p
                    if new_k * new_sigma <= max_m:
                        new_results.append((new_k, new_sigma))
                p_pow *= p
                if p_pow > max_k:
                    break
                sigma_p += p_pow
            results = new_results
        return results
    
    # Generate combinations for left and right halves
    left_combos = generate_combinations(left_primes)
    if time_check():
        return {'M': 333312, 'k': [336, 372, 434]}
    
    right_combos = generate_combinations(right_primes)
    if time_check():
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # For each left combo, we need to find right combos such that:
    # (k_left * sigma_left) * (k_right * sigma_right) = M
    # This means k_left * k_right * sigma_left * sigma_right = M
    # We want to find M values that can be formed in many ways
    
    # Build a dictionary: for each (k_left, sigma_left), store it
    # Then for each right combo, compute M = k_left * k_right * sigma_left * sigma_right
    # But this is O(n_left * n_right) which might be too slow
    
    # Better approach: for each M, count how many (k_left, sigma_left, k_right, sigma_right) 
    # combinations give k_left * k_right * sigma_left * sigma_right = M
    # and k_left * k_right <= sqrt(M)
    
    # Alternative: iterate over all pairs and compute M, then group by M
    # But we need to be careful about the constraint k <= sqrt(M)
    
    m_to_ks = defaultdict(list)
    
    # For efficiency, sort right_combos and use some pruning
    # For each left combo (k_l, s_l), for each right combo (k_r, s_r):
    # k = k_l * k_r, sigma = s_l * s_r, M = k * sigma
    # Constraint: k <= sqrt(M) => k_l * k_r <= sqrt(k_l * k_r * s_l * s_r)
    # => (k_l * k_r)^2 <= k_l * k_r * s_l * s_r
    # => k_l * k_r <= s_l * s_r
    
    count = 0
    for k_l, s_l in left_combos:
        if time_check():
            break
        for k_r, s_r in right_combos:
            count += 1
            if count % 1000000 == 0 and time_check():
                break
            k = k_l * k_r
            sigma = s_l * s_r
            M = k * sigma
            if M > max_m:
                continue
            # Check constraint: k <= sqrt(M)
            if k * k <= M:
                m_to_ks[M].append(k)
    
    # Find the best M
    best_m = 0
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = M
            best_ks = ks
    
    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_m), 'k': best_ks}
# EVOLVE-BLOCK-END
