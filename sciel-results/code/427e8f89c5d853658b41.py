import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def _is_dissociated_subset(S, recips=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    i.e., all subset sums of {1/s for s in S} are distinct.
    """
    if not S:
        return True
    
    n = len(S)
    
    if recips is None:
        reciprocals = [1.0 / s for s in S]
    else:
        reciprocals = [recips[s] for s in S]
    
    # Generate all subset sums
    sums = [0.0]
    for r in reciprocals:
        # Add r to all existing sums
        new_sums = [s + r for s in sums]
        sums.extend(new_sums)
        
    # Check for duplicates
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums) - 1):
        if sorted_sums[i] == sorted_sums[i+1]:
            return False
            
    return True

def _mod_hash_check(S, primes, mod_recips):
    """
    Fast check using modular arithmetic with multiple primes.
    If all subset sums are distinct modulo each prime, they are likely distinct.
    """
    n = len(S)
    if n == 0:
        return True
    
    # For each prime, compute all subset sums mod p
    for p in primes:
        sums_mod = [0]
        for s in S:
            r_mod = mod_recips[s][primes.index(p)]
            new_sums = [(x + r_mod) % p for x in sums_mod]
            sums_mod.extend(new_sums)
        
        # Check for duplicates using a set
        if len(set(sums_mod)) != len(sums_mod):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Initialize random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Greedy with various orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    max_greedy_randoms = 100
    
    for _ in range(max_greedy_randoms):
        if time.time() > deadline - 20:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 20:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            new_set = current_set + [x]
            if _is_dissociated_subset(new_set, recips):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        if best_size >= 15:
            break

    # Strategy 2: Random sampling with modular hash check for speed
    # Use multiple large primes for collision-resistant fingerprinting
    # If subset sums are distinct over integers, they are almost certainly distinct mod p
    primes = [1000000007, 1000000009, 998244353]
    
    # Precompute modular reciprocals: 1/s mod p
    mod_recips = {}
    for s in range(1, 23):
        mod_recips[s] = []
        for p in primes:
            # Compute modular inverse of s mod p using Fermat's little theorem
            inv_s = pow(s, p-2, p)
            mod_recips[s].append(inv_s)
    
    max_target = min(22, max(best_size + 1, 14))
    
    for target_size in range(max_target, best_size, -1):
        if time.time() > deadline - 5:
            break
        
        # Adjust attempts based on size
        if target_size >= 14:
            max_attempts = 2000
        elif target_size >= 12:
            max_attempts = 10000
        else:
            max_attempts = 50000
            
        attempts = 0
        found = False
        
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            
            candidate = random.sample(universe, target_size)
            
            # First, fast modular check
            if _mod_hash_check(candidate, primes, mod_recips):
                # Verify with exact floating point check
                if _is_dissociated_subset(candidate, recips):
                    best_set = candidate
                    best_size = target_size
                    found = True
                    break
            attempts += 1
        
        if found:
            break

    # Fallback
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
