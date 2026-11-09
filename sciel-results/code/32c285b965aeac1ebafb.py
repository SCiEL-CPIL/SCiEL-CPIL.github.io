"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from math import isqrt

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def _construct_with_threshold(primes, threshold, use_least_populated, rng=None):
    """Try a specific sieve threshold and residue strategy. Returns the resulting set."""
    if rng is None:
        rng = random.Random()
    S_mask = np.ones(N + 1, dtype=bool)
    
    # Sieve small primes up to threshold
    small_primes = primes[primes <= threshold]
    for p in small_primes:
        if use_least_populated:
            # Find least populated residue mod p in current S
            S_indices = np.where(S_mask)[0]
            if len(S_indices) == 0:
                break
            residues = S_indices % p
            counts = np.bincount(residues, minlength=p)
            # Randomize among tied minimums
            min_count = int(np.min(counts))
            tied = np.where(counts == min_count)[0]
            if len(tied) > 1:
                best_r = int(rng.choice(tied))
            else:
                best_r = int(tied[0])
            to_remove = S_indices[residues == best_r]
            S_mask[to_remove] = False
        else:
            # Remove residue 0
            multiples = np.arange(0, N + 1, p)
            S_mask[multiples] = False
    
    remaining_primes = primes[primes > threshold]
    
    changed = True
    while changed:
        S_indices = np.where(S_mask)[0]
        if len(S_indices) == 0:
            break
        
        candidates = remaining_primes[remaining_primes <= len(S_indices)]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p_val = -1
        best_tied_list = None
        
        for p in candidates:
            residues = S_indices % p
            unique_res = np.unique(residues)
            if len(unique_res) == p:
                counts = np.bincount(residues, minlength=p)
                min_count = int(np.min(counts))
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    tied = np.where(counts == min_count)[0]
                    if len(tied) > 1:
                        best_r = int(rng.choice(tied))
                        best_tied_list = tied
                    else:
                        best_r = int(tied[0])
                        best_tied_list = None
                    best_p_val = p
                    bad_p = p
        
        if bad_p is not None:
            to_remove_mask = (S_indices % best_p_val) == best_r
            values_to_remove = S_indices[to_remove_mask]
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
    
    return np.where(S_mask)[0].tolist()

def _is_admissible(S, primes):
    """Check if S is admissible: for every prime p <= len(S), some residue class mod p is missed."""
    S_set = set(S)
    n = len(S)
    for p in primes:
        if p > n:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return False
    return True

def _refine_local(S, primes, max_iters=50, rng=None):
    """Try to improve S by swapping elements to maintain admissibility while keeping size or improving robustness."""
    if rng is None:
        rng = random.Random()
    S_list = list(S)
    n = len(S_list)
    S_set = set(S_list)
    
    # Build list of elements not in S for potential swaps
    all_elements = list(range(N + 1))
    not_in_S = [x for x in all_elements if x not in S_set]
    
    for _ in range(max_iters):
        # Try random swaps: remove one element, add another
        if not not_in_S:
            break
        
        idx_remove = rng.randrange(n)
        elem_remove = S_list[idx_remove]
        elem_add = rng.choice(not_in_S)
        
        # Try the swap
        new_S_list = S_list[:]
        new_S_list[idx_remove] = elem_add
        new_S_set = set(new_S_list)
        
        if len(new_S_set) != n:
            continue
        
        # Check admissibility
        if _is_admissible(new_S_list, primes):
            # Accept the swap
            S_set.discard(elem_remove)
            S_set.add(elem_add)
            S_list[idx_remove] = elem_add
            not_in_S.remove(elem_add)
            not_in_S.append(elem_remove)
    
    return S_list

def construct():
    start_time = time.time()
    deadline = 295.0  # Safety margin
    
    primes = _primes(N)
    
    # Try multiple thresholds and strategies with randomization
    best_result = None
    best_size = 0
    
    thresholds = [50, 75, 100, 125, 150, 200]
    strategies = [False, True]  # False = residue 0, True = least populated
    
    # First pass: deterministic runs
    for threshold in thresholds:
        for use_least in strategies:
            if time.time() - start_time > deadline:
                break
            rng = random.Random(42)  # Fixed seed for reproducibility in first pass
            result = _construct_with_threshold(primes, threshold, use_least, rng)
            size = len(result)
            if size > best_size:
                best_size = size
                best_result = result
    
    # Second pass: randomized restarts to find better solutions
    num_random_restarts = 20
    for i in range(num_random_restarts):
        if time.time() - start_time > deadline:
            break
        
        rng = random.Random(i * 1000 + 7)
        threshold = rng.choice(thresholds)
        use_least = rng.choice(strategies)
        
        result = _construct_with_threshold(primes, threshold, use_least, rng)
        size = len(result)
        
        if size > best_size:
            best_size = size
            best_result = result
        elif size == best_size and best_result is not None:
            # Try local refinement on tied solutions
            refined = _refine_local(result, primes, max_iters=20, rng=rng)
            if len(refined) > best_size:
                best_size = len(refined)
                best_result = refined
    
    # Final refinement pass on the best result
    if best_result is not None and time.time() - start_time < deadline - 10:
        rng = random.Random(999)
        refined = _refine_local(best_result, primes, max_iters=100, rng=rng)
        if len(refined) > best_size:
            best_result = refined
            best_size = len(refined)
    
    if best_result is None:
        # Fallback: just return all numbers (shouldn't happen)
        best_result = list(range(N + 1))
    
    return best_result

# EVOLVE-BLOCK-END
