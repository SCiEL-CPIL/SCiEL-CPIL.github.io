import time
import random
import math
import sys
from functools import lru_cache

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _greedy_construct():
    """
    Fast greedy construction.
    Start with all numbers.
    While there is a violation for some prime p <= len(S):
        Find a prime p <= len(S) such that all residues mod p are present.
        Remove the residue class with the minimum count.
    This is the standard heuristic that usually gets close to the optimum.
    """
    S = set(range(N + 1))
    # We need to check primes up to len(S). As S shrinks, the max prime to check shrinks.
    while True:
        len_S = len(S)
        viol_p = None
        # Only primes p <= len_S can be violated (since we need p distinct residues, 
        # and if p > len_S, we can't have p distinct residues in a set of size len_S)
        for p in PRIMES:
            if p > len_S:
                break
            # Check if all residues mod p are present in S
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_p = p
                break
        
        if viol_p is None:
            break
            
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        
        min_count = min(counts)
        # Pick the residue with min count. Tie-break randomly or deterministically?
        # Deterministic is faster, random helps explore. Let's do deterministic first, 
        # then random restarts in the main loop.
        r_to_remove = min(r for r in range(p) if counts[r] == min_count)
        
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _greedy_construct_randomized(rng):
    """
    Randomized version of greedy.
    """
    S = set(range(N + 1))
    while True:
        len_S = len(S)
        viol_p = None
        # Find all violating primes
        viol_primes = []
        for p in PRIMES:
            if p > len_S:
                break
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_primes.append(p)
        
        if not viol_primes:
            break
            
        # Pick a random violating prime
        p = rng.choice(viol_primes)
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        
        # Pick a random residue among those with minimum count
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        r_to_remove = rng.choice(candidates)
        
        S = {x for x in S if x % p != r_to_remove}
        
    return list(S)

def _local_search_improve(S, rng, time_limit=10.0):
    """
    Try to improve the set by adding back elements that were removed.
    This is a form of local search: try adding x, if it violates, remove it.
    If it doesn't violate, keep it.
    """
    start_time = time.time()
    S_set = set(S)
    excluded = [x for x in range(N + 1) if x not in S_set]
    rng.shuffle(excluded)
    
    # We need to quickly check if adding x violates any constraint.
    # A constraint for prime p is violated if ALL residues mod p are present.
    # Adding x with residue r mod p can only cause a violation if:
    # 1. p <= len(S) + 1
    # 2. Before adding, residue r was the ONLY missing residue for p.
    #    i.e., all other residues 0..p-1 (except r) are present in S.
    
    # To check this efficiently, we can maintain the set of "almost satisfied" primes.
    # Or, since N is small (5000), we can just check on the fly if we are careful.
    # However, checking all primes for each x is O(pi(N) * N) which is ~669 * 5000 = 3.3M ops.
    # With 5000 excluded elements, that's 16B ops. Too slow.
    
    # Optimization: Only primes p <= len(S) + 1 matter.
    # Also, if a prime p is not "close to being violated" (i.e., more than 1 residue missing),
    # adding one element won't violate it.
    
    # Let's precompute for each x, the primes p where x % p is the "critical" residue?
    # That's hard to precompute dynamically.
    
    # Alternative: Just try adding elements in random order, and if it fails, skip.
    # To make the check fast, we can maintain a count of residues for each prime?
    # That requires O(pi(N) * N) memory and update time per insertion.
    # Insertion: update counts for all p. O(pi(N)) ~ 669 ops.
    # Check violation: check if all counts > 0. O(p).
    # Total per insertion attempt: O(sum of p for active primes)? No, just O(pi(N)) for update, 
    # and we only need to check if the NEW count made a prime fully covered.
    # A prime becomes fully covered only if the count of the added residue was 0 before.
    # So we only need to check if count[x%p] was 0 before, and if so, if all other counts are > 0.
    
    # Let's implement this incremental check.
    
    active_primes = [p for p in PRIMES if p <= len(S) + 5] # Safe bound
    
    residue_counts = {}
    for p in active_primes:
        rc = [0] * p
        for x in S_set:
            rc[x % p] += 1
        residue_counts[p] = rc
        
    # Also track how many residues are missing for each prime
    missing_counts = {}
    for p in active_primes:
        rc = residue_counts[p]
        missing_counts[p] = sum(1 for c in rc if c == 0)
        
    for x in excluded:
        if time.time() - start_time > time_limit:
            break
            
        n_new = len(S_set) + 1
        is_valid = True
        
        # Only primes p <= n_new can be violated
        for p in active_primes:
            if p > n_new:
                break
                
            r = x % p
            # If this residue was already present (count > 0), adding x doesn't change "missing" status
            if residue_counts[p][r] > 0:
                continue
                
            # If residue was missing (count == 0), adding x makes it present.
            # This can only cause a violation if this was the LAST missing residue.
            # i.e., missing_counts[p] == 1.
            if missing_counts[p] == 1:
                is_valid = False
                break
                
        if is_valid:
            # Add x
            S_set.add(x)
            for p in active_primes:
                if p > len(S_set):
                    break
                r = x % p
                if residue_counts[p][r] == 0:
                    missing_counts[p] -= 1
                residue_counts[p][r] += 1
            
    return list(S_set)

def construct():
    start_time = time.time()
    
    # Phase 1: Deterministic Greedy
    best_S = _greedy_construct()
    best_size = len(best_S)
    
    # Phase 2: Local Search Improvement
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 5:
        rng = random.Random(42)
        improved_S = _local_search_improve(best_S, rng, time_limit=min(10, time_left - 2))
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(best_S)
            
    # Phase 3: Randomized Greedy Restarts
    # The greedy method can get stuck in local optima depending on tie-breaking.
    # Randomized restarts often find better solutions.
    time_left = MAX_TIME - (time.time() - start_time)
    rng = random.Random(123)
    
    while time.time() - start_time < MAX_TIME - 10:
        S_rand = _greedy_construct_randomized(rng)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(best_S)
            
        # Try to improve the new candidate
        time_left_inner = MAX_TIME - (time.time() - start_time)
        if time_left_inner > 5:
            S_improved = _local_search_improve(S_rand, rng, time_limit=min(5, time_left_inner - 2))
            if len(S_improved) > best_size:
                best_S = S_improved
                best_size = len(S_improved)
                
    return best_S

# Precompute primes at module level
PRIMES = _primes(N)

if __name__ == "__main__":
    result = construct()
    print(f"Best size: {len(result)}")
