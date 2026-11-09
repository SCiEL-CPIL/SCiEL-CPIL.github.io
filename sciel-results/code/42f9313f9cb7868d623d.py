import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(arr, n, all_primes):
    """Check if arr (length n) is admissible."""
    if n == 0:
        return True
    arr_set = set(arr)
    for p in all_primes:
        if p > n:
            break
        residues = set(x % p for x in arr_set)
        if len(residues) == p:
            return False
    return True


def _remove_residue_class(arr, p, target_res):
    """Remove all elements from arr that are congruent to target_res mod p."""
    return [x for x in arr if x % p != target_res]


def _find_best_residue_class(arr, p):
    """Find the residue class mod p whose removal minimizes the loss.
    Returns the residue class index."""
    counts = {}
    for x in arr:
        r = x % p
        counts[r] = counts.get(r, 0) + 1
    
    if not counts:
        return 0
    
    # Find the residue class with minimum count
    min_count = min(counts.values())
    # Among those with min count, pick the one that causes least future damage
    # Use a heuristic: prefer residue classes that are "spread out" less
    candidates = [r for r, c in counts.items() if c == min_count]
    
    if len(candidates) == 1:
        return candidates[0]
    
    # For ties, use a more sophisticated heuristic:
    # Prefer removing classes that have elements more concentrated in certain ranges
    # This is a proxy for future constraint satisfaction
    best_res = candidates[0]
    best_score = float('inf')
    
    for r in candidates:
        elements = [x for x in arr if x % p == r]
        if not elements:
            continue
        # Score: standard deviation (lower = more concentrated = better to remove)
        if len(elements) > 1:
            std = np.std(elements)
        else:
            std = 0
        # Also consider: does removing this class help with other primes?
        # For now, just use std as the tie-breaker
        if std < best_score:
            best_score = std
            best_res = r
    
    return best_res


def _greedy_sieve(all_primes, time_limit, start_time):
    """Run the greedy sieve algorithm with random tie-breaking."""
    alive = np.ones(N + 1, dtype=bool)
    alive_count = N + 1
    
    for p in all_primes:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        if p > alive_count:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        residues = alive_indices % p
        counts = np.bincount(residues, minlength=p)
        
        if np.min(counts) == 0:
            continue
        
        # Find all residue classes with minimum count
        min_count = np.min(counts)
        min_classes = np.where(counts == min_count)[0]
        
        if len(min_classes) == 1:
            min_res = int(min_classes[0])
        else:
            # Random tie-breaking among minimum classes
            min_res = int(np.random.choice(min_classes))
        
        to_remove_mask = (residues == min_res)
        remove_indices = alive_indices[to_remove_mask]
        
        alive[remove_indices] = False
        alive_count -= len(remove_indices)
    
    return alive


def construct():
    """Construct the largest admissible tuple in [0, N].
    
    An admissible tuple satisfies: for every prime p <= len(tuple),
    some residue class mod p is completely missed by the tuple.
    """
    start_time = time.time()
    TIME_LIMIT = 290  # Safety margin before 300s evaluator kill
    
    # Precompute primes up to N+1 (max possible length)
    all_primes = _primes(N + 1)
    
    best_result = []
    best_size = 0
    
    # Run multiple random restarts with different tie-breaking
    np.random.seed(42)  # Fixed seed for reproducibility
    
    max_restarts = 50
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Run greedy sieve with random tie-breaking
        alive = _greedy_sieve(all_primes, TIME_LIMIT - 5, start_time)
        result_indices = np.where(alive)[0]
        result = result_indices.tolist()
        
        # Verify admissibility
        n = len(result)
        if n > best_size:
            if _is_admissible(result, n, all_primes):
                best_result = result
                best_size = n
        
        # If we've found a very good result, we can stop early
        if best_size > 0.95 * (N + 1):
            break
    
    # If no valid result found, fall back to the deterministic approach
    if best_size == 0:
        alive = np.ones(N + 1, dtype=bool)
        alive_count = N + 1
        
        for p in all_primes:
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT:
                break
            
            if p > alive_count:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            residues = alive_indices % p
            counts = np.bincount(residues, minlength=p)
            
            if np.min(counts) == 0:
                continue
            
            min_res = int(np.argmin(counts))
            to_remove_mask = (residues == min_res)
            remove_indices = alive_indices[to_remove_mask]
            
            alive[remove_indices] = False
            alive_count -= len(remove_indices)
        
        result_indices = np.where(alive)[0]
        best_result = result_indices.tolist()
        best_size = len(best_result)
    
    # Final verification and repair
    n = len(best_result)
    if n == 0:
        return best_result
    
    result_set = set(best_result)
    primes_to_check = _primes(n)
    
    for p in primes_to_check:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        residues_covered = set(x % p for x in result_set)
        if len(residues_covered) == p:
            class_members = {}
            for x in result_set:
                r = x % p
                if r not in class_members:
                    class_members[r] = []
                class_members[r].append(x)
            
            min_class = min(class_members.values(), key=len)
            for x in min_class:
                result_set.discard(x)
            n = len(result_set)
    
    return sorted(result_set)

# EVOLVE-BLOCK-END
