import time
import random
import math
import numpy as np

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

def _primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def _find_violating_prime(points, primes_list):
    """Find a prime p <= len(points) where all residues are covered, or None if admissible."""
    k = len(points)
    if k == 0:
        return None
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return p
    return None

def _min_residue_class(points, p):
    """Find the residue class mod p with the fewest elements."""
    counts = [0] * p
    for v in points:
        counts[v % p] += 1
    min_count = min(counts)
    min_residues = [r for r in range(p) if counts[r] == min_count]
    return min_residues, min_count

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    # Precompute residue vectors for all primes for fast counting
    # For each prime p, precompute v % p for v in range(N)
    prime_residues = {}
    for p in primes:
        if p <= N:
            prime_residues[p] = np.arange(N) % p
    
    # Strategy: Vectorized greedy repair with residue count tracking
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Start with all numbers
        in_set = np.ones(N, dtype=bool)
        
        # Maintain residue counts for each prime p <= current_size
        # We'll track which primes are "satisfied" (have a missing residue)
        
        max_iterations = 5000
        for _ in range(max_iterations):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            current_size = int(np.sum(in_set))
            if current_size == 0:
                break
            
            # Find a violating prime using vectorized operations
            violating_p = None
            for p in primes:
                if p > current_size:
                    break
                
                # Get residues of elements in set
                residues = prime_residues[p]
                mask = in_set
                # Check if all residues are covered
                covered = np.zeros(p, dtype=bool)
                for r in range(p):
                    if np.any(residues[mask] == r):
                        covered[r] = True
                
                if np.all(covered):
                    violating_p = p
                    break
            
            if violating_p is None:
                # Admissible
                break
            
            # Find the residue class with the fewest elements in the violating prime
            residues = prime_residues[violating_p]
            mask = in_set
            counts = np.zeros(violating_p, dtype=np.int64)
            for r in range(violating_p):
                counts[r] = np.sum(residues[mask] == r)
            
            # Pick the minimum count residue class
            min_idx = int(np.argmin(counts))
            
            # Remove all elements in this residue class
            to_remove = (residues == min_idx) & mask
            in_set[to_remove] = False
        
        # Convert to set for adding back
        current_set_set = set(np.where(in_set)[0].tolist())
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
        
        # Try to add back elements to improve size
        max_add_passes = 15
        for pass_num in range(max_add_passes):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
                
            # Get candidates: elements not in current_set
            candidates = [x for x in range(N) if x not in current_set_set]
            random.shuffle(candidates)
            
            # Limit candidates to check to avoid timeout
            check_limit = 2000
            added_any = False
            for x in candidates[:check_limit]:
                if time.time() - start_time > TIME_LIMIT - 2:
                    break
                
                # Check if current_set U {x} is admissible
                new_size = len(current_set_set) + 1
                is_adm = True
                for p in primes:
                    if p > new_size:
                        break
                    # Check if all residues mod p are covered in current_set U {x}
                    seen = bytearray(p)
                    count = 0
                    for v in current_set_set:
                        r = v % p
                        if not seen[r]:
                            seen[r] = 1
                            count += 1
                            if count == p:
                                break
                    rx = x % p
                    if not seen[rx]:
                        count += 1
                        if count == p:
                            is_adm = False
                            break
                if is_adm:
                    current_set_set.add(x)
                    added_any = True
            
            # If we added none, we are likely at a local maximum
            if not added_any:
                break
        
        if len(current_set_set) > best_size:
            best_set = set(current_set_set)
            best_size = len(current_set_set)
            
    return list(best_set)

# Initial call to set start time and warm up
construct()
