import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _construct_with_order(primes_order):
    """Construct admissible tuple processing primes in given order."""
    R = np.zeros(N + 1, dtype=bool)
    current_size = N + 1
    
    for p in primes_order:
        if p > current_size:
            break
            
        # Find residue class with minimum new removals
        best_r = 0
        min_new = N + 1
        
        for r in range(p):
            indices = np.arange(r, N + 1, p)
            new_count = np.count_nonzero(~R[indices])
            if new_count < min_new:
                min_new = new_count
                best_r = r
        
        # Update R
        indices = np.arange(best_r, N + 1, p)
        R[indices] = True
        current_size -= min_new
    
    return np.count_nonzero(~R)

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    # Strategy 1: Ascending order (standard)
    best_points = None
    best_size = 0
    
    R = np.zeros(N + 1, dtype=bool)
    current_size = N + 1
    
    for p in primes:
        if time.time() > deadline:
            break
        if p > current_size:
            break
            
        best_r = 0
        min_new = N + 1
        
        for r in range(p):
            indices = np.arange(r, N + 1, p)
            new_count = np.count_nonzero(~R[indices])
            if new_count < min_new:
                min_new = new_count
                best_r = r
        
        indices = np.arange(best_r, N + 1, p)
        R[indices] = True
        current_size -= min_new
    
    if best_size < current_size:
        best_size = current_size
        best_points = np.nonzero(~R)[0].tolist()
    
    # Strategy 2: Descending order (large primes first)
    if time.time() < deadline - 5:
        R = np.zeros(N + 1, dtype=bool)
        current_size = N + 1
        
        for p in reversed(primes):
            if time.time() > deadline:
                break
            if p > current_size:
                continue  # Still need to check smaller primes later
            
            best_r = 0
            min_new = N + 1
            
            for r in range(p):
                indices = np.arange(r, N + 1, p)
                new_count = np.count_nonzero(~R[indices])
                if new_count < min_new:
                    min_new = new_count
                    best_r = r
            
            indices = np.arange(best_r, N + 1, p)
            R[indices] = True
            current_size -= min_new
        
        if current_size > best_size:
            best_size = current_size
            best_points = np.nonzero(~R)[0].tolist()
    
    # Strategy 3: Process only primes that are still fully covered, in ascending order
    # But with a twist: check if we can skip some primes
    if time.time() < deadline - 5:
        R = np.zeros(N + 1, dtype=bool)
        current_size = N + 1
        
        for p in primes:
            if time.time() > deadline:
                break
            if p > current_size:
                break
            
            # Check if all residue classes are covered
            all_covered = True
            for r in range(p):
                indices = np.arange(r, N + 1, p)
                if np.all(R[indices]):
                    all_covered = False
                    break
            
            if not all_covered:
                continue
            
            best_r = 0
            min_new = N + 1
            
            for r in range(p):
                indices = np.arange(r, N + 1, p)
                new_count = np.count_nonzero(~R[indices])
                if new_count < min_new:
                    min_new = new_count
                    best_r = r
            
            indices = np.arange(best_r, N + 1, p)
            R[indices] = True
            current_size -= min_new
        
        if current_size > best_size:
            best_size = current_size
            best_points = np.nonzero(~R)[0].tolist()
    
    if best_points is None:
        # Fallback: return empty or minimal valid set
        return [0]
    
    return best_points
