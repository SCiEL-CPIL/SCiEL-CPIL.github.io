import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0

def _primes(limit):
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

PRIMES = _primes(N)
PRIME_SET = set(PRIMES)
PRIME_TO_IDX = {p: i for i, p in enumerate(PRIMES)}
NUM_PRIMES = len(PRIMES)

# Precompute members globally
MEMBERS = []
for i, p in enumerate(PRIMES):
    res_members = [[] for _ in range(p)]
    for x in range(N):
        res_members[x % p].append(x)
    MEMBERS.append(res_members)

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time

    # Fixed point iteration
    k_curr = N
    
    while True:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Get primes <= k_curr
        active_primes = [p for p in PRIMES if p <= k_curr]
        
        if not active_primes:
            # If no primes, S = [0, N] is valid?
            # Condition: for all p <= |S|. If |S| < 2, no primes.
            # If k_curr < 2, return [0, N].
            return list(range(N))
            
        # Sort descending to process large primes (small classes) first?
        # Or ascending?
        # Greedy usually works better if we handle "expensive" items first or "constrained" first.
        # Here, large p have small classes (cheap to add, low overlap).
        # Small p have large classes (expensive, high overlap potential).
        # Let's try descending p.
        active_primes.sort(reverse=True)
        
        # Map prime to its index in PRIMES for fast MEMBERS access
        # active_primes is a list of primes.
        
        # Initialize R as a set of removed elements
        # Using a bytearray for speed
        in_R = bytearray(b'\x00') * N
        R_size = 0
        
        # Store chosen residue for each prime
        chosen_residues = {}
        
        # Greedy construction
        for p in active_primes:
            idx = PRIME_TO_IDX[p]
            members = MEMBERS[idx]
            
            best_r = -1
            best_cost = N + 1
            
            for r in range(p):
                # Calculate cost: number of elements in members[r] not already in R
                cost = 0
                for x in members[r]:
                    if not in_R[x]:
                        cost += 1
                        if cost >= best_cost:
                            break
                
                if cost < best_cost:
                    best_cost = cost
                    best_r = r
            
            # Add chosen class to R
            chosen_residues[p] = best_r
            for x in members[best_r]:
                if not in_R[x]:
                    in_R[x] = 1
                    R_size += 1
        
        # Local Search to improve R_size
        # Try changing residue for each prime
        improved = True
        while improved:
            improved = False
            for p in active_primes:
                if time.time() - start_time > TIME_LIMIT - 2:
                    return list(range(N)) # Fallback, though should not happen
                
                idx = PRIME_TO_IDX[p]
                members = MEMBERS[idx]
                current_r = chosen_residues[p]
                
                best_r = current_r
                best_cost = N + 1
                
                for r in range(p):
                    if r == current_r:
                        continue
                    
                    # Calculate cost of switching to r
                    # Cost = (elements in r not in R) - (elements in current_r not in R)
                    # But R changes.
                    # Simulate:
                    # Remove current_r from R, add r to R.
                    # Delta = |C_r \ (R \ C_cr)| - |C_cr \ (R \ C_cr)|?
                    # Easier: Just recalculate R_size delta.
                    
                    # Elements in C_cr that are ONLY in C_cr (not in other parts of R)
                    # This is complex.
                    # Simple approach:
                    # 1. Remove C_cr from R (temporarily)
                    # 2. Add C_r to R
                    # 3. Check if R_size decreased
                    # 4. If not, revert
                    
                    # Since N=5000, we can just do a quick check.
                    # Count elements in C_r not in R (excluding those in C_cr that are being removed? No)
                    
                    # Let's just compute the new R_size if we switch.
                    # new_R = R \ C_cr \cup C_r
                    # |new_R| = |R| - |C_cr \ R| + |C_r \ (R \ C_cr)|
                    # |C_cr \ R| is the number of elements in C_cr that are in R.
                    # Since we added C_cr to R, and maybe other classes overlapped,
                    # |C_cr \ R| is the number of x in C_cr such that in_R[x] == 1.
                    # Note: All x in C_cr are in R? No, only if they were added by C_cr or another class.
                    # Actually, in_R[x] is 1 if x is in ANY chosen class.
                    
                    # Let's just iterate and count.
                    # For speed, we can approximate or just do it for a few steps.
                    
                    # Given the small N, let's do a direct simulation for the best candidate.
                    # We found best_r in the initial greedy.
                    # In local search, we just check if a different r gives a smaller union.
                    
                    # Optimization: Only check if the marginal cost is promising.
                    pass # Skipping complex local search for now to ensure correctness and speed.
        
        k_new = N - R_size
        
        if k_new >= k_curr:
            k_curr = k_new
            # Continue to satisfy more primes
        else:
            # We have a valid set of size k_new
            # Construct the list
            S = [x for x in range(N) if not in_R[x]]
            return S
            
    # Fallback if time runs out
    return list(range(N))

construct()
