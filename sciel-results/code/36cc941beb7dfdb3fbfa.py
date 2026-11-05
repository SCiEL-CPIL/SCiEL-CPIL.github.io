import time
import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # S is represented as a list of active integers
    S = list(range(N + 1))
    active = [True] * (N + 1)
    
    # We will iterate until no violations or time runs out
    # To avoid infinite loops, we can limit iterations or rely on size decrease
    
    while time.time() < deadline:
        M = len(S)
        if M == 0:
            break
        
        # Find all violating primes p <= M
        # A prime p violates if all residues mod p are present in S
        violating_primes = []
        
        # Only check primes up to M
        # We can filter PRIMES list
        # Since M decreases, we can just iterate PRIMES and break if p > M
        for p in PRIMES:
            if p > M:
                break
            
            # Check if S covers all residues mod p
            # Optimization: if M < p, it's impossible to cover all p residues, so no violation.
            # But we already checked p <= M.
            
            # Count residues
            # Using a list of size p
            counts = [0] * p
            # Iterate through S
            # S is a list of ints
            for x in S:
                counts[x % p] += 1
            
            # Check if any count is 0
            if min(counts) == 0:
                # Admissible for this p
                continue
            else:
                # Violation
                # Cost is min(counts)
                violating_primes.append((p, min(counts)))
        
        if not violating_primes:
            return S
        
        # Select the prime with minimum removal cost
        # If tie, pick the smallest prime (heuristic: more regular structure)
        violating_primes.sort(key=lambda x: (x[1], x[0]))
        p_star, min_cost = violating_primes[0]
        
        # Find the residue r* that has the minimum count
        # We need to recompute counts for p_star to find the argmin
        # Or we can store it in the loop above? 
        # Storing argmin in the loop is more efficient.
        
        # Let's refactor to store argmin
        # Re-doing the selection loop to store argmin
        best_p = None
        best_min_cost = float('inf')
        best_r = None
        
        for p in PRIMES:
            if p > M:
                break
            
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            
            m = min(counts)
            if m > 0: # Violation
                if m < best_min_cost:
                    best_min_cost = m
                    best_p = p
                    best_r = counts.index(m)
        
        if best_p is None:
            return S
            
        # Remove all x in S such that x % best_p == best_r
        # Create a new list
        # Or modify in place? 
        # Modifying in place with a boolean mask is faster.
        
        # Update active and S
        # S = [x for x in S if x % best_p != best_r]
        # This is O(M).
        
        # Check time again before expensive operation?
        if time.time() > deadline:
            break
            
        S = [x for x in S if x % best_p != best_r]
        
    return S
