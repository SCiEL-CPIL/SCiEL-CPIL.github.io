"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(residues, primes, cap=10**5):
    """Compute the largest L such that every integer 1..L is covered by at least 2 congruences."""
    # For each prime p with residue r, it covers integers x where x ≡ r (mod p), i.e., x = r + kp.
    # We need to count, for each x in 1..cap, how many primes cover x.
    # Efficient approach: for each prime, iterate over its covered positions and increment a count array.
    
    n = len(primes)
    # count[x] = number of congruences satisfied by x
    # We only care about x in 1..cap
    # But cap can be up to 10^5, and we have 168 primes.
    # For each prime p, the number of covered positions in 1..cap is roughly cap/p.
    # Total work: sum over primes of cap/p ≈ cap * sum(1/p) ≈ cap * log(log(1000)) which is fine.
    
    counts = [0] * (cap + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        # First covered integer >= 1
        # x ≡ r (mod p), x >= 1
        if r == 0:
            # x = p, 2p, 3p, ...
            first = p
        else:
            first = r
            if first < 1:
                first += p
        # Iterate from first to cap, step p
        x = first
        while x <= cap:
            counts[x] += 1
            x += p
    
    # Find largest L such that counts[x] >= 2 for all x in 1..L
    L = 0
    for x in range(1, cap + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _compute_L_fast(residues, primes, cap=10**5):
    """Faster computation using a different approach - check incrementally."""
    # Same as above but we can early-terminate
    counts = [0] * (cap + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
            if first < 1:
                first += p
        x = first
        while x <= cap:
            counts[x] += 1
            x += p
    
    L = 0
    for x in range(1, cap + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _greedy_construct(primes, max_time=140.0):
    """Greedy + local search approach to maximize the doubly-covered prefix length."""
    start_time = time.time()
    deadline = start_time + max_time
    
    n = len(primes)
    
    # Initial assignment: try to cover small integers densely
    # Strategy: assign residues so that each integer 1..K is hit by at least 2 primes
    # For integer x, the primes that can cover it are all primes (since x mod p is defined for all p)
    # But we want to maximize the prefix where every x has count >= 2.
    
    # Start with a reasonable initial assignment
    # Use the seed idea but improved: assign r_p = x mod p for some x
    # Actually, let's start with r_p = 1 for all p, then improve.
    
    # Better initial: for each prime p, choose r_p to maximize coverage of small integers
    # Let's start with r_p = p - 1 (covers 0, p-1, 2p-1, ...) or r_p = 1 (covers 1, p+1, ...)
    
    # Initial: r_p = 1 for all
    residues = [1] * n
    
    # Compute initial L
    best_L = _compute_L(residues, primes)
    best_residues = residues[:]
    
    # Local search: try changing one residue at a time
    # For each prime p_i, try all possible residues and keep the best
    # This is O(n * p_i * cost_of_eval) per full pass
    # cost_of_eval is O(cap * n / avg_p) which is manageable
    
    # To speed up, we can use incremental updates
    # But for simplicity, let's do full recomputation but limit the number of passes
    
    random.seed(42)
    
    for iteration in range(200):  # Limit iterations
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Shuffle the order of primes to avoid bias
        order = list(range(n))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 5:
                break
            p = primes[i]
            old_r = residues[i]
            
            # Try a subset of residues (not all p, since p can be up to 997)
            # Try: current, 0, 1, p-1, and some random ones
            candidates = set([old_r, 0, 1, p - 1])
            # Add some random candidates
            for _ in range(min(20, p)):
                candidates.add(random.randint(0, p - 1))
            
            best_r = old_r
            best_L_local = best_L
            
            for r in candidates:
                if r == old_r:
                    continue
                residues[i] = r
                L = _compute_L(residues, primes)
                residues[i] = old_r
                
                if L > best_L_local:
                    best_L_local = L
                    best_r = r
            
            if best_r != old_r:
                residues[i] = best_r
                best_L = best_L_local
                improved = True
                best_residues = residues[:]
        
        if not improved:
            # No improvement in this pass, try a random restart of some residues
            # Perturb a few random residues
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, n - 1)
                p = primes[i]
                residues[i] = random.randint(0, p - 1)
    
    return best_residues, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    n = len(primes)
    
    # Get a good initial solution via greedy + local search
    residues, best_L = _greedy_construct(primes, max_time=120.0)
    
    # Final check
    if time.time() > deadline - 2:
        return residues
    
    # Verify and return
    final_L = _compute_L(residues, primes)
    
    return residues
# EVOLVE-BLOCK-END
