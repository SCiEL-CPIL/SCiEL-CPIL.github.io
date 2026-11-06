"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_counts(residues, primes, max_check):
    """Compute coverage counts for x=1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _find_L(counts, max_check):
    """Find largest L such that all x in 1..L have count >= 2."""
    # Check from 1 to max_check
    # We want the first index i (1-based) where count < 2
    # counts[0] is unused for x=0, but our loop starts at 1.
    # We look at counts[1:max_check+1]
    below = np.where(counts[1:max_check+1] < 2)[0]
    if len(below) == 0:
        return max_check
    return below[0] # This is the 0-based index in the slice, so it corresponds to x = below[0] + 1? 
    # Wait. counts[1] corresponds to x=1.
    # If counts[1] < 2, below[0] is 0. L should be 0? 
    # The problem says "every integer 1..L". If 1 is not covered by 2, L=0.
    # If counts[2] < 2 (and counts[1]>=2), below[0] is 1 (index in slice). x = 1+1=2. L=1.
    # So L = below[0].
    # Let's verify: 
    # x=1 is counts[1]. If counts[1]<2, violation at slice index 0. L=0. Correct.
    # x=2 is counts[2]. If counts[1]>=2, counts[2]<2, violation at slice index 1. L=1. Correct.
    return below[0]

def _apply_change(counts, r, p, max_check, sign):
    """Apply or revert a residue change. sign=1 for add, -1 for remove."""
    start = r if r > 0 else p
    if start <= max_check:
        counts[start::p] += sign

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    max_check = 5000 # Check up to 5000 to be safe, L can grow
    
    start_time = time.time()
    deadline = start_time + 140 # 10s safety margin
    
    # Initialize residues to 0
    residues = [0] * n
    counts = _compute_counts(residues, primes, max_check)
    current_L = _find_L(counts, max_check)
    
    # Local Search
    # We will iterate through primes and try to improve L
    # To avoid getting stuck in local minima, we might restart or use randomization
    
    # Strategy: 
    # 1. Greedy initialization is often weak. 
    # 2. Random initialization + Local Search is robust.
    # Let's do a few random restarts if time permits.
    
    best_residues = residues.copy()
    best_L = current_L
    
    # Number of restarts depends on time.
    # Each restart: Init (fast), Local Search (slow).
    # Let's do 1 restart with a good heuristic init, then local search.
    
    # Heuristic Init: 
    # For small primes, 0 is often good? 
    # Let's try random init for diversity.
    
    np.random.seed(42) # Fixed seed for reproducibility in debugging, but maybe random is better?
    # The task says "Maximize". Randomness helps escape local minima.
    
    max_restarts = 5
    restarts_done = 0
    
    while restarts_done < max_restarts and (time.time() - start_time) < deadline:
        restarts_done += 1
        
        # Initialization
        if restarts_done == 1:
            # Start with 0s
            residues = [0] * n
        else:
            # Random initialization
            residues = [np.random.randint(0, p) for p in primes]
        
        counts = _compute_counts(residues, primes, max_check)
        current_L = _find_L(counts, max_check)
        
        # Local Search Loop
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            # Shuffle primes to avoid bias? Or just iterate.
            # Iterating in order is fine.
            for i in range(n):
                if (time.time() - start_time) > deadline:
                    break
                
                p = primes[i]
                current_r = residues[i]
                
                # Generate candidates
                # 1. Current
                # 2. Residues that cover the current bottleneck (current_L + 1)
                # 3. A few random ones
                
                candidates = {current_r}
                
                # Target the frontier
                target_x = current_L + 1
                if target_x <= max_check:
                    # We want to cover target_x
                    target_r = target_x % p
                    candidates.add(target_r)
                    
                    # Also cover neighbors
                    for delta in range(-3, 4):
                        nx = target_x + delta
                        if 1 <= nx <= max_check:
                            candidates.add(nx % p)
                
                # Add some random candidates
                for _ in range(5):
                    candidates.add(np.random.randint(0, p))
                
                best_candidate = current_r
                best_candidate_L = current_L
                
                for new_r in candidates:
                    if new_r == current_r:
                        continue
                    
                    # Try the change
                    _apply_change(counts, current_r, p, max_check, -1)
                    _apply_change(counts, new_r, p, max_check, 1)
                    
                    new_L = _find_L(counts, max_check)
                    
                    # Revert
                    _apply_change(counts, new_r, p, max_check, -1)
                    _apply_change(counts, current_r, p, max_check, 1)
                    
                    if new_L > best_candidate_L:
                        best_candidate_L = new_L
                        best_candidate = new_r
                
                # Apply best improvement
                if best_candidate != current_r:
                    _apply_change(counts, current_r, p, max_check, -1)
                    _apply_change(counts, best_candidate, p, max_check, 1)
                    residues[i] = best_candidate
                    current_L = best_candidate_L
                    improved = True
        
        # Update global best
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
    return best_residues
# EVOLVE-BLOCK-END
