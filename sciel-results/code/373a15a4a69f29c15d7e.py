import numpy as np
import time
import random
import math

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # We aim for L up to 100,000 to allow for the possibility of reaching 1000.
    # The score is capped at 10^5, so checking up to 100,000 is sufficient.
    N = 100000
    
    best_residues = None
    best_score = 0
    
    # Function to build counts from residues
    def build_counts(residues):
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        return counts

    # Function to get score from counts
    def get_score(counts):
        # We care about x=1..N
        # Find first index >= 1 where counts < 2
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1

    # Function to update counts for a single residue change
    def update_counts(counts, i, p, r_old, r_new):
        if r_old == 0:
            if p <= N:
                counts[p::p] -= 1
        else:
            if r_old <= N:
                counts[r_old::p] -= 1
                
        if r_new == 0:
            if p <= N:
                counts[p::p] += 1
        else:
            if r_new <= N:
                counts[r_new::p] += 1

    # Heuristic initializations
    initial_strategies = [
        lambda i, p: random.randint(0, p - 1),
        lambda i, p: 1 % p,
        lambda i, p: (p // 2) % p,
        lambda i, p: (i * 13 + 5) % p,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: (i * 31 + 11) % p,
        lambda i, p: (i * 17 + 7) % p,
        lambda i, p: (i * 23 + 19) % p,
    ]
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    # Run multiple restarts
    for strat_idx, strat in enumerate(initial_strategies):
        if time.time() > deadline - 10.0:
            break
            
        residues = np.array([strat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = build_counts(residues)
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
        
        # Local Search / Hill Climbing
        # Use a more aggressive plateau escape strategy
        stall_count = 0
        max_stall = 1500  # Reduced to allow more restarts
        
        while time.time() < deadline - 5.0:
            # Pick a random prime to mutate
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            
            # Pick new residue
            # Strategy: If the current point is under-covered, try to fix it.
            # If over-covered, try to free up coverage elsewhere or just random.
            # For simplicity and robustness, we use random mutation but with a
            # slightly higher acceptance rate for neutral moves to escape plateaus.
            r_new = int(rng.integers(0, p))
            
            if r_old == r_new:
                continue
                
            # Update counts
            update_counts(counts, i, p, r_old, r_new)
            
            new_score = get_score(counts)
            
            # Acceptance Criterion
            accepted = False
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                # Accept with probability 0.6 to allow random walk on plateaus
                # This helps escape local minima where the score doesn't change
                # immediately but might improve after a few neutral steps.
                if rng.random() < 0.6:
                    accepted = True
            
            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert counts
                update_counts(counts, i, p, r_new, r_old)
                
                stall_count += 1
                if stall_count > max_stall:
                    break # Restart with new initialization
    
    # Ensure we have a valid result to return
    if best_residues is None:
        # Fallback: return a simple valid set of residues
        best_residues = np.array([1 % p for p in primes], dtype=np.int64)
        
    return best_residues.tolist()
