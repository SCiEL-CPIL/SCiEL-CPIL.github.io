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
    
    # Search range: We aim for L up to 20,000 for faster iteration.
    # If we cover 20,000, we can check further in a final verification step.
    N = 20000
    
    best_residues = None
    best_score = 0
    best_counts = None
    
    # Heuristic initializations
    initial_strategies = [
        lambda i, p: random.randint(0, p - 1),
        lambda i, p: 1 % p,
        lambda i, p: (p // 2) % p,
        lambda i, p: (i * 13 + 5) % p,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: (i * 31 + 11) % p,
        lambda i, p: (i * 17 + 2) % p,
    ]
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
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

    def get_score(counts):
        # We care about x=1..N
        # Find first index >= 1 where counts < 2
        # We can optimize this by checking from the end if we expect high scores, 
        # but np.nonzero is quite fast for 20k.
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1

    # Run multiple restarts
    for strat in initial_strategies:
        if time.time() > deadline - 5.0:
            break
            
        residues = np.array([strat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = build_counts(residues)
        current_score = get_score(counts)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            best_counts = counts.copy()
            
        # Local Search
        stall_count = 0
        max_stall = 1500 # Reset or restart if no improvement for this many steps
        
        while time.time() < deadline - 2.0:
            # Decide move type
            # 50% random, 50% directed
            move_type = rng.random()
            
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            
            if move_type < 0.5:
                # Random move
                r_new = int(rng.integers(0, p))
            else:
                # Directed move: Try to cover the first failure
                # First failure is at current_score + 1
                target = current_score + 1
                if target > N:
                    # If we covered the whole range, pick a random target in the middle
                    target = N // 2
                r_new = target % p
            
            if r_old == r_new:
                continue
                
            # Update counts
            # Subtract old
            if r_old == 0:
                if p <= N:
                    counts[p::p] -= 1
            else:
                if r_old <= N:
                    counts[r_old::p] -= 1
                    
            # Add new
            if r_new == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r_new <= N:
                    counts[r_new::p] += 1
            
            new_score = get_score(counts)
            
            # Acceptance Criterion
            accepted = False
            if new_score > current_score:
                accepted = True
            elif new_score == current_score:
                # Accept with probability 0.3 to allow random walk on plateaus
                if rng.random() < 0.3:
                    accepted = True
            
            if accepted:
                residues[i] = r_new
                current_score = new_score
                stall_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                    best_counts = counts.copy()
            else:
                # Revert counts
                if r_new == 0:
                    if p <= N:
                        counts[p::p] -= 1
                else:
                    if r_new <= N:
                        counts[r_new::p] -= 1
                        
                if r_old == 0:
                    if p <= N:
                        counts[p::p] += 1
                else:
                    if r_old <= N:
                        counts[r_old::p] += 1
                        
                stall_count += 1
                if stall_count > max_stall:
                    break # Restart with new initialization
            
    # Final Verification: Check if we can go beyond N
    if best_residues is not None and time.time() < deadline - 1.0:
        # Check up to 100,000
        N_verify = 100000
        counts_v = np.zeros(N_verify + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(best_residues[i])
            if r == 0:
                if p <= N_verify:
                    counts_v[p::p] += 1
            else:
                if r <= N_verify:
                    counts_v[r::p] += 1
        
        failures = np.nonzero(counts_v[1:N_verify+1] < 2)[0]
        if len(failures) == 0:
            final_score = N_verify
        else:
            final_score = int(failures[0]) + 1
            
        # The score returned by construct is just the residues.
        # The evaluator calculates the score.
        # We just need to return the best residues found.
        
    return best_residues.tolist()
