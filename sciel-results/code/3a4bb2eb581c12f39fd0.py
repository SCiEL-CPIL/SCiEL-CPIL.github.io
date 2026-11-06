import numpy as np
import time
import random

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
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    # Dynamic window strategy
    # Start small to explore quickly, expand if we cover the window.
    N_current = 5000
    N_max = 100000
    N_expansion_factor = 2
    
    best_residues = None
    best_score = 0
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    def build_counts(residues, N):
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

    def get_score(counts, N):
        # Find first index >= 1 where counts < 2
        # If no failure, return N
        # Using argmax on boolean array (True where count < 2)
        # If all are False, argmax returns 0. We need to check if any failure exists.
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1

    # Initial Strategies
    def init_random():
        return np.array([rng.integers(0, p) for p in primes], dtype=np.int64)
    
    def init_zero():
        return np.zeros(n_primes, dtype=np.int64)
        
    def init_one():
        return np.ones(n_primes, dtype=np.int64)
        
    def init_half():
        return np.array([p // 2 for p in primes], dtype=np.int64)
        
    initial_strategies = [
        init_random,
        init_zero,
        init_one,
        init_half,
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
    ]
    
    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
            
        residues = strat()
        N_cur = N_current
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Local Search with Dynamic Expansion
        stall_count = 0
        max_stall = 200 # Lower stall limit for faster restarts
        
        while time.time() < deadline - 5.0:
            # If we covered the current window, try to expand
            if current_score == N_cur:
                # Expand window
                if N_cur * N_expansion_factor > N_max:
                    break # Reached max size, can't expand further
                    
                N_cur = min(N_cur * N_expansion_factor, N_max)
                # Update counts for the new range
                # We need to add contributions for the new indices [old_N+1, new_N]
                # Actually, it's easier to just rebuild counts for the new N if the expansion is large,
                # or update incrementally.
                # Rebuilding is O(n_primes * N/p). For N=100000, this is fast enough (~1M ops).
                # Let's just rebuild to avoid bugs in incremental updates.
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                
                # If we still cover the new window, loop will expand again
                continue
            
            # Find the first failure
            x_fail = current_score
            
            # Move Selection
            # Strategy: 
            # 1. Pick a random prime index i.
            # 2. If p_i does NOT cover x_fail, set r_i = x_fail % p_i.
            # 3. If p_i DOES cover x_fail, we have two options:
            #    a. Randomly change r_i (might lose coverage of x_fail, risky)
            #    b. Pick a different prime.
            # To be more effective, let's bias towards primes that do NOT cover x_fail.
            # But checking which primes cover x_fail is O(n_primes).
            # We can just pick a random prime and check.
            
            i = int(rng.integers(n_primes))
            p = primes[i]
            r_old = int(residues[i])
            
            # Does p cover x_fail?
            covers_x = (x_fail % p == r_old)
            
            if covers_x:
                # This prime is contributing to x_fail.
                # Changing it might reduce count[x_fail] below 2.
                # We should probably pick a different prime to ADD coverage.
                # Let's retry with a different prime, or do a random move with lower probability.
                if rng.random() < 0.8:
                    # Pick a different prime
                    i = int(rng.integers(n_primes))
                    p = primes[i]
                    r_old = int(residues[i])
                    covers_x = (x_fail % p == r_old)
                    if covers_x:
                        # If we still pick a covering prime, just do a random perturbation
                        # to escape local minima, but accept only if score doesn't drop.
                        r_new = int(rng.integers(0, p))
                        if r_new == r_old:
                            continue
                        # Apply and check
                        if r_old == 0:
                            if p <= N_cur:
                                counts[p::p] -= 1
                        else:
                            if r_old <= N_cur:
                                counts[r_old::p] -= 1
                                
                        if r_new == 0:
                            if p <= N_cur:
                                counts[p::p] += 1
                        else:
                            if r_new <= N_cur:
                                counts[r_new::p] += 1
                                
                        new_score = get_score(counts, N_cur)
                        
                        if new_score > current_score:
                            accepted = True
                        elif new_score == current_score:
                            if rng.random() < 0.2:
                                accepted = True
                            else:
                                accepted = False
                        else:
                            accepted = False
                            
                        if accepted:
                            residues[i] = r_new
                            current_score = new_score
                            stall_count = 0
                            if current_score > best_score:
                                best_score = current_score
                                best_residues = residues.copy()
                        else:
                            # Revert
                            if r_new == 0:
                                if p <= N_cur:
                                    counts[p::p] -= 1
                            else:
                                if r_new <= N_cur:
                                    counts[r_new::p] -= 1
                                    
                            if r_old == 0:
                                if p <= N_cur:
                                    counts[p::p] += 1
                            else:
                                if r_old <= N_cur:
                                    counts[r_old::p] += 1
                                    
                            stall_count += 1
                            if stall_count > max_stall:
                                break
                        continue
                    else:
                        # We picked a non-covering prime in the retry
                        r_new = int(x_fail % p)
                else:
                    # Small chance to perturb a covering prime
                    r_new = int(rng.integers(0, p))
                    if r_new == r_old:
                        continue
                    # Apply and check (same logic as above)
                    if r_old == 0:
                        if p <= N_cur:
                            counts[p::p] -= 1
                    else:
                        if r_old <= N_cur:
                            counts[r_old::p] -= 1
                            
                    if r_new == 0:
                        if p <= N_cur:
                            counts[p::p] += 1
                    else:
                        if r_new <= N_cur:
                            counts[r_new::p] += 1
                            
                    new_score = get_score(counts, N_cur)
                    
                    if new_score > current_score:
                        accepted = True
                    elif new_score == current_score:
                        if rng.random() < 0.2:
                            accepted = True
                        else:
                            accepted = False
                    else:
                        accepted = False
                        
                    if accepted:
                        residues[i] = r_new
                        current_score = new_score
                        stall_count = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues.copy()
                    else:
                        # Revert
                        if r_new == 0:
                            if p <= N_cur:
                                counts[p::p] -= 1
                        else:
                            if r_new <= N_cur:
                                counts[r_new::p] -= 1
                                
                        if r_old == 0:
                            if p <= N_cur:
                                counts[p::p] += 1
                        else:
                            if r_old <= N_cur:
                                counts[r_old::p] += 1
                                
                        stall_count += 1
                        if stall_count > max_stall:
                            break
                    continue
            else:
                # p does NOT cover x_fail.
                # Set r_new to cover x_fail.
                r_new = int(x_fail % p)
                
                # Apply move
                if r_old == 0:
                    if p <= N_cur:
                        counts[p::p] -= 1
                else:
                    if r_old <= N_cur:
                        counts[r_old::p] -= 1
                        
                if r_new == 0:
                    if p <= N_cur:
                        counts[p::p] += 1
                else:
                    if r_new <= N_cur:
                        counts[r_new::p] += 1
                        
                new_score = get_score(counts, N_cur)
                
                # Acceptance
                if new_score > current_score:
                    accepted = True
                elif new_score == current_score:
                    if rng.random() < 0.3:
                        accepted = True
                    else:
                        accepted = False
                else:
                    accepted = False
                
                if accepted:
                    residues[i] = r_new
                    current_score = new_score
                    stall_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues.copy()
                else:
                    # Revert
                    if r_new == 0:
                        if p <= N_cur:
                            counts[p::p] -= 1
                    else:
                        if r_new <= N_cur:
                            counts[r_new::p] -= 1
                            
                    if r_old == 0:
                        if p <= N_cur:
                            counts[p::p] += 1
                    else:
                        if r_old <= N_cur:
                            counts[r_old::p] += 1
                            
                    stall_count += 1
                    if stall_count > max_stall:
                        break

    # Final Verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
        
    return best_residues.tolist()
