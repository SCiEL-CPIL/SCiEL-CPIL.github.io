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
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
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
        failures_mask = counts[1:N+1] < 2
        if not np.any(failures_mask):
            return N
        return int(np.argmax(failures_mask)) + 1

    def local_search(residues, N_start):
        N_cur = N_start
        # We will track the current score
        counts = build_counts(residues, N_cur)
        current_score = get_score(counts, N_cur)
        
        stall_count = 0
        max_stall = 500
        last_improvement_time = time.time()
        
        while time.time() < deadline - 5.0:
            # If we covered the current window, try to expand
            if current_score == N_cur:
                if N_cur >= 100000:
                    break
                N_cur = min(N_cur * 2, 100000)
                counts = build_counts(residues, N_cur)
                current_score = get_score(counts, N_cur)
                last_improvement_time = time.time()
                continue
            
            # Find the first failure
            x_fail = current_score
            
            # We need to cover x_fail.
            # Identify primes that do NOT cover x_fail.
            # For each such prime, calculate the "damage" of reassigning it to cover x_fail.
            # Damage = number of y in [1, N_cur] such that y is currently covered by this prime
            # and counts[y] == 2 (so removing this prime's coverage will break y).
            
            best_prime_idx = -1
            best_damage = float('inf')
            best_r_new = 0
            
            # Iterate through all primes
            for i, p in enumerate(primes):
                r_old = int(residues[i])
                
                # Check if this prime covers x_fail
                if x_fail % p == r_old:
                    continue # This prime already covers x_fail, changing it might lose coverage. 
                             # We prefer primes that don't cover it, to add coverage.
                             # However, if all primes cover it, we are stuck. 
                             # But we need count >= 2. If count[x_fail] is 1, we need to add 1.
                             # If count[x_fail] is 0, we need to add 2.
                             # So we should only pick primes that do NOT cover it.
                
                r_new = x_fail % p
                
                # Calculate damage
                # Damage is the number of y in [1, N_cur] where y == r_old (mod p) and counts[y] == 2
                if r_old == 0:
                    if p <= N_cur:
                        # indices: p, 2p, ...
                        damage = np.sum(counts[p::p] == 2)
                else:
                    if r_old <= N_cur:
                        # indices: r_old, r_old+p, ...
                        damage = np.sum(counts[r_old::p] == 2)
                    else:
                        damage = 0
                
                # We want to minimize damage
                # Add a small random noise to break ties and avoid deterministic loops
                damage_with_noise = damage + rng.random() * 0.1
                
                if damage_with_noise < best_damage:
                    best_damage = damage_with_noise
                    best_prime_idx = i
                    best_r_new = r_new
            
            if best_prime_idx == -1:
                # No prime found? This shouldn't happen unless all primes cover x_fail.
                # If all primes cover x_fail, then counts[x_fail] >= n_primes = 168.
                # But we only enter this block if counts[x_fail] < 2.
                # So this case is impossible.
                break
            
            # Apply the move
            i = best_prime_idx
            p = primes[i]
            r_old = int(residues[i])
            r_new = best_r_new
            
            # Update counts
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
                    
            residues[i] = r_new
            
            # Calculate new score
            new_score = get_score(counts, N_cur)
            
            if new_score > current_score:
                current_score = new_score
                stall_count = 0
                last_improvement_time = time.time()
            else:
                stall_count += 1
                # If we are stuck for too long, we might want to restart or perturb
                if stall_count > max_stall:
                    break
            
            # If we haven't improved in a while, we might be stuck in a local minimum
            # We can break and restart with a different seed
            if time.time() - last_improvement_time > 5.0:
                break
                
        return current_score, residues

    # Main loop: try different initializations and restarts
    # We will use a few different initial strategies
    initial_strategies = [
        lambda: np.array([rng.integers(0, p) for p in primes], dtype=np.int64),
        lambda: np.zeros(n_primes, dtype=np.int64),
        lambda: np.ones(n_primes, dtype=np.int64),
        lambda: np.array([p // 2 for p in primes], dtype=np.int64),
    ]
    
    for strat in initial_strategies:
        if time.time() > deadline - 10.0:
            break
        
        residues = strat()
        score, residues = local_search(residues, 1000)
        
        if score > best_score:
            best_score = score
            best_residues = residues.copy()
            
        # If we reached a high score, we can try to polish it
        if best_score > 5000:
             # Try a few more restarts from the best solution
             for _ in range(5):
                 if time.time() > deadline - 5.0:
                     break
                 residues = best_residues.copy()
                 # Perturb a few residues
                 for _ in range(10):
                     i = int(rng.integers(n_primes))
                     residues[i] = int(rng.integers(0, primes[i]))
                 score, residues = local_search(residues, max(1000, best_score // 2))
                 if score > best_score:
                     best_score = score
                     best_residues = residues.copy()

    # Final Verification
    if best_residues is not None and time.time() < deadline - 1.0:
        N_verify = 100000
        counts_v = build_counts(best_residues, N_verify)
        final_score = get_score(counts_v, N_verify)
        if final_score > best_score:
            best_score = final_score
            # Note: best_residues is already the one that produced this score in local_search
            # But local_search might have stopped early. The final verification is just to confirm.
            # Actually, local_search returns the residues that achieved the score.
            # So we don't need to update best_residues here unless we found a better one.
            # But wait, local_search returns the residues. So best_residues is already correct.

    return best_residues.tolist()
