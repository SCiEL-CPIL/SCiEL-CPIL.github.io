import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)
    M = 100000  # search window for the score
    
    # Precompute weights for the SA objective
    # We want to maximize sum(i * I(count_i >= 2))
    # This strongly favors covering early integers, but allows some flexibility
    weights = np.arange(1, M + 1, dtype=np.int64)

    def calc_L(counts):
        """Largest L s.t. every integer 1..L has count >= 2."""
        mask = counts[1:M + 1] < 2
        if not np.any(mask):
            return M
        # np.argmax returns the index of the first True in the mask
        # Since mask is aligned with counts[1:], index 0 corresponds to integer 1
        # If mask[0] is True, L=0. If mask[0] is False and mask[1] is True, L=1.
        # Wait, if mask[0] is True, argmax is 0. L should be 0.
        # If mask[0] is False, mask[1] is True, argmax is 1. L should be 1.
        # So L = argmax(mask).
        return int(np.argmax(mask))

    def calc_S(counts):
        """Weighted coverage score."""
        # counts[1:M+1] corresponds to integers 1..M
        covered = (counts[1:M + 1] >= 2).astype(np.int64)
        return np.dot(weights, covered)

    def make_counts(residues):
        counts = np.zeros(M + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
        return counts

    def greedy_polish(residues, counts, max_iter=500):
        """Target the bottleneck integer and fix it via a 1D repair over primes."""
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            L = calc_L(counts)
            if L >= M:
                break
            x = L + 1
            candidates = []
            for i, p in enumerate(primes):
                r = residues[i]
                if (x % p) != (r % p):
                    candidates.append(i)
            
            if not candidates:
                break
            
            best_i = None
            best_gain = -1
            best_r = None
            best_r_old = None
            best_p = None
            
            # Try to find a move that maximizes L
            for i in candidates:
                p = primes[i]
                r_new = x % p
                r_old = residues[i]
                if r_new == r_old:
                    continue
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                new_L = calc_L(counts)
                gain = new_L - L
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_r = r_new
                    best_r_old = r_old
                    best_p = p
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
            
            if best_i is not None and best_gain > 0:
                counts[best_r_old::best_p] -= 1
                counts[best_r::best_p] += 1
                residues[best_i] = best_r
            else:
                # No single move improves L, break or try random?
                # In polish phase, we stop if stuck.
                break
        return residues, counts

    # --- Main SA Loop ---
    random.seed(42)
    
    # Initial state
    residues = np.array([random.randrange(p) for p in primes], dtype=np.int64)
    counts = make_counts(residues)
    current_S = calc_S(counts)
    current_L = calc_L(counts)
    
    best_S = current_S
    best_L = current_L
    best_residues = residues.copy()
    best_counts = counts.copy()
    
    T = 5000.0  # Initial temperature
    T_min = 1.0
    cooling = 0.999
    
    # We will run SA for most of the time, then polish
    sa_deadline = deadline - 10.0
    
    steps = 0
    while time.time() < sa_deadline:
        # Pick a random prime to modify
        i = random.randrange(n_primes)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randrange(p)
        
        if r_old == r_new:
            continue
        
        # Apply move
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        residues[i] = r_new
        
        # Evaluate
        new_S = calc_S(counts)
        delta_S = new_S - current_S
        
        # Accept/Reject
        if delta_S >= 0 or random.random() < np.exp(delta_S / T):
            current_S = new_S
            # Update best if S is higher
            if current_S > best_S:
                best_S = current_S
                best_residues = residues.copy()
                best_counts = counts.copy()
                # Occasionally check L to see if we improved the actual metric
                # Doing this every step is expensive, so maybe every 100 steps?
                # But we want to track best_L accurately.
                # Let's check L if S improves.
                new_L = calc_L(counts)
                if new_L > best_L:
                    best_L = new_L
        else:
            # Revert
            counts[r_new::p] -= 1
            counts[r_old::p] += 1
            residues[i] = r_old
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher T? Or just continue?
            # If T is too low, we might get stuck.
            # Let's do a "reheat" if stuck for a long time?
            # For now, simple cooling.
        
        steps += 1

    # --- Polish Phase ---
    # Use the best state found in SA
    residues = best_residues.copy()
    counts = best_counts.copy()
    
    # Run greedy polish to push L as far as possible
    residues, counts = greedy_polish(residues, counts, max_iter=1000)
    
    final_L = calc_L(counts)
    
    # If polish improved L, update best
    if final_L > best_L:
        best_L = final_L
        best_residues = residues.copy()

    return [int(r) for r in best_residues]
