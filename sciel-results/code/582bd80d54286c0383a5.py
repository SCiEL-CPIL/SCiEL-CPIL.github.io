import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i**0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    primes_arr = np.array(primes, dtype=np.int64)
    n_primes = len(primes_arr)
    
    # We'll track coverage up to a reasonable limit
    max_limit = 5000
    
    def compute_L(residues, primes_arr, max_limit):
        """Compute the largest L such that every integer 1..L has coverage >= 2."""
        counts = np.zeros(max_limit + 1, dtype=np.int32)
        for i, p in enumerate(primes_arr):
            r = residues[i]
            # Generate all x in [1, max_limit] where x % p == r
            if r == 0:
                start = p
            else:
                start = r
            if start > max_limit:
                continue
            # Use vectorized approach
            xs = np.arange(start, max_limit + 1, p)
            counts[xs] += 1
        # Find largest L where all 1..L have count >= 2
        if max_limit > 0:
            valid = counts[1:] >= 2
            if not np.any(valid):
                return 0
            # Find first False
            idx = np.argmax(~valid)
            if valid[idx]:
                # All are valid
                return max_limit
            else:
                return idx
        return 0

    def compute_L_fast(residues, primes_arr, max_limit):
        """Faster version using numpy broadcasting for small primes."""
        counts = np.zeros(max_limit + 1, dtype=np.int32)
        for i, p in enumerate(primes_arr):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > max_limit:
                continue
            xs = np.arange(start, max_limit + 1, p)
            counts[xs] += 1
        # Find largest L where all 1..L have count >= 2
        valid = counts[1:max_limit+1] >= 2
        if not np.any(valid):
            return 0
        # Find first False
        idx = np.argmax(~valid)
        if valid[idx]:
            return max_limit
        else:
            return idx

    def compute_L_with_counts(residues, primes_arr, max_limit):
        """Compute L and return (L, counts) for reuse."""
        counts = np.zeros(max_limit + 1, dtype=np.int32)
        for i, p in enumerate(primes_arr):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > max_limit:
                continue
            xs = np.arange(start, max_limit + 1, p)
            counts[xs] += 1
        valid = counts[1:max_limit+1] >= 2
        if not np.any(valid):
            return 0, counts
        idx = np.argmax(~valid)
        if valid[idx]:
            return max_limit, counts
        else:
            return idx, counts

    def get_bottleneck_from_counts(counts, max_limit):
        """Find the first position where coverage < 2."""
        valid = counts[1:max_limit+1] >= 2
        if not np.any(valid):
            return 1
        idx = np.argmax(~valid)
        if valid[idx]:
            return max_limit + 1
        else:
            return idx + 1

    # Simulated annealing
    best_L = 0
    best_residues = None
    
    # Multiple restarts
    n_restarts = 5
    time_per_restart = (deadline - (time.time() - start_time)) / n_restarts
    
    for restart in range(n_restarts):
        # Initialize residues randomly
        residues = np.zeros(n_primes, dtype=np.int64)
        for i in range(n_primes):
            residues[i] = random.randint(0, primes_arr[i] - 1)
        
        L, counts = compute_L_with_counts(residues, primes_arr, max_limit)
        current_L = L
        current_residues = residues.copy()
        
        if current_L > best_L:
            best_L = current_L
            best_residues = current_residues.copy()
        
        # Simulated annealing parameters
        T_start = 50.0
        T_end = 0.1
        n_iterations = 20000  # Will be limited by time
        
        for it in range(n_iterations):
            if time.time() > deadline - 2.0:
                break
            
            # Temperature schedule (geometric cooling)
            progress = it / n_iterations
            T = T_start * (T_end / T_start) ** progress
            
            # Pick a random prime to change
            idx = random.randint(0, n_primes - 1)
            p = primes_arr[idx]
            r_old = residues[idx]
            r_new = random.randint(0, p - 1)
            
            if r_old == r_new:
                continue
            
            # Compute new coverage efficiently
            # Subtract old contributions
            if r_old == 0:
                start_old = p
            else:
                start_old = r_old
            if start_old <= max_limit:
                xs_old = np.arange(start_old, max_limit + 1, p)
                counts[xs_old] -= 1
            
            # Add new contributions
            if r_new == 0:
                start_new = p
            else:
                start_new = r_new
            if start_new <= max_limit:
                xs_new = np.arange(start_new, max_limit + 1, p)
                counts[xs_new] += 1
            
            # Compute new L
            valid = counts[1:max_limit+1] >= 2
            if not np.any(valid):
                new_L = 0
            else:
                idx2 = np.argmax(~valid)
                if valid[idx2]:
                    new_L = max_limit
                else:
                    new_L = idx2
            
            # Acceptance decision
            delta = new_L - current_L
            if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                # Accept
                current_L = new_L
                residues[idx] = r_new
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # Reject: revert
                if r_old == 0:
                    start_old = p
                else:
                    start_old = r_old
                if start_old <= max_limit:
                    xs_old = np.arange(start_old, max_limit + 1, p)
                    counts[xs_old] += 1
                
                if r_new == 0:
                    start_new = p
                else:
                    start_new = r_new
                if start_new <= max_limit:
                    xs_new = np.arange(start_new, max_limit + 1, p)
                    counts[xs_new] -= 1
        
        # After each restart, if we've reached max_limit, we could try to extend
        # but for now, stick with max_limit

    # Final answer
    if best_residues is not None:
        return [int(r) for r in best_residues]
    else:
        # Fallback: seed heuristic
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
