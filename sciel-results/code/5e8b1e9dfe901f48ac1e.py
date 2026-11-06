import numpy as np

def construct():
    # Generate primes <= 1000
    primes = []
    for i in range(2, 1001):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    n_primes = len(primes)
    LIMIT = 5000
    WINDOW = 200

    counts = np.zeros(LIMIT + 1, dtype=np.int32)
    residues = [0] * n_primes
    current_L = 0

    for i, p in enumerate(primes):
        best_r = 0
        best_new_L = -1
        best_conversions = -1
        
        for r in range(p):
            # Create indices affected by r
            start = r if r > 0 else p
            if start <= LIMIT:
                indices = np.arange(start, LIMIT + 1, p)
            else:
                indices = np.array([], dtype=np.int32)
            
            # Copy counts and update
            new_counts = counts.copy()
            if len(indices) > 0:
                new_counts[indices] += 1
            
            # Calculate new_L
            if current_L < LIMIT:
                rest = new_counts[current_L + 1:]
                # Find first index where count < 2
                mask = rest < 2
                if np.any(mask):
                    first_fail = np.argmax(mask)
                    new_L = current_L + first_fail
                else:
                    new_L = LIMIT
            else:
                new_L = LIMIT
            
            # Calculate conversions in window [new_L+1, new_L+WINDOW]
            # Conversions: positions that were 1 (became 2) and are covered by r
            conversions = 0
            if new_L < LIMIT:
                win_start = new_L + 1
                win_end = min(LIMIT, new_L + WINDOW)
                if win_start <= win_end:
                    win_indices = np.arange(win_start, win_end + 1)
                    win_counts = new_counts[win_indices]
                    # Check which of these were affected by r and are now 2
                    mod_check = win_indices % p == r
                    conv_mask = (win_counts == 2) & mod_check
                    conversions = np.sum(conv_mask)
            
            if new_L > best_new_L or (new_L == best_new_L and conversions > best_conversions):
                best_new_L = new_L
                best_conversions = conversions
                best_r = r
        
        residues[i] = best_r
        start = best_r if best_r > 0 else p
        if start <= LIMIT:
            indices = np.arange(start, LIMIT + 1, p)
            counts[indices] += 1
        
        # Update current_L
        if current_L < LIMIT:
            rest = counts[current_L + 1:]
            mask = rest < 2
            if np.any(mask):
                first_fail = np.argmax(mask)
                current_L = current_L + first_fail
            else:
                current_L = LIMIT

    return residues
