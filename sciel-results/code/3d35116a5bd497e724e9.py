import numpy as np

def construct():
    # Generate primes up to 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    G = limit
    
    # Map prime value to index using array for O(1) vectorized access
    prime_idx_arr = np.full(limit + 1, -1, dtype=np.int16)
    for i, p in enumerate(primes):
        prime_idx_arr[p] = i
        
    # M[i, g] = max length of valid sequence ending at primes[i] with last gap <= g
    # Initialize with 1 (sequence of length 1 is always valid)
    M = np.ones((N, G + 1), dtype=np.int16)
    
    for j in range(N):
        pj = primes[j]
        if pj <= 3:
            continue
            
        max_g = pj - 2
        # g ranges from 1 to max_g
        g_vals = np.arange(1, max_g + 1)
        prev_vals = pj - g_vals
        
        # Check which prev_vals are prime
        is_prev_prime = is_prime[prev_vals]
        valid_mask = is_prev_prime
        
        if not np.any(valid_mask):
            continue
            
        valid_g = g_vals[valid_mask]
        valid_prev = prev_vals[valid_mask]
        valid_i = prime_idx_arr[valid_prev]
        
        # Get candidate lengths: 1 + M[valid_i, valid_g]
        cand_lengths = 1 + M[valid_i, valid_g]
        
        # Initialize raw row for j
        raw_M_j = np.zeros(G + 1, dtype=np.int16)
        raw_M_j[valid_g] = cand_lengths
        
        # Prefix max
        M[j, :] = np.maximum.accumulate(raw_M_j)
        # Ensure min 1 (though accumulate of zeros is 0, we need 1)
        # Wait, M is initialized to 1. But we overwrite M[j, :].
        # If no valid gaps, M[j, :] remains 1 from init? No, we skip with continue.
        # If valid gaps exist, raw_M_j has 0s and cand_lengths.
        # accumulate will propagate max.
        # If all cand_lengths < 1? No, min is 2.
        # But for g < min(valid_g), raw_M_j is 0.
        # So M[j, g] will be 0 for small g?
        # We need M[j, g] >= 1.
        M[j, :] = np.maximum(M[j, :], 1)
        
    # Find the maximum length
    max_len = np.max(M)
    
    # Reconstruct the sequence
    flat_idx = np.argmax(M)
    j_end = flat_idx // (G + 1)
    g_end = flat_idx % (G + 1)
    
    # Backtrack
    seq = []
    j = j_end
    g = g_end
    while True:
        seq.append(primes[j])
        if M[j, g] == 1:
            break
        
        # Find effective last gap
        while g > 0 and M[j, g] == M[j, g - 1]:
            g -= 1
            
        if g == 0:
            break
            
        prev_prime = primes[j] - g
        if prev_prime < 2 or not is_prime[prev_prime]:
            break
            
        i = prime_idx_arr[prev_prime]
        
        # Find effective previous gap
        g_prev = g
        while g_prev > 0 and M[i, g_prev - 1] == M[i, g]:
            g_prev -= 1
            
        j = i
        g = g_prev
        
    seq.reverse()
    return seq
