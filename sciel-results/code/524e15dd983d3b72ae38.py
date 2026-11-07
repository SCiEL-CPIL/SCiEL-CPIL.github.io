import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try a range large enough to find collisions
    # Hint suggests 5M finds 5 solutions. Let's try 10M.
    N = 10_000_000
    
    # Sieve for sigma(n)
    # sigma[i] = sum of divisors of i
    sigma = np.ones(N + 1, dtype=np.uint64)
    
    # Standard sieve for sigma
    # For each i, add i to all multiples of i
    # This is O(N log N)
    for i in range(2, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        # Fallback to a smaller set or known solution if sieve is too slow
        return {'M': 333312, 'k': [336, 372, 434]}

    # Compute M_k = k * sigma(k)
    # k is 1..N
    k_vals = np.arange(1, N + 1, dtype=np.uint64)
    M_vals = k_vals * sigma[1:]
    
    # Filter M_vals <= 10^18
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    if len(M_vals) == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Find the M with the most occurrences
    # Using np.unique is efficient for this
    unique_M, counts, inverse = np.unique(M_vals, return_counts=True, return_inverse=True)
    
    # Find the index of the max count
    max_idx = np.argmax(counts)
    best_M = int(unique_M[max_idx])
    best_count = int(counts[max_idx])
    
    # If we didn't find at least 3 solutions (unlikely), fallback
    if best_count < 3:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Get the k values corresponding to best_M
    # inverse maps original indices to unique_M indices
    # We need original indices where inverse == max_idx
    best_k_indices = np.where(inverse == max_idx)[0]
    best_k = k_vals[best_k_indices]
    
    # Convert to Python list of ints
    # Ensure they are distinct (they are, by construction of k_vals)
    k_list = [int(k) for k in best_k]
    
    # Verify constraints: k_i <= isqrt(M)
    # The problem states k_i <= isqrt(M)
    # My sieve generates k up to N.
    # We must ensure k_i * sigma(k_i) = M implies k_i is a valid solution.
    # But the constraint k_i <= isqrt(M) is part of the contract.
    # Does k*sigma(k)=M imply k <= sqrt(M)?
    # sigma(k) >= k+1 (for k>1). So k*sigma(k) > k^2.
    # Thus M > k^2, so k < sqrt(M).
    # So any k satisfying k*sigma(k)=M automatically satisfies k < sqrt(M).
    # So we don't need to filter k by sqrt(M), they are all valid.
    
    return {'M': best_M, 'k': k_list}
