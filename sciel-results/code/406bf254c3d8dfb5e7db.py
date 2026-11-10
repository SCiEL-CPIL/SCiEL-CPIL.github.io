import numpy as np
import time

def construct():
        start = time.time()
        deadline = 120.0 # Safety margin
        
        # We will try increasing limits
        # N=10^6 is instant. N=10^7 is fast. N=5*10^7 might be slow/memory intensive.
        # Let's start with 10^7 and see how much time is left.
        
        best_M = 333312
        best_k = [336, 372, 434]
        best_count = 3
        
        limits = [10**6, 5*10**6, 10**7, 2*10**7, 5*10**7]
        
        for N in limits:
            if time.time() - start > deadline - 30:
                break
            
            try:
                sigma = np.zeros(N + 1, dtype=np.int64)
                # Sieve for sigma
                # For each i, add i to sigma[j] for all multiples j of i
                # This is O(N log N)
                for i in range(1, N + 1):
                    sigma[i::i] += i
                
                # Calculate M = k * sigma(k)
                # Note: k is 1..N
                k_vals = np.arange(1, N + 1, dtype=np.int64)
                M_vals = k_vals * sigma[1:]
                
                # Filter M <= 10^18
                mask = M_vals <= 10**18
                M_valid = M_vals[mask]
                k_valid = k_vals[mask]
                
                # Count frequencies
                # np.unique returns unique values and indices
                unique_M, inverse_indices, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
                
                # Find max count
                max_count = np.max(counts)
                if max_count > best_count:
                    # Get the M with max count
                    # There might be multiple M with the same max count, pick one
                    # We need the list of k's for that M
                    idx = np.argmax(counts)
                    target_M = unique_M[idx]
                    
                    # Get all k's that map to target_M
                    # inverse_indices maps M_valid[i] to unique_M[inverse_indices[i]]
                    # So we need k_valid where inverse_indices == idx
                    # Wait, inverse_indices corresponds to M_valid.
                    # k_valid[i] is the k for M_valid[i].
                    # So k_solutions = k_valid[inverse_indices == idx]
                    
                    k_solutions = k_valid[inverse_indices == idx]
                    
                    # Verify and convert to list of ints
                    # The task requires distinct k_i. The sieve produces distinct k.
                    # We should verify a few to be safe? Or trust the sieve?
                    # The sieve is exact for int64 if no overflow.
                    # Max M is 10^18. Max sigma is ~10^9 (for k=10^7, sigma ~ k log log k ~ 10^7 * 3 = 3*10^7).
                    # k*sigma ~ 3*10^14. No overflow in int64.
                    
                    best_M = int(target_M)
                    best_k = [int(x) for x in k_solutions]
                    best_count = int(max_count)
                    
            except MemoryError:
                break
            except Exception:
                break
        
        return {'M': best_M, 'k': best_k}
