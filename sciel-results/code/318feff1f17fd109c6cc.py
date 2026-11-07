import math
import time
import numpy as np

def construct():
        start_time = time.time()
        deadline = start_time + 110.0
        
        # Estimate N based on time?
        # Let's try N = 2 * 10**6 first.
        # If it takes too long, we fail.
        # Let's try to be smarter.
        
        # We know 333312 has 3 solutions.
        # If we find > 3, we win.
        
        # Let's try N = 5 * 10**6.
        # If time > 100s, return best found so far?
        # The sieve is atomic in the sense that we need the full array.
        # We can interrupt the loop?
        
        N = 5 * 10**6
        try:
            sigma = np.ones(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                if time.time() > deadline - 5.0:
                    break
                sigma[i::i] += i
            
            # Compute M = k * sigma(k)
            # k = 1..N
            ks = np.arange(1, N + 1, dtype=np.int64)
            Ms = ks * sigma[1:]
            
            # Find most frequent M
            # We need to group by M.
            # Using pandas or numpy unique?
            # unique, counts = np.unique(Ms, return_counts=True)
            # This is O(N log N).
            
            # But we also need to retrieve the k's for the best M.
            # So we should sort by M, then group.
            
            # Sorting 5 million elements takes ~1-2 seconds.
            
            if time.time() > deadline - 10.0:
                # Fallback to known best
                return {'M': 333312, 'k': [336, 372, 434]}
                
            # Sort indices by Ms
            # To save memory/time, we can just sort Ms and track indices?
            # Or use a dict? Dict with 5M entries is slow in Python.
            
            # Numpy approach:
            # sorted_indices = np.argsort(Ms)
            # sorted_Ms = Ms[sorted_indices]
            # Then scan for runs of equal values.
            
            sorted_indices = np.argsort(Ms)
            sorted_Ms = Ms[sorted_indices]
            
            # Find the longest run of identical values in sorted_Ms
            # And check if M <= 10**18
            
            best_count = 0
            best_M = 333312
            best_k_list = [336, 372, 434]
            
            # Scan
            # This loop is O(N).
            current_M = None
            current_count = 0
            current_ks = []
            
            # We can't store all ks for every M in a list of lists easily in the loop without overhead.
            # We just need the max count.
            
            # However, we need to return the k's.
            # So if we find a count > best_count, we need to extract those k's.
            
            # Optimization:
            # Find the max count first?
            # Or just update best when we find a longer run.
            
            # To extract k's, we need the slice of sorted_indices.
            
            # Let's just do a pass.
            # Note: np.unique is faster but doesn't give the original indices easily for the "best" one without extra work.
            
            # Let's use np.unique with return_inverse?
            # unique_Ms, inverse = np.unique(Ms, return_inverse=True)
            # Then bincount?
            # counts = np.bincount(inverse)
            # max_idx = np.argmax(counts)
            # best_M = unique_Ms[max_idx]
            # best_k_list = ks[inverse == max_idx]
            
            # This is very efficient.
            
            if time.time() > deadline - 15.0:
                 return {'M': 333312, 'k': [336, 372, 434]}

            unique_Ms, inverse = np.unique(Ms, return_inverse=True)
            counts = np.bincount(inverse)
            
            # Filter M <= 10**18
            # All Ms should be <= N * sigma(N) ~ 10^7 * 10^8 = 10^15, so all are valid.
            
            max_count = np.max(counts)
            
            if max_count > len(best_k_list):
                # Find the M with this count
                # There might be multiple M with the same max count.
                # Any is fine.
                best_idx = np.argmax(counts)
                best_M = int(unique_Ms[best_idx])
                best_k_list = [int(k) for k in ks[inverse == best_idx]]
                
            return {'M': int(best_M), 'k': best_k_list}
            
        except Exception:
            return {'M': 333312, 'k': [336, 372, 434]}
