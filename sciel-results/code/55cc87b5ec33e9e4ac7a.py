import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110  # 10s margin from 120s limit
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    def elapsed():
        return time.time() - start_time

    def verify_solution(M, k_list):
        """Verify k*sigma(k)=M for all k in k_list using exact factorization."""
        if not k_list:
            return False
        for k in k_list:
            if k > math.isqrt(M):
                return False
            # Exact sigma
            temp = k
            s = 1
            p = 2
            while p * p <= temp:
                if temp % p == 0:
                    power = 0
                    while temp % p == 0:
                        temp //= p
                        power += 1
                    s *= (p**(power+1) - 1) // (p - 1)
                p += 1 if p == 2 else 2
            if temp > 1:
                s *= (temp*temp - 1) // (temp - 1)
            if k * s != M:
                return False
        return True

    # Try increasing N
    # Start with 20M, try 50M if time permits
    N_attempts = [20_000_000, 50_000_000]
    
    for N in N_attempts:
        if elapsed() > deadline - 30:
            break
            
        # Check memory
        # 50M * 8 bytes = 400MB. OK.
        
        try:
            sigma = np.zeros(N + 1, dtype=np.int64)
            # Sieve
            # Optimization: Iterate d from 1 to N
            # This is the bottleneck.
            # We can skip even numbers after 2? No, sigma needs all divisors.
            # But we can optimize the inner loop?
            # NumPy slice assignment is fast.
            
            for d in range(1, N + 1):
                if d % 1_000_000 == 0:
                    if elapsed() > deadline - 20:
                        raise TimeoutError("Sieve too slow")
                sigma[d::d] += d
                
            # Compute M = k * sigma[k]
            k_arr = np.arange(1, N + 1, dtype=np.int64)
            M_arr = k_arr * sigma[1:]
            
            # Filter M <= 10^18
            mask = M_arr <= 10**18
            M_valid = M_arr[mask]
            k_valid = k_arr[mask]
            
            if len(M_valid) == 0:
                continue
                
            # Sort by M
            # argsort is O(N log N)
            order = np.argsort(M_valid)
            M_sorted = M_valid[order]
            k_sorted = k_valid[order]
            
            # Find max group
            # Iterate through sorted M to find runs
            max_count = 0
            max_M_val = 0
            max_k_list = []
            
            # Vectorized group finding?
            # changes = np.where(M_sorted[1:] != M_sorted[:-1])[0]
            # This creates a large array.
            # Better to loop? 50M loop in Python is too slow.
            # Use numpy to find starts and ends.
            
            if len(M_sorted) > 0:
                # Find indices where M changes
                # Note: M_sorted[0] is the first element
                # diff = np.diff(M_sorted)
                # changes = np.where(diff != 0)[0] + 1
                # starts = np.concatenate([[0], changes])
                # ends = np.concatenate([changes, [len(M_sorted)]])
                
                # This might use a lot of memory if there are many changes (which there are, ~N of them).
                # M_sorted has ~50M elements.
                # diff has 50M elements.
                # changes has ~50M elements (since most M are unique).
                # This is heavy.
                
                # Alternative:
                # We only care about groups with size > best_count.
                # Most groups have size 1.
                # We can use a dictionary on the *valid* M values?
                # But 50M items in a dict is slow and memory heavy.
                
                # Let's stick to the sorting approach but be careful.
                # Actually, if most M are unique, the "changes" array is huge.
                # Is there a better way?
                
                # How about:
                # We know we want the MAX count.
                # We can sample? No, we need exact.
                
                # Let's try the np.unique approach?
                # unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
                # max_idx = np.argmax(counts)
                # max_count = counts[max_idx]
                # max_M_val = unique_M[max_idx]
                # k_list = k_valid[inverse == max_idx]
                
                # np.unique on 50M int64s:
                # Sorting is the main cost.
                # This is likely faster than manual scanning in Python.
                
                unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
                
                if len(counts) > 0:
                    max_idx = np.argmax(counts)
                    current_max_count = int(counts[max_idx])
                    
                    if current_max_count > best_count:
                        best_count = current_max_count
                        best_M = int(unique_M[max_idx])
                        # Get the k's
                        # inverse == max_idx gives boolean mask
                        max_k_list = k_valid[inverse == max_idx].tolist()
                        
        except Exception as e:
            # If something goes wrong (memory, time), break
            break
            
        if elapsed() > deadline - 10:
            break
            
    # Final Verification
    # The numpy sieve is exact for integers < 2^63.
    # M < 10^18 < 2^63.
    # sigma(k) for k < 5*10^7 is small.
    # So the values are exact.
    # However, we must ensure k <= isqrt(M).
    # As established, k < sqrt(M) for k>1.
    # We should verify the specific list returned.
    
    if best_count > 3:
        # Verify the new best
        if verify_solution(best_M, best_k_list):
            pass
        else:
            # Fallback if verification fails (should not happen)
            best_M = 333312
            best_k_list = [336, 372, 434]
            best_count = 3
    else:
        # If we didn't find anything better, verify the hardcoded one just in case
        if not verify_solution(333312, [336, 372, 434]):
             # This should never happen, but fallback to k=1, M=1
             best_M = 1
             best_k_list = [1]
             best_count = 1

    return {'M': int(best_M), 'k': list(best_k_list)}

if __name__ == "__main__":
    construct()
