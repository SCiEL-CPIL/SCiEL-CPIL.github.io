import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (150s kill, 5s buffer)

    # Sieve primes up to 10^6 for factorization
    limit = 1000000
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, end):
        """Compute phi(n) for n in [start, end] using segmented approach."""
        size = end - start + 1
        # Use int64 to avoid overflow
        phi = np.arange(start, end + 1, dtype=np.int64)
        rem = phi.copy()
        
        for p in primes:
            if p * p > end:
                break
            # Find indices where rem is divisible by p
            # first multiple of p >= start
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            
            # Indices in the array corresponding to multiples of p
            start_idx = first - start
            # step is p
            idxs = np.arange(start_idx, size, p)
            
            if len(idxs) == 0:
                continue
            
            # Check which of these are actually divisible by p in rem
            # We use a mask to identify indices where rem[idx] is divisible by p
            # This is necessary because rem[idx] might have been divided by smaller primes
            # but still be divisible by p? No, if p is prime and p > smaller primes,
            # and rem[idx] was divisible by p initially, it remains divisible by p
            # unless we divided it out. But we only divide by p in this step.
            # Wait, if n = 12, p=2. rem=12. 12%2==0. phi -= phi//2. rem=6. 6%2==0. rem=3.
            # Next p=3. idxs for 12? 12 is multiple of 3. rem=3. 3%3==0.
            # So the mask is correct.
            
            # Optimization: Use np.remainder and np.floor_divide with out=
            # But we need to update phi too.
            
            # Calculate mask
            # rem[idxs] % p == 0
            # To avoid creating a huge temporary array for rem[idxs],
            # we can do:
            # vals = rem[idxs]
            # mask = vals % p == 0
            
            vals = rem[idxs]
            mask = vals % p == 0
            if not mask.any():
                continue
            
            sel = idxs[mask]
            
            # Update phi: phi -= phi // p
            # phi[sel] -= phi[sel] // p
            # This is equivalent to phi[sel] = phi[sel] * (p-1) // p
            # But phi[sel] is divisible by p?
            # phi(n) = n * prod(1-1/p).
            # Initially phi = n.
            # If n is divisible by p, phi is divisible by p?
            # Yes, because we subtract phi//p.
            # So phi[sel] is divisible by p.
            
            phi[sel] -= phi[sel] // p
            
            # Update rem: divide by p as many times as possible
            # rem[sel] //= p
            # while rem[sel] % p == 0: rem[sel] //= p
            
            # We can do this in a loop
            rem[sel] //= p
            while True:
                still = rem[sel] % p == 0
                if not still.any():
                    break
                sel = sel[still]
                rem[sel] //= p
        
        # Handle remaining prime factors > 1
        mask = rem > 1
        if mask.any():
            phi[mask] -= phi[mask] // rem[mask]
        
        return phi

    def find_longest_run_sliding(phi_vals):
        """Find longest consecutive run of distinct phi values using sliding window."""
        n = len(phi_vals)
        if n == 0:
            return 1, 0
        best_k = 1
        best_s = 0
        seen = {}
        left = 0
        for right in range(n):
            val = int(phi_vals[right])
            if val in seen and seen[val] >= left:
                left = seen[val] + 1
            seen[val] = right
            cur_k = right - left + 1
            if cur_k > best_k:
                best_k = cur_k
                best_s = left
            if best_k >= 200:
                break
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Phase 1: Scan high range in large segments
    # Start from 10^12 downwards
    seg_size = 1000000  # Reduced to 1M for better cache performance and faster iteration
    current_pos = 10**12 - seg_size + 1
    
    # We will scan downwards
    while current_pos >= 1:
        if time.time() - start_time > deadline - 5:
            break
        
        seg_start = current_pos
        seg_end = min(current_pos + seg_size - 1, 10**12)
        
        try:
            phi_vals = compute_phi_segment(seg_start, seg_end)
            k, s = find_longest_run_sliding(phi_vals)
            actual_s = seg_start + s
            
            if k > best_k:
                best_k = k
                best_s = actual_s
                # print(f"Found k={best_k} at s={best_s}")
                
                if best_k == 200:
                    break
        except Exception:
            pass
            
        current_pos -= seg_size

    # Final validation and clamping
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
