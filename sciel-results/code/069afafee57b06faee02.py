"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(12, m) is the
    least positive L such that (m, m+L] contains 12 distinct integers b_1..b_12
    with i | b_i.
    """
    n = 12
    max_m = 1000000
    period = 27720 # lcm(1..12)
    
    # Precompute LCMs and popcounts for all subsets of {1..12}
    # Mask i represents subset of indices. Bit j corresponds to index j+1.
    num_masks = 1 << n
    lcm_arr = np.zeros(num_masks, dtype=np.int64)
    popcount_arr = np.zeros(num_masks, dtype=np.int64)
    
    # lcm_arr[0] = 1 (empty set lcm is 1, but we don't use it in union sum usually, 
    # actually inclusion-exclusion sum is over non-empty T. 
    # F[S] = sum_{T subset S, T!=0} (-1)^{|T|+1} cnt[lcm(T)]
    # My zeta transform logic:
    # A[T] = (-1)^{|T|+1} cnt[lcm(T)] for T!=0, A[0]=0.
    # F[S] = sum_{T subset S} A[T].
    # This matches.
    
    for mask in range(num_masks):
        # Calculate popcount
        pc = 0
        l = 1
        temp = mask
        while temp:
            if temp & 1:
                pc += 1
                # Calculate lcm of indices present
                # Index is j+1 where bit j is set
                # We need to find which bits are set
                pass
            temp >>= 1
        # Simpler way to calculate LCM
        l = 1
        for j in range(n):
            if mask & (1 << j):
                l = l * (j + 1) // math.gcd(l, j + 1)
        lcm_arr[mask] = l
        popcount_arr[mask] = pc
        
    # Precompute signs for inclusion-exclusion
    # A[T] = (-1)^{|T|+1} cnt[lcm(T)]
    # We will apply this sign to cnt before zeta transform.
    signs = np.ones(num_masks, dtype=np.int64)
    signs[0] = 0 # Empty set contributes 0
    for mask in range(1, num_masks):
        if popcount_arr[mask] % 2 == 0:
            signs[mask] = -1
        else:
            signs[mask] = 1
            
    # Multiply signs into a coefficient array? 
    # Actually, we compute cnt, then multiply by signs, then zeta transform.
    
    def can_match(m, L):
        """Check if there exist distinct b_1..b_12 in (m, m+L] with i|b_i using Hall's theorem."""
        if L <= 0:
            return False
            
        # Calculate counts of multiples of lcm(T) in (m, m+L]
        # cnt[T] = floor((m+L)/lcm[T]) - floor(m/lcm[T])
        # Handle lcm[0]=1, but signs[0]=0 so it doesn't matter.
        cnt = (m + L) // lcm_arr - m // lcm_arr
        # Apply inclusion-exclusion signs
        # A[T] = signs[T] * cnt[T]
        A = signs * cnt
        
        # Zeta transform: F[S] = sum_{T subset S} A[T]
        F = A.copy()
        # Iterate over bits
        for i in range(n):
            # Reshape to separate the bit i
            # Shape: (2^i, 2, 2^(n-i-1))
            # We want to add the part where bit i is 0 to the part where bit i is 1.
            # In the reshaped array, axis 1 has size 2. Index 0 is bit 0, Index 1 is bit 1.
            F = F.reshape((1 << i), 2, (1 << (n - i - 1)))
            F[:, 1, :] += F[:, 0, :]
            F = F.reshape(num_masks)
            
        # Hall's condition: F[S] >= |S| for all S
        # F[S] is the size of the union of multiples of elements in S.
        # |S| is popcount_arr[S].
        return np.all(F >= popcount_arr)
        
    def f_value(m):
        """Compute f(12, m): least positive L such that matching exists."""
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        if can_match(m, lo):
            return lo
        return 145 # Should not happen
        
    # Search strategy:
    # 1. Coarse search over one period [0, period) with step 10
    # 2. Fine search around the best candidate with step 1
    
    best_m = 0
    best_f = 0
    
    # Coarse search
    for m in range(0, period, 10):
        f = f_value(m)
        if f > best_f:
            best_f = f
            best_m = m
            
    # Fine search around best_m
    # Expand the window if needed, but start with +/- 20
    search_range = 50
    for m in range(max(0, best_m - search_range), min(period, best_m + search_range + 1)):
        f = f_value(m)
        if f > best_f:
            best_f = f
            best_m = m
            
    # Ensure we return an int in [0, 1000000]
    # The best_m is in [0, period). 
    # Since the function is periodic, any m + k*period is also valid.
    # We return best_m % period.
    return int(best_m % period)

# EVOLVE-BLOCK-END
