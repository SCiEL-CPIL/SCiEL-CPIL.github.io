import time
import math

def construct():
    start_time = time.time()
    limit = 88  # Leave 2s for overhead, evaluator kills at 90s
    
    N = 12
    period = 27720
    
    def compute_f(m):
        # Compute the least L >= 1 such that there's a perfect matching
        # Using bitmask and MRV backtracking
        
        # Precompute r_i for each i
        # r_i = (-m - 1) % i
        # Valid j for i: j % i == r_i
        
        # We will check L from 1 upwards.
        # Max L is bounded, say 150.
        
        for L in range(1, 151):
            # Build masks for each i
            # mask[i] has bit j set if j is valid for i in [0, L)
            masks = [0] * (N + 1)
            for i in range(1, N + 1):
                ri = (-m - 1) % i
                mask = 0
                for j in range(ri, L, i):
                    mask |= (1 << j)
                masks[i] = mask
            
            # Check if perfect matching exists
            # State: used_mask (bits set for used positions), next indices to assign
            # We use a list of indices 1..N.
            # MRV: at each step, pick i with fewest available bits (mask[i] & ~used_mask)
            
            indices = list(range(1, N + 1))
            
            def backtrack(used_mask, remaining_indices):
                if not remaining_indices:
                    return True
                
                # Find index with minimum remaining values
                best_idx = -1
                best_count = L + 1
                best_avail = 0
                for idx in remaining_indices:
                    avail = masks[idx] & (~used_mask)
                    count = bin(avail).count('1')
                    if count < best_count:
                        best_count = count
                        best_idx = idx
                        best_avail = avail
                    if count == 0:
                        return False # No options for this idx
                
                # Try all options for best_idx
                # Iterate through bits in best_avail
                avail = best_avail
                while avail:
                    # Get lowest set bit
                    lsb = avail & (-avail)
                    avail ^= lsb
                    
                    # Assign this position to best_idx
                    if backtrack(used_mask | lsb, [x for x in remaining_indices if x != best_idx]):
                        return True
                
                return False
            
            # Initial call
            # We can sort initial indices by density to help MRV, but MRV handles it dynamically.
            if backtrack(0, indices):
                return L
        
        return 150

    best_m = 0
    best_f = 0
    
    # Search through one full period
    # We can try to jump to likely candidates? No, just scan.
    # The period is 27720.
    
    for m in range(period):
        if time.time() - start_time > limit:
            break
        
        f_val = compute_f(m)
        
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
