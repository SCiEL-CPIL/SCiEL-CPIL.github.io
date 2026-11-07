import time
import random

# EVOLVE-BLOCK-START

def _is_valid_mask(mask):
    """
    Check if the set represented by the bitmask violates the condition.
    Bit i represents integer i+1 (1..128).
    Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
    """
    # We check for any a, b, c distinct in S
    # a+b in S, a+c in S, b+c in S
    
    # Optimization: 
    # If a+b in S, then a,b < a+b <= 128.
    # So a,b must be in 1..127.
    # Also a != b.
    
    # Iterate over potential sums s in S.
    # s = a+b.
    # If s in S, check if there exist a,b in S, a!=b, a<b, a+b=s
    # AND if there exists c in S, c!=a, c!=b, such that a+c in S and b+c in S.
    
    # This is still potentially slow.
    # Let's stick to the O(N^3) logic but with bit tricks for the inner check.
    
    # Actually, the previous _is_valid_mask was O(N^2) with bit tricks.
    # Let's reuse that logic which was fast enough.
    
    for i in range(128):
        if not (mask >> i) & 1:
            continue
        a = i + 1
        for j in range(i + 1, 128):
            if not (mask >> j) & 1:
                continue
            b = j + 1
            
            # Check if a+b is in S
            # a+b = i+j+2
            # Bit index i+j+1
            if i + j + 1 >= 128:
                continue
            if not (mask >> (i + j + 1)) & 1:
                continue
            
            # If a+b in S, check if there exists c > b such that a+c in S and b+c in S
            # c is value k+1, k > j
            # a+c = i+k+2 => Bit i+k+1
            # b+c = j+k+2 => Bit j+k+1
            
            # candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            # We need bit k set in mask, mask>>(i+1), mask>>(j+1)
            # And k > j
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            
            # Mask out bits <= j
            # ~((1 << (j + 1)) - 1)
            
            if candidates & ~((1 << (j + 1)) - 1):
                return False
    return True

def _greedy_construct_mask(deadline):
    """
    Greedy construction using bitmasks.
    Start with the known good set of size 81.
    """
    # Start with 65..128
    best_mask = (1 << 128) - 1
    best_mask &= ~((1 << 64) - 1)
    
    # Add 1, 3, ..., 31, 64
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        bit = 1 << (v - 1)
        trial_mask = best_mask | bit
        if _is_valid_mask(trial_mask):
            best_mask = trial_mask
    
    # Try to add elements from 1..64 not yet added
    for v in range(1, 65):
        if time.time() > deadline:
            break
        bit = 1 << (v - 1)
        if best_mask & bit:
            continue
        trial_mask = best_mask | bit
        if _is_valid_mask(trial_mask):
            best_mask = trial_mask
            
    # Local search: try to swap elements to increase size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and adding others
        for rem in range(1, 129):
            if time.time() > deadline:
                break
            rem_bit = 1 << (rem - 1)
            if not (best_mask & rem_bit):
                continue
            
            trial_base = best_mask & ~rem_bit
            base_size = trial_base.bit_count()
            
            # Try to add elements from 1..128 not in trial_base
            for add in range(1, 129):
                if add == rem:
                    continue
                add_bit = 1 << (add - 1)
                if trial_base & add_bit:
                    continue
                if time.time() > deadline:
                    break
                trial = trial_base | add_bit
                if _is_valid_mask(trial):
                    trial_base = trial
            # If we added more than we removed (1), size increased
            if trial_base.bit_count() > best_mask.bit_count():
                best_mask = trial_base
                improved = True
                
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Initialize with the strong baseline
    best_mask = (1 << 128) - 1
    best_mask &= ~((1 << 64) - 1)
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        bit = 1 << (v - 1)
        trial_mask = best_mask | bit
        if _is_valid_mask(trial_mask):
            best_mask = trial_mask
            
    best_size = best_mask.bit_count()
    
    # Run greedy
    if time.time() < deadline - 10:
        greedy_result = _greedy_construct_mask(deadline - 10)
        if greedy_result.bit_count() > best_size:
            best_mask = greedy_result
            best_size = best_mask.bit_count()
    
    # Final check
    if not _is_valid_mask(best_mask):
        # Fallback
        best_mask = (1 << 128) - 1
        best_mask &= ~((1 << 64) - 1)
        for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
            bit = 1 << (v - 1)
            trial_mask = best_mask | bit
            if _is_valid_mask(trial_mask):
                best_mask = trial_mask

    result = []
    for i in range(128):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
