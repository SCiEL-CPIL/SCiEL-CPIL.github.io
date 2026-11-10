"""Simulated Annealing approach for 3-AP maximization without 4-APs."""
import time
import numpy as np

N = 256
DEADLINE_FRACTION = 0.85

def _precompute():
    """Precompute 4-AP conflicts and 3-AP structures."""
    # 1. Precompute 4-AP conflicts: conflicts[x] is a set of y such that {x, y} is part of a 4-AP
    # Actually, we need to know if adding x creates a 4-AP given current set S.
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # If we add x, it might be the 1st, 2nd, 3rd, or 4th term.
    # It creates a 4-AP if the other 3 terms are already in S.
    # So, conflicts[x] should be a list of "triples" of indices that, if all present, would form a 4-AP with x.
    # However, checking "all present" is expensive.
    # Alternative: Precompute for each x, the list of sets of 3 elements that would complete a 4-AP.
    # There are at most (N/3) such sets per x.
    
    # Let's use a different strategy for speed:
    # We will maintain a bitmask of the current set.
    # To check if adding x is valid:
    #   For each d such that x, x+d, x+2d, x+3d in [1, N]:
    #       if (mask & (1<<(x+d)) & (1<<(x+2d)) & (1<<(x+3d))) != 0: invalid
    #   Similarly for other positions.
    # This is O(N) per check. With N=256, this is fast enough.
    # But we can precompute the masks for each position and d.
    
    # masks_1st[x][d] = bitmask of the other 3 terms if x is 1st
    # masks_2nd[x][d] = bitmask of the other 3 terms if x is 2nd
    # masks_3rd[x][d] = bitmask of the other 3 terms if x is 3rd
    # masks_4th[x][d] = bitmask of the other 3 terms if x is 4th
    
    # Actually, simpler:
    # For each x, generate a list of bitmasks. If (current_mask & bm) == bm, then adding x is invalid.
    
    invalid_masks = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x is 1st: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            bm = (1 << (x + d - 1)) | (1 << (x + 2*d - 1)) | (1 << (x + 3*d - 1))
            invalid_masks[x].append(bm)
            
        # x is 2nd: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            bm = (1 << (x - d - 1)) | (1 << (x + d - 1)) | (1 << (x + 2*d - 1))
            invalid_masks[x].append(bm)
            
        # x is 3rd: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            bm = (1 << (x - 2*d - 1)) | (1 << (x - d - 1)) | (1 << (x + d - 1))
            invalid_masks[x].append(bm)
            
        # x is 4th: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            bm = (1 << (x - 3*d - 1)) | (1 << (x - 2*d - 1)) | (1 << (x - d - 1))
            invalid_masks[x].append(bm)
            
    return invalid_masks

def _count_3ap_mask(mask):
    """Count 3-APs using numpy from bitmask."""
    # Convert mask to array
    # 256 bits -> 32 uint64s or just use np.unpackbits on a bytes representation?
    # np.frombuffer is fast.
    # Or just loop over d and use bit operations?
    # Vectorized numpy is likely faster for counting.
    
    # Create a boolean array of length N
    # mask is a Python int.
    # We can use:
    # arr = np.zeros(N, dtype=bool)
    # for i in range(N):
    #     if (mask >> i) & 1: arr[i] = True
    # This is O(N) python loop, which is slow.
    
    # Better:
    # b = mask.to_bytes(32, 'little') # 256 bits
    # arr = np.frombuffer(b, dtype=np.uint8)
    # arr = arr.reshape(32, 8) # 32 bytes, 8 bits each
    # arr = np.unpackbits(arr, axis=1)
    # arr = arr.flatten()
    # arr = arr[:N] # ensure length N
    
    # Let's try this.
    b = mask.to_bytes(32, 'little')
    arr = np.frombuffer(b, dtype=np.uint8)
    arr = np.unpackbits(arr, axis=1).flatten()
    arr = arr[:N]
    
    count = 0
    for d in range(1, (N + 1) // 2):
        max_a = N - 2 * d
        if max_a < 1:
            break
        # 0-indexed: a-1, a+d-1, a+2d-1
        # a ranges from 1 to max_a
        # indices: 0 to max_a-1
        count += int(np.sum(arr[:max_a] & arr[d:d+max_a] & arr[2*d:2*d+max_a]))
    return count

def _is_valid_addition(mask, x, invalid_masks):
    """Check if adding x to mask is valid (no 4-AP)."""
    for bm in invalid_masks[x]:
        if (mask & bm) == bm:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    invalid_masks = _precompute()
    
    # Initial solution: Greedy
    # Try a few random greedy orders to get a good starting point
    rng = np.random.default_rng(42)
    
    best_mask = 0
    best_score = -1
    
    # Generate initial candidates
    for _ in range(20):
        order = rng.permutation(N).tolist()
        mask = 0
        for x in order:
            if _is_valid_addition(mask, x, invalid_masks):
                mask |= (1 << (x - 1))
        
        score = _count_3ap_mask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    if best_mask == 0:
        return [1]
        
    # Simulated Annealing
    current_mask = best_mask
    current_score = best_score
    T = 1.0
    cooling_rate = 0.995
    
    # We need to be able to remove elements too.
    # Move: add x, remove x, or swap x, y.
    # Adding x: if valid, new_score = current_score + delta.
    #   delta = number of new 3-APs formed by x.
    # Removing x: always valid. new_score = current_score - delta.
    #   delta = number of 3-APs involving x.
    # Swap: remove x, add y.
    
    # To speed up, we can compute the "contribution" of each element to the 3-AP count.
    # contrib[x] = number of 3-APs where x is a member.
    # If we add x, new 3-APs are those where x is a member and the other 2 are already in set.
    # This is still O(N) to compute per x.
    
    # Let's stick to full re-evaluation for simplicity first, but optimize the counting.
    # _count_3ap_mask is O(N^2) roughly (sum of N-2d). ~N^2/4 = 16k ops.
    # With numpy, this is very fast.
    
    while time.time() < deadline:
        # Generate a random move
        move_type = rng.random()
        if move_type < 0.3:
            # Add a random element
            x = rng.integers(1, N + 1)
            if not (current_mask & (1 << (x - 1))):
                if _is_valid_addition(current_mask, x, invalid_masks):
                    new_mask = current_mask | (1 << (x - 1))
                    new_score = _count_3ap_mask(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < np.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                else:
                    # Invalid, skip
                    pass
        elif move_type < 0.6:
            # Remove a random element
            # Find a set bit
            if current_mask:
                # Get list of set bits? Or just pick random x and check if set.
                x = rng.integers(1, N + 1)
                if current_mask & (1 << (x - 1)):
                    new_mask = current_mask & ~(1 << (x - 1))
                    new_score = _count_3ap_mask(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < np.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
        else:
            # Swap
            x1 = rng.integers(1, N + 1)
            x2 = rng.integers(1, N + 1)
            if x1 == x2:
                continue
            b1 = 1 << (x1 - 1)
            b2 = 1 << (x2 - 1)
            
            # Case 1: x1 in set, x2 not in set -> remove x1, add x2
            if (current_mask & b1) and not (current_mask & b2):
                temp_mask = current_mask & ~b1
                if _is_valid_addition(temp_mask, x2, invalid_masks):
                    new_mask = temp_mask | b2
                    new_score = _count_3ap_mask(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < np.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            # Case 2: x2 in set, x1 not in set -> remove x2, add x1
            elif (current_mask & b2) and not (current_mask & b1):
                temp_mask = current_mask & ~b2
                if _is_valid_addition(temp_mask, x1, invalid_masks):
                    new_mask = temp_mask | b1
                    new_score = _count_3ap_mask(new_mask)
                    delta = new_score - current_score
                    if delta > 0 or rng.random() < np.exp(delta / T):
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
                            
        # Cool down
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    # Final local search polish on best_mask
    # Try adding any valid element that increases score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not (best_mask & (1 << (x - 1))):
                if _is_valid_addition(best_mask, x, invalid_masks):
                    new_mask = best_mask | (1 << (x - 1))
                    new_score = _count_3ap_mask(new_mask)
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                        improved = True
                        
    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return result if result else [1]

# EVOLVE-BLOCK-END
