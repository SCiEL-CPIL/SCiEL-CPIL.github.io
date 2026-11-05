import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    N = 512
    K = 32
    
    # Initialize mask with K random bits
    # Values are 1..512, so bits 0..511
    indices = random.sample(range(N), K)
    mask = 0
    for i in indices:
        mask |= (1 << i)
        
    def get_score(m):
        score = 0
        # d can be at most 255 since 2*d < 512
        for d in range(1, 256):
            # Check if there exists a such that a, a+d, a+2d are in m
            # m has bit i set if value i+1 is in set.
            # Let's say bit i corresponds to value i. 
            # Wait, if bit i corresponds to value i+1, then:
            # Value a is bit a-1.
            # Value a+d is bit a+d-1.
            # Value a+2d is bit a+2d-1.
            # We want to check if (bit a-1) AND (bit a+d-1) AND (bit a+2d-1) is 1 for some a.
            # Let M be the mask.
            # M & (M >> d) & (M >> 2d)
            # If bit k is set in M, it represents value k+1.
            # If bit k is set in M >> d, it means bit k+d was set in M, representing value k+d+1.
            # So if bit k is set in M & (M >> d), it means value k+1 and value k+d+1 are in S.
            # The difference is (k+d+1) - (k+1) = d.
            # Then M >> 2d: bit k set means bit k+2d was set in M, representing value k+2d+1.
            # So if bit k is set in M & (M >> d) & (M >> 2d), we have values:
            # v1 = k+1
            # v2 = k+d+1
            # v3 = k+2d+1
            # v2 - v1 = d
            # v3 - v2 = d
            # So yes, this logic holds.
            
            if (m & (m >> d) & (m >> (2*d))) != 0:
                score += 1
        return score

    current_score = get_score(mask)
    best_mask = mask
    best_score = current_score
    
    # Local search
    while time.time() < deadline:
        # Pick a random element to remove and one to add
        # Get list of indices in mask and not in mask
        # This is slow if done every step. 
        # We can maintain lists or just pick random bits and check.
        
        # Efficient way: 
        # Pick random i in 0..511. If bit i is set, it's a candidate to remove.
        # Pick random j in 0..511. If bit j is not set, it's a candidate to add.
        
        i = random.randint(0, N-1)
        if not (mask & (1 << i)):
            # i is not in set, so we can't remove it. 
            # Instead, let's just pick two random bits, one from set, one from not set.
            # To avoid bias, we can sample until we find one.
            # Or, maintain a list of set bits and unset bits.
            # Given N=512, maintaining lists is fine.
            pass
            
    # Let's optimize the selection of i and j.
    # We'll maintain a list of set bits `set_bits` and unset bits `unset_bits`.
    
    set_bits = [i for i in range(N) if (mask >> i) & 1]
    unset_bits = [i for i in range(N) if not ((mask >> i) & 1)]
    
    current_score = get_score(mask)
    best_mask = mask
    best_score = current_score
    
    while time.time() < deadline:
        # Pick random i from set_bits, j from unset_bits
        idx_i = random.randint(0, len(set_bits) - 1)
        idx_j = random.randint(0, len(unset_bits) - 1)
        
        i = set_bits[idx_i]
        j = unset_bits[idx_j]
        
        # Swap
        new_mask = mask ^ (1 << i) ^ (1 << j)
        
        new_score = get_score(new_mask)
        
        if new_score > current_score:
            # Accept
            mask = new_mask
            current_score = new_score
            
            # Update lists
            set_bits[idx_i] = j
            unset_bits[idx_j] = i
            # Note: The lists are no longer sorted, but we just need membership and random access.
            # The values in the lists are the indices.
            # set_bits now contains j instead of i.
            # unset_bits now contains i instead of j.
            
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
                
        elif new_score == current_score and random.random() < 0.1:
            # Accept with probability to explore
            mask = new_mask
            # current_score unchanged
            set_bits[idx_i] = j
            unset_bits[idx_j] = i
            
        else:
            # Reject, do nothing
            pass

    # Convert best_mask to list of integers
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
