"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct a large subset of [1, 512] with no 4-term AP.
    
    Strategy: Use Salem-Spencer base-3 construction on overlapping blocks,
    then merge with conflict-aware union and local repair.
    """
    
    start_time = time.time()
    deadline = start_time + 88.0
    
    N = 512
    
    # Precompute conflicts for validity checking
    conflicts = [[] for _ in range(N)]
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[a - 1].append(mask_a)
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            conflicts[b - 1].append(mask_b)
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            conflicts[c - 1].append(mask_c)
            mask_e = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[e - 1].append(mask_e)
    
    def is_valid_addition(x, S):
        idx = x - 1
        for m in conflicts[idx]:
            if (S & m) == m:
                return False
        return True
    
    def salem_spencer_block(start, length):
        """
        Generate a 4-AP-free subset of [start, start+length-1] using base-3 construction.
        Numbers with only 0s and 1s in base 3 are 4-AP-free.
        """
        result = []
        for x in range(start, start + length):
            # Check if x in base 3 has only digits 0 and 1
            y = x
            valid = True
            while y > 0:
                if y % 3 == 2:
                    valid = False
                    break
                y //= 3
            if valid:
                result.append(x)
        return result
    
    def greedy_construct_bitmask(order=None):
        S = 0
        if order is None:
            order = list(range(1, N + 1))
            random.shuffle(order)
        for x in order:
            if is_valid_addition(x, S):
                S |= (1 << (x - 1))
        return S
    
    # Phase 1: Generate block-based seeds using Salem-Spencer
    block_len = 81
    overlap = 10
    step = block_len - overlap
    
    best_S = 0
    best_size = 0
    
    # Try multiple block configurations
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        
        # Randomize block starting offset
        offset = random.randint(0, 20)
        
        # Generate seeds from overlapping blocks
        S = 0
        for start in range(1 + offset, N + 1, step):
            end = min(start + block_len - 1, N)
            block = salem_spencer_block(start, end - start + 1)
            # Add elements from this block if valid
            for x in block:
                if not (S & (1 << (x - 1))):
                    if is_valid_addition(x, S):
                        S |= (1 << (x - 1))
        
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
    
    # Phase 2: Greedy restarts with random order
    for _ in range(40):
        if time.time() > deadline - 10:
            break
        S = greedy_construct_bitmask()
        size = bin(S).count('1')
        if size > best_size:
            best_size = size
            best_S = S
    
    # Phase 3: Simulated Annealing
    current_S = best_S
    current_size = best_size
    
    T = 1.5
    cooling = 0.9998
    min_T = 0.005
    
    while time.time() < deadline - 3 and T > min_T:
        if random.random() < 0.55:
            # Try to add
            for _ in range(15):
                x = random.randint(1, N)
                if not (current_S & (1 << (x - 1))):
                    if is_valid_addition(x, current_S):
                        current_S |= (1 << (x - 1))
                        current_size += 1
                        break
        else:
            # Try to remove
            for _ in range(15):
                x = random.randint(1, N)
                if current_S & (1 << (x - 1)):
                    if random.random() < np.exp(-1.0 / T):
                        current_S &= ~(1 << (x - 1))
                        current_size -= 1
                        break
        
        T *= cooling
        
        if current_size > best_size:
            best_size = current_size
            best_S = current_S
    
    # Phase 4: Final local search - try to add any remaining elements
    for x in range(1, N + 1):
        if not (best_S & (1 << (x - 1))):
            if is_valid_addition(x, best_S):
                best_S |= (1 << (x - 1))
                best_size += 1
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if best_S & (1 << i):
            result.append(i + 1)
    
    # Verification
    def has_4ap(s):
        s_set = set(s)
        for a in s_set:
            for b in s_set:
                if a >= b:
                    continue
                d = b - a
                if d == 0:
                    continue
                c = b + d
                e = c + d
                if c in s_set and e in s_set:
                    return True
        return False
    
    if has_4ap(result):
        return [1]
        
    return sorted(result)

# EVOLVE-BLOCK-END
