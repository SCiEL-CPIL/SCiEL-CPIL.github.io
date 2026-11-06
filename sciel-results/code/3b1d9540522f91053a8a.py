import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    N = 32
    MAX_VAL = 512
    
    # Precompute AP masks for each difference d
    # For a given d, an AP is (a, a+d, a+2d).
    # We want to know if there exists ANY a such that all three are in S.
    # Instead of checking each a, we can use bitsets.
    # Let S_bits be a bitset of length MAX_VAL+1.
    # For a fixed d, the condition "exists a: a, a+d, a+2d in S" is equivalent to:
    # There is no direct single bitwise operation on S_bits that gives "exists a" directly 
    # unless we shift and AND.
    # (S_bits & (S_bits >> d) & (S_bits >> 2*d)) != 0 ?
    # Wait, S_bits is an array of 0/1.
    # If we treat it as an integer bitset, shifting is fast.
    # But MAX_VAL=512 is too large for a single Python int shift to be faster than numpy?
    # Actually, Python ints are arbitrary precision. 512 bits is tiny.
    # Let's use Python integers for the bitset. This is extremely fast for bitwise ops.
    
    # Precompute masks for each d?
    # For a fixed d, the set of valid 'a' positions is {1, ..., 512-2d}.
    # We can precompute a mask valid_a_mask[d] which has bits set for a in 1..512-2d.
    # But we don't need the mask for 'a' if we just check if the intersection is non-empty.
    # Let S_int be the integer representation of the set S.
    # Bit i is set if i in S.
    # For difference d:
    # We need to check if there exists a such that bit a, bit a+d, bit a+2d are all 1.
    # This is equivalent to: (S_int & (S_int >> d) & (S_int >> 2*d)) & valid_a_mask[d] != 0
    # Where valid_a_mask[d] has 1s at positions a where a+2d <= 512.
    
    valid_a_masks = []
    for d in range(1, MAX_VAL // 2 + 1):
        mask = 0
        for a in range(1, MAX_VAL - 2 * d + 1):
            mask |= (1 << a)
        valid_a_masks.append(mask)
        
    # Function to count distinct d for a given integer bitset S_int
    def count_diffs(S_int):
        count = 0
        # We can iterate d from 1 to 256
        # For each d, check condition
        for d in range(1, 257):
            # Check if any AP exists for this d
            # S_int & (S_int >> d) & (S_int >> 2*d)
            # We need to mask out bits that correspond to a > 512-2d
            # Actually, if a > 512-2d, then a+2d > 512, so bit a+2d would be 0 in S_int (since S only has bits 1..512).
            # So we don't strictly need valid_a_masks if we assume bits > 512 are 0.
            # Let's verify: S_int only has bits 1..512 set.
            # If a is such that a+2d > 512, then bit a+2d is 0. So (S_int >> 2d) will have 0 at position a.
            # So the AND will be 0 at that position.
            # So we can just check if (S_int & (S_int >> d) & (S_int >> 2*d)) != 0.
            # Wait, this checks if there is ANY a.
            # Is it possible that the AND is non-zero due to bits > 512? No, S_int has 0 there.
            # So this works.
            if (S_int & (S_int >> d) & (S_int >> 2 * d)) != 0:
                count += 1
        return count

    # Initial random set
    def random_set():
        s = set(random.sample(range(1, MAX_VAL + 1), N))
        s_int = 0
        for x in s:
            s_int |= (1 << x)
        return s, s_int

    # Simulated Annealing
    best_set, best_int = random_set()
    best_score = count_diffs(best_int)
    
    current_set, current_int = best_set, best_int
    current_score = best_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Number of steps we can do?
    # count_diffs is ~256 iterations of simple bit ops. Very fast.
    # Swap evaluation:
    # 1. Remove old x: current_int &= ~(1 << old_x)
    # 2. Add new x: current_int |= (1 << new_x)
    # 3. Count diffs: ~256 ops.
    # 4. Revert if needed.
    # This is extremely fast. We can do many steps.
    
    while time.time() < deadline and T > T_min:
        # Pick random index in current_set
        old_x = random.choice(list(current_set))
        # Pick random new_x not in current_set
        new_x = random.randint(1, MAX_VAL)
        while new_x in current_set:
            new_x = random.randint(1, MAX_VAL)
        
        # Apply swap
        temp_int = (current_int & ~(1 << old_x)) | (1 << new_x)
        temp_score = count_diffs(temp_int)
        
        delta = temp_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            current_set.remove(old_x)
            current_set.add(new_x)
            current_int = temp_int
            current_score = temp_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
                best_int = current_int
                
        T *= cooling
        
    return list(best_set)

# EVOLVE-BLOCK-END
