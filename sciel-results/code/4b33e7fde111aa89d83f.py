import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    # Bitmask representation: bit i is set if i is in the set
    # We use 1..512, so we need bits 1..512. 
    # Using a Python int as the bitmask.
    
    def score_mask(mask):
        """
        Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set.
        mask: int where bit i is 1 if i is in the set.
        """
        valid_diffs = set()
        # To find a, a+d, a+2d, we can iterate over possible a and d.
        # However, iterating all a, d is 512*256 = 131k ops. 
        # A faster way: 
        # For a fixed d, a is in set, a+d in set, a+2d in set.
        # This is equivalent to: (mask & (mask >> d) & (mask >> 2*d)) != 0
        # But we only care about d such that this is true.
        # Max d is such that a+2d <= 512 => 2d <= 511 => d <= 255.
        
        # Optimization: 
        # If we have the mask, we can check each d from 1 to 255.
        # Check if there exists any a such that bits a, a+d, a+2d are set.
        # This is (mask & (mask >> d) & (mask >> 2*d)) != 0
        
        count = 0
        for d in range(1, 256):
            # Shift mask right by d and 2d
            # If the bitwise AND of the original, shifted by d, and shifted by 2d is non-zero,
            # then there is at least one triplet.
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        # Iterate through bits 1 to 512
        for i in range(1, 513):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Initial state: Random set of 32 distinct integers in 1..512
    # We want to start with a decent configuration. 
    # A random start is okay, but maybe a "structured" random start is better?
    # Let's just use a random start, but ensure validity.
    
    # Generate initial random set
    init_list = random.sample(range(1, 513), N)
    current_mask = list_to_mask(init_list)
    current_score = score_mask(current_mask)
    
    best_mask = current_mask
    best_score = current_score
    
    # Simulated Annealing
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling = 0.995  # Cooling factor
    
    # We will perform moves until time limit
    # Move: Pick a random element in the set, remove it, pick a random element NOT in the set, add it.
    # This maintains size 32.
    
    while time.time() < deadline:
        # Generate neighbor
        # Pick index to remove
        remove_idx = random.randint(0, N - 1)
        # We need to map index in list to value. 
        # Maintaining a list in sync with mask is overhead.
        # Alternative: 
        # 1. Get list from mask (slow but simple). 
        # 2. Or maintain a list and update it.
        
        # Let's maintain a list `current_list` to make sampling easier.
        # But updating list on swap is O(N). N=32 is small.
        
        # Actually, let's just keep the list.
        current_list = mask_to_list(current_mask)
        
        # Pick element to remove
        val_out = current_list[random.randint(0, N - 1)]
        
        # Pick element to add
        # We need a random number in 1..512 not in current_list.
        # Since 512 is small, we can just pick random until not in set.
        # Or pick from a precomputed complement? 
        # Simpler: 
        val_in = random.randint(1, 512)
        while val_in in current_list:
            val_in = random.randint(1, 512)
        
        # Apply move
        new_list = current_list.copy()
        new_list[new_list.index(val_out)] = val_in
        # No need to sort for mask conversion, but score_mask doesn't care about order.
        
        new_mask = list_to_mask(new_list)
        new_score = score_mask(new_mask)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.71828, delta / T):
            # Accept
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    return mask_to_list(best_mask)
# EVOLVE-BLOCK-END
