# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_sunflower(existing, new):
    """
    Check if adding `new` to `existing` creates a sunflower violation.
    For every pair (a, b) in `existing`, check if {a, b, new} is a sunflower.
    A sunflower is 3 distinct sets with all pairwise intersections equal.
    """
    for i in range(len(existing)):
        a = existing[i]
        for j in range(i + 1, len(existing)):
            b = existing[j]
            # Check if a, b, new form a sunflower
            # Intersections: a&b, a&new, b&new must be equal
            ab = a & b
            an = a & new
            bn = b & new
            if ab == an == bn:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 60s limit (or 90s)
    
    # All 4-subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    
    # Greedy initialization with random shuffles to find a large independent set
    best_family = []
    
    # Precompute intersections for speed? 
    # Number of sets is C(16,4) = 1820. 
    # Pairwise intersections: 1820^2 / 2 ~ 1.6M operations. This is small.
    # We can precompute a bitset representation or just use sets.
    
    # Let's use a faster representation: integers (bitmasks)
    # Map element 1..16 to bit 0..15
    mask_map = {i: 1 << (i-1) for i in range(1, 17)}
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= mask_map[x]
        return m
    
    masks = [to_mask(s) for s in all_sets]
    # Store mapping back to original set for output
    mask_to_set = {to_mask(s): s for s in all_sets}
    
    # We will work with masks for intersection checks
    # Intersection of masks a, b is a & b
    
    # Greedy strategy:
    # 1. Shuffle all masks
    # 2. Iterate through shuffled masks
    # 3. If adding mask doesn't create a sunflower with any pair in current family, add it
    
    # To speed up the check, we can maintain the family as a list of masks.
    # Check complexity: O(N^2) where N is current family size.
    # Max family size is ~250.
    # 1820 candidates * 250^2/2 checks = 1820 * 31250 ~ 57M intersection checks.
    # Python set/mask intersection is fast, but 57M might be too slow in pure Python loops.
    # C(16,4) = 1820.
    # Let's optimize the inner loop.
    
    # Alternative: Randomized greedy with multiple restarts.
    
    best_size = 0
    best_masks = []
    
    num_restarts = 5000 # Adjust based on time
    random.seed(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        shuffled_masks = masks[:]
        random.shuffle(shuffled_masks)
        
        family = []
        
        for m in shuffled_masks:
            # Check if m can be added
            is_valid = True
            n = len(family)
            for i in range(n):
                a = family[i]
                for j in range(i + 1, n):
                    b = family[j]
                    if (a & b) == (a & m) == (b & m):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                family.append(m)
        
        if len(family) > best_size:
            best_size = len(family)
            best_masks = family[:]
            
        # Early stop if we hit the theoretical max or a high limit?
        # The task says max 250. If we hit 250, we can stop.
        if best_size >= 250:
            break

    # Convert best_masks back to sorted lists
    result = []
    for m in best_masks:
        s = mask_to_set[m]
        result.append(sorted(s))
    
    return result

# EVOLVE-BLOCK-END
