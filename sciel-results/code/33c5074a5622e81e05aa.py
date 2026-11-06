import time
import random
from itertools import combinations

ALL_MASKS = []
ALL_SETS = []

def init():
    global ALL_MASKS, ALL_SETS
    for comb in combinations(range(16), 4):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        ALL_MASKS.append(mask)
        ALL_SETS.append(comb)

def generate_conflicts(mask_a, mask_b):
    """
    Returns a set of masks C such that {A, B, C} is a sunflower.
    A and B are distinct masks.
    """
    k_mask = mask_a & mask_b
    union_mask = mask_a | mask_b
    # R is the set of bits not in A or B
    # We need to pick 4 - popcount(k_mask) bits from R
    # and OR them with k_mask.
    
    # Calculate R bits
    r_bits = []
    for i in range(16):
        if not (union_mask & (1 << i)):
            r_bits.append(i)
            
    k_size = bin(k_mask).count('1')
    need = 4 - k_size
    
    if need < 0:
        return set()
        
    if need > len(r_bits):
        return set()
        
    conflicts = set()
    # Generate combinations of 'need' bits from r_bits
    for comb in combinations(r_bits, need):
        c_mask = k_mask
        for i in comb:
            c_mask |= (1 << i)
        conflicts.add(c_mask)
        
    return conflicts

def construct():
    init()
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_size = 0
    best_solution = []
    
    all_masks = ALL_MASKS
    n_masks = len(all_masks)
    
    while time.time() < deadline:
        random.shuffle(all_masks)
        selected = []
        forbidden = set()
        
        for mask in all_masks:
            if mask in forbidden:
                continue
                
            # Check if mask is already selected? 
            # Since we iterate unique masks and only add if not forbidden, 
            # and we add selected to forbidden? 
            # Wait, if we select mask, do we add it to forbidden?
            # If we select mask M, it shouldn't be selected again.
            # But we iterate each mask once. So no issue.
            
            # Is it valid?
            # We need to ensure it doesn't form a sunflower with any pair in selected.
            # Our 'forbidden' set contains all masks that form a sunflower with some pair in selected.
            # So if mask is not in forbidden, it is safe to add.
            
            # Add mask to selected
            selected.append(mask)
            
            # Update forbidden
            # For each existing set Y in selected (excluding the new one),
            # add conflicts of {mask, Y} to forbidden
            # Note: selected[-1] is the new mask.
            for i in range(len(selected) - 1):
                y_mask = selected[i]
                conf = generate_conflicts(mask, y_mask)
                forbidden.update(conf)
                
            # Optimization: If the new mask itself is in conflicts of some pair?
            # No, conflicts are for the *third* set C.
            # The pair is {A, B}. C is the one forbidden.
            # So when we add M (as the third set C to some pair A,B), 
            # M would have been in forbidden if A,B were already in selected.
            # But we are adding M now.
            # We need to forbid *future* sets C that would form sunflowers with pairs involving M.
            # So we update forbidden with conflicts of {M, Y} for all Y in selected.
            # This is exactly what the loop does.
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = selected[:]
            
            # Optional: Break if we hit the theoretical max or a high bar?
            # Max is 250.
            if best_size >= 250:
                break

    # Convert masks back to sorted lists of 1..16
    result = []
    for mask in best_solution:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result

# Pre-compute conflicts for all pairs to speed up?
# There are 1820 masks. Pairs ~ 1.6M.
# Storing a set of conflicts for each pair is too much memory.
# But we can cache generate_conflicts.
from functools import lru_cache

@lru_cache(maxsize=None)
def gen_conflicts_cached(mask_a, mask_b):
    return frozenset(generate_conflicts(mask_a, mask_b))

# Modify construct to use cache
# Note: generate_conflicts returns a set. lru_cache requires hashable return or just args.
# It's better to cache the function result.

# Let's rewrite construct to use the cached version.
