import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = 16
    k = 4

    # Generate all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)
    
    num_subsets = len(all_subsets)
    
    # Precompute subsets of each mask for fast iteration?
    # Or just iterate 16 subsets on the fly.
    # submasks of a 4-bit mask:
    def get_submasks(mask):
        sub = mask
        while sub:
            yield sub
            sub = (sub - 1) & mask
        yield 0

    def is_conflict(candidate, A, B, core):
        # Check if (A \ core) & candidate == 0 and (B \ core) & candidate == 0
        # A & B == core is guaranteed by how active_pairs is built
        return ((A & ~core) & candidate) == 0 and ((B & ~core) & candidate) == 0

    best_family = []
    best_size = 0
    
    # Strategy: Randomized Greedy with active pair tracking
    # We run multiple restarts with different random seeds/orders
    
    num_restarts = 2000 # We will break early based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
        
        random.seed(42 + restart)
        order = list(range(num_subsets))
        random.shuffle(order)
        
        current_family = []
        # active_pairs[core] = set of tuples (idx_a, idx_b)
        # We use indices into all_subsets
        active_pairs = {}
        
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            
            mask = all_subsets[idx]
            
            # Check if mask can be added
            valid = True
            # Iterate all submasks of mask
            for core in get_submasks(mask):
                if core in active_pairs:
                    pairs = active_pairs[core]
                    for (i, j) in pairs:
                        A = all_subsets[i]
                        B = all_subsets[j]
                        if is_conflict(mask, A, B, core):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                current_family.append(idx)
                
                # Update active_pairs
                # Add pairs (idx, existing) for all existing in current_family
                for existing_idx in current_family[:-1]:
                    existing_mask = all_subsets[existing_idx]
                    core = mask & existing_mask
                    pair = (min(idx, existing_idx), max(idx, existing_idx))
                    if core not in active_pairs:
                        active_pairs[core] = set()
                    active_pairs[core].add(pair)
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
            # If we hit the theoretical max or a very high number, we might stop
            # Max possible is 1820, but sunflower-free is much smaller.
            # Known bounds for 4-sunflower-free families in [n] choose k?
            # For k=4, n=16, it's likely a few hundred.
            if best_size >= 250:
                 # The task says maximum=250 in the prompt text "1..maximum=250"
                 # Wait, "1..maximum=250 distinct k=4 element subsets"
                 # This phrasing "1..maximum=250" might imply the range of valid outputs is 1 to 250?
                 # Or that the max size is 250?
                 # "Maximize family cardinality."
                 # If 250 is the max, we found it.
                 break

    # Local search / Polish
    if best_size > 0 and time.time() < deadline - 10.0:
        # Try to improve by swapping
        # This is complex to implement correctly with the active_pairs structure
        # because removing a set requires removing all its pairs from active_pairs.
        # Given the time, we'll stick to the greedy result if it's good enough.
        # The greedy with random restarts is usually strong.
        pass

    # Convert back to sorted lists
    result = []
    for idx in best_family:
        mask = all_subsets[idx]
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
    
    return result
