import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer
    
    # Pre-compute for each element x, the list of (a, b, c) triples such that
    # adding x would complete a 4-AP (a, b, c, x) or (x, a, b, c) or (a, x, b, c) or (a, b, x, c)
    # Actually, let's precompute: for each x, what are all 4-APs containing x?
    # But that's what we have. Let's think differently.
    
    # For a bitset approach: 
    # in_set is a Python int where bit i is 1 if element i is in the set.
    # To check if adding x is safe: for each 4-AP containing x, check if the other 3 elements are all in the set.
    # We can precompute for each x a list of bitmasks representing the "other 3 elements" of each AP containing x.
    
    # Precompute: for each x, list of (other1, other2, other3) for each 4-AP containing x
    element_to_aps = [[] for _ in range(N)]
    
    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(0, max_a):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # AP is (a, b, c, e)
            # For element a: others are b, c, e
            element_to_aps[a].append((b, c, e))
            # For element b: others are a, c, e
            element_to_aps[b].append((a, c, e))
            # For element c: others are a, b, e
            element_to_aps[c].append((a, b, e))
            # For element e: others are a, b, c
            element_to_aps[e].append((a, b, c))
    
    # Precompute bitmasks for each element's APs
    # ap_mask[x][i] is a bitmask with bits set for the 3 other elements in AP i
    ap_masks = []
    for x in range(N):
        masks = []
        for (o1, o2, o3) in element_to_aps[x]:
            mask = (1 << o1) | (1 << o2) | (1 << o3)
            masks.append(mask)
        ap_masks.append(masks)
    
    # Helper: check if adding x to in_set (as Python int) is safe
    def can_add(x, in_set_int):
        for mask in ap_masks[x]:
            # Check if all 3 bits in mask are set in in_set_int
            if (in_set_int & mask) == mask:
                return False
        return True
    
    # Helper: add x to in_set_int
    def add_to_int(x, in_set_int):
        return in_set_int | (1 << x)
    
    # Helper: remove x from in_set_int
    def remove_from_int(x, in_set_int):
        return in_set_int & ~(1 << x)
    
    # Helper: get size
    def get_size_int(in_set_int):
        return bin(in_set_int).count('1')
    
    # 1. Initial Greedy Pass with bitset - run many more restarts
    best_in_set_int = 0
    best_size = 0
    
    # Run many greedy passes - bitset makes each pass fast
    num_restarts = 500  # Can do many more with bitset
    for _ in range(num_restarts):
        if time.time() > deadline - 5:
            break
        in_set_int = 0
        candidates = list(range(N))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, in_set_int):
                in_set_int = add_to_int(x, in_set_int)
        
        size = get_size_int(in_set_int)
        if size > best_size:
            best_size = size
            best_in_set_int = in_set_int
    
    # 2. Local Search with Kicks using bitset
    current_in_set_int = best_in_set_int
    current_size = best_size
    
    kick_sizes = [1, 2, 3]
    
    while time.time() < deadline:
        # Perform a kick
        k = random.choice(kick_sizes)
        
        # Get list of elements in set
        current_elements = [i for i in range(N) if (current_in_set_int >> i) & 1]
        
        if len(current_elements) < k:
            for x in range(N):
                if not (current_in_set_int >> x) & 1:
                    if can_add(x, current_in_set_int):
                        current_in_set_int = add_to_int(x, current_in_set_int)
            continue
            
        # Select k random elements to remove
        removed_elements = random.sample(current_elements, k)
        
        # Store state before kick
        old_in_set_int = current_in_set_int
        
        # Remove elements
        for x in removed_elements:
            current_in_set_int = remove_from_int(x, current_in_set_int)
            
        # Try to add back elements
        for x in range(N):
            if not (current_in_set_int >> x) & 1:
                if can_add(x, current_in_set_int):
                    current_in_set_int = add_to_int(x, current_in_set_int)
        
        new_size = get_size_int(current_in_set_int)
        
        # Acceptance
        if new_size >= current_size:
            current_size = new_size
        else:
            # Revert
            current_in_set_int = old_in_set_int
    
    # Final polish
    for x in range(N):
        if not (current_in_set_int >> x) & 1:
            if can_add(x, current_in_set_int):
                current_in_set_int = add_to_int(x, current_in_set_int)
    
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if (current_in_set_int >> i) & 1]
    return sorted(result)
