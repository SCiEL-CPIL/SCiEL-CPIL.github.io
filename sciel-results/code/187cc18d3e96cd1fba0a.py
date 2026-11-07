import time
import itertools

def construct():
    n = 12
    m = 40
    
    # Use universe of size 6 (vertices 0..5)
    # Vertices 6..11 are not used in any set.
    
    # Generate all subsets of {0, 1, 2, 3, 4, 5}
    universe = list(range(6))
    
    # 1. Include all subsets of size 0, 1, 2
    # Size 0: 1 set
    # Size 1: 6 sets
    # Size 2: 15 sets
    # Total: 22 sets
    selected_masks = set()
    for r in range(3):
        for combo in itertools.combinations(universe, r):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            selected_masks.add(mask)
            
    # 2. Select 3 specific size-4 sets
    # A1 = {0, 1, 2, 3}
    # A2 = {0, 1, 4, 5}
    # A3 = {2, 3, 4, 5}
    a1 = (0, 1, 2, 3)
    a2 = (0, 1, 4, 5)
    a3 = (2, 3, 4, 5)
    
    size4_sets = [a1, a2, a3]
    
    # Add these size-4 sets
    for combo in size4_sets:
        mask = 0
        for v in combo:
            mask |= (1 << v)
        selected_masks.add(mask)
        
    # 3. Add all size-3 subsets of these size-4 sets
    for combo in size4_sets:
        for sub in itertools.combinations(combo, 3):
            mask = 0
            for v in sub:
                mask |= (1 << v)
            selected_masks.add(mask)
            
    # Current count: 22 (size<=2) + 3 (size 4) + 12 (size 3 from size 4) = 37
    # Need 3 more sets.
    # Add any 3 remaining size-3 sets from the universe.
    all_size3 = list(itertools.combinations(universe, 3))
    added_count = 0
    for combo in all_size3:
        if added_count >= 3:
            break
        mask = 0
        for v in combo:
            mask |= (1 << v)
        if mask not in selected_masks:
            selected_masks.add(mask)
            added_count += 1
            
    # Convert masks to lists of integers
    result = []
    for mask in selected_masks:
        s = []
        for i in range(6):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    # Ensure exactly 40 distinct lists
    # The construction guarantees distinctness.
    # Check length
    if len(result) != 40:
        # Fallback if logic failed (should not happen)
        # Just return first 40 of all subsets
        all_subsets = []
        for mask in range(1 << 6):
            s = [i for i in range(6) if (mask >> i) & 1]
            all_subsets.append(s)
        return all_subsets[:40]
        
    return result
