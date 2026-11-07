"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    n = 128
    # Use bitset representation for speed
    # Bit i represents whether number i is in the set (1-indexed, so bit 1 = number 1)
    
    # Start with all large numbers (65..128)
    selected_mask = 0
    for i in range(65, 129):
        selected_mask |= (1 << i)
    
    def is_in_set(x, mask):
        return (mask >> x) & 1 == 1
    
    def add_to_set(x, mask):
        return mask | (1 << x)
    
    def remove_from_set(x, mask):
        return mask & ~(1 << x)
    
    def is_safe(x, mask):
        """Check if x can be added without creating a bad triple."""
        # For each pair (y,z) in current set, y≠z, check if {x,y,z} is bad
        # Optimize: y+z must be ≤ 128 for it to be in set
        # Also x+y ≤ 128 and x+z ≤ 128
        for y in range(1, 129):
            if not is_in_set(y, mask) or y == x:
                continue
            for z in range(y+1, 129):
                if not is_in_set(z, mask) or z == x:
                    continue
                # Check triple {x,y,z}
                s1 = x + y
                s2 = x + z
                s3 = y + z
                if s3 > 128:
                    continue
                if s1 > 128 or s2 > 128:
                    continue
                if is_in_set(s1, mask) and is_in_set(s2, mask) and is_in_set(s3, mask):
                    return False
        return True
    
    # Greedy add small numbers
    for x in range(1, 65):
        if is_safe(x, selected_mask):
            selected_mask = add_to_set(x, selected_mask)
    
    # Local search
    improved = True
    while improved:
        improved = False
        for x in range(1, 65):
            if is_in_set(x, selected_mask):
                continue
            # Try adding x by removing minimal elements
            # First try adding directly
            if is_safe(x, selected_mask):
                selected_mask = add_to_set(x, selected_mask)
                improved = True
                break
            # Try removing one element
            for y in range(1, 129):
                if not is_in_set(y, selected_mask):
                    continue
                trial = remove_from_set(y, selected_mask)
                if is_safe(x, trial):
                    selected_mask = add_to_set(x, trial)
                    improved = True
                    break
            if improved:
                break
    
    # Convert to list
    result = []
    for i in range(1, 129):
        if is_in_set(i, selected_mask):
            result.append(i)
    
    return result
# EVOLVE-BLOCK-END
