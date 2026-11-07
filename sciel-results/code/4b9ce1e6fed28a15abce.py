import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of distinct integers in 1..256 maximizing size k
    such that subset sums of different cardinalities are disjoint.
    Uses an Arithmetic Progression (AP) model.
    """
    max_val = 256
    deadline = time.time() + 50  # Safety margin
    
    best_set = None
    best_k = 0

    # Search for largest k
    for k in range(60, 0, -1):
        if time.time() > deadline:
            break
            
        found = False
        # d must be small enough to fit in 256
        # x_1 > d * max_j j(k-j)
        # x_k = x_1 + (k-1)d <= 256
        # So d * (max_j j(k-j) + k - 1) < 256
        max_prod = 0
        for j in range(1, k):
            prod = j * (k - j)
            if prod > max_prod:
                max_prod = prod
        
        # Upper bound for d
        # d * (max_prod + 1 + k - 1) <= 256
        # d * (max_prod + k) <= 256
        max_d = 256 // (max_prod + k) if (max_prod + k) > 0 else 256
        
        for d in range(1, max_d + 1):
            if time.time() > deadline:
                break
            
            # Required x_1
            # x_1 > d * max_prod
            min_x1 = d * max_prod + 1
            
            # Check range
            max_val_in_set = min_x1 + (k - 1) * d
            if max_val_in_set <= max_val:
                # Construct set
                set_vals = [min_x1 + i * d for i in range(k)]
                
                # Verify distinctness and range
                if len(set(set_vals)) == k and all(1 <= v <= max_val for v in set_vals):
                    # The condition min(S_{j+1}) > max(S_j) is satisfied by construction
                    # x_1 > d * j(k-j) for all j
                    # This ensures disjointness of cardinality sums
                    best_set = set_vals
                    best_k = k
                    found = True
                    break
        
        if found:
            break

    if best_set is None:
        # Fallback: return a trivial valid set
        # {1} is valid
        return [1]
        
    return best_set
# EVOLVE-BLOCK-END
