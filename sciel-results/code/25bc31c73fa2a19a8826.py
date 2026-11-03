import math

def construct():
    best_score = 0
    best_set = [0, 1, 2, 5]
    
    # Search for optimal k in the construction U = {0, ..., k} U {2k+1}
    # This construction maximizes the difference/sum ratio for small max elements
    # by creating a single large gap that prevents sumset overlaps while
    # keeping the difference set components disjoint.
    
    for k in range(1, 100):
        # Construct the set
        u = list(range(k + 1))
        max_u = 2 * k + 1
        u.append(max_u)
        
        # Calculate score
        # We can use the derived formulas for this specific structure
        # |U+U| = 3k + 3
        # |U-U| = 4k + 3
        # These are valid for M = 2k+1
        
        num_sums = 3 * k + 3
        num_diffs = 4 * k + 3
        
        if num_sums == 0:
            continue
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            continue
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            continue
            
        score = 1 + math.log2(ratio) / denom
        
        if score > best_score:
            best_score = score
            best_set = u

    return best_set
