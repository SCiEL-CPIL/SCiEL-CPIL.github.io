import itertools
import time

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """
    Returns a normalized key for the line passing through p1 and p2.
    p1, p2 are (x, y) tuples.
    Line equation: Ax + By + C = 0
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    Normalize by GCD and sign.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    if A == 0 and B == 0:
        return None # Should not happen for distinct points
        
    # Find GCD of A, B, C
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
        
    g = gcd(gcd(A, B), C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    # Normalize sign: first non-zero of A, B, C should be positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 5x5 Grid
    points = [(x, y) for x in range(5) for y in range(5)]
    n_points = len(points)
    
    # Precompute line keys for all pairs
    # Index points by their position in the list
    # pair_keys[i][j] = line_key(points[i], points[j])
    pair_keys = [[None]*n_points for _ in range(n_points)]
    for i in range(n_points):
        for j in range(i+1, n_points):
            key = _line_key(points[i], points[j])
            pair_keys[i][j] = key
            pair_keys[j][i] = key

    best_count = float('inf')
    best_subset = None
    
    # Iterate all combinations of 12 points
    # Total combinations: C(25, 12) = 5,200,300
    for subset_indices in itertools.combinations(range(n_points), 12):
        if time.time() > deadline:
            break
            
        # Count ordinary lines
        # Use a dictionary to count pairs per line key
        line_pair_counts = {}
        for i in range(12):
            idx_i = subset_indices[i]
            for j in range(i+1, 12):
                idx_j = subset_indices[j]
                key = pair_keys[idx_i][idx_j]
                line_pair_counts[key] = line_pair_counts.get(key, 0) + 1
        
        # Ordinary lines are those with exactly 1 pair
        ordinary_count = 0
        for count in line_pair_counts.values():
            if count == 1:
                ordinary_count += 1
                
        if ordinary_count < best_count:
            best_count = ordinary_count
            best_subset = subset_indices
            # Optional: if we find the theoretical minimum, we can stop
            if best_count <= 6:
                break

    if best_subset is None:
        # Fallback to 3x4 grid if search failed or timed out without finding anything
        # (Though 5x5 search should find something)
        return [[x, y] for x in range(4) for y in range(3)]
        
    # Convert indices to coordinates
    result = [list(points[i]) for i in best_subset]
    return result

# EVOLVE-BLOCK-END
