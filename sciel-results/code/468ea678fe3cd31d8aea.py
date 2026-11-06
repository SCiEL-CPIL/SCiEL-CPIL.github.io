import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target n values to try, in descending order
    # We aim for 26, the known maximum from Lazebnik-Woldar.
    target_ns = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    # Fallback result
    # A trivial coloring for n=11 (or smaller) is always valid if we just return it.
    # But we want to maximize n.
    # Let's ensure we have a valid small n result.
    # For n=5, any coloring works? No, K5 has no C4.
    # For n=6, K6 has C4s.
    # We can use a simple greedy or random search for small n if large n fails.
    
    best_result = None
    best_n_found = 0
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        # Try to find a valid coloring for this n
        result = find_coloring(n, time_limit=50)
        if result is not None:
            best_result = result
            best_n_found = n
            break # Found the largest n, so stop
            
    if best_result is None:
        # Fallback: return a valid small n
        # n=1 is trivial
        return {'n': 1, 'colors': []}
        
    return best_result

def find_coloring(n, time_limit=10):
    """
    Attempts to find a valid 5-coloring of K_n with no C4 in any color class.
    Returns a dict with 'n' and 'colors' if successful, else None.
    """
    if n < 5:
        # Trivial cases
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        return {'n': n, 'colors': colors}
        
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # colors[e] is the color of edge e (0-4)
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Precompute edge index for fast access
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # Neighbor bitmasks for each color and vertex
    # neighbors[c][v] is a bitmask of neighbors of v in color c
    neighbors = [[0] * n for _ in range(5)]
    
    # Initialize neighbors
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
        
    # Calculate initial violations
    # Violation count = sum over all colors c, and all pairs (u, v) of C(common_neighbors_c(u, v), 2)
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = bin(common).count('1')
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
        
    current_violations = count_violations()
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    # Simulated Annealing
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    start_time = time.time()
    iterations = 0
    max_iterations = 1000000
    
    best_violations = current_violations
    best_colors = colors[:]
    best_neighbors = [row[:] for row in neighbors]
    
    while time.time() < start_time + time_limit and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = random.randint(0, 4)
        if new_color == old_color:
            continue
            
        # Calculate delta in violations
        # This requires updating neighbors and recalculating affected violations.
        # For simplicity and speed in Python, we might just recalculate total violations
        # if the incremental update is too complex to implement correctly in a short time.
        # However, O(n^2) recalculation is slow for n=26 (676 pairs * 5 colors * 26^2/2? No, 5 * 325 * bitcount).
        # Bitcount is fast. 5 * 325 = 1625 bitcounts per iteration.
        # 1625 * 10000 iterations = 16 million bitcounts. Should be okay in 10s?
        # Let's try the simple recalculation first, but optimize the bitcount.
        
        # Update neighbors
        neighbors[old_color][u] &= ~(1 << v)
        neighbors[old_color][v] &= ~(1 << u)
        neighbors[new_color][u] |= (1 << v)
        neighbors[new_color][v] |= (1 << u)
        colors[e_idx] = new_color
        
        new_violations = count_violations()
        
        # Acceptance criterion
        if new_violations < current_violations or random.random() < pow(2.71828, (current_violations - new_violations) / T):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                best_neighbors = [row[:] for row in neighbors]
                
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
        else:
            # Reject: revert
            neighbors[old_color][u] |= (1 << v)
            neighbors[old_color][v] |= (1 << u)
            neighbors[new_color][u] &= ~(1 << v)
            neighbors[new_color][v] &= ~(1 << u)
            colors[e_idx] = old_color
            
        T *= cooling
        if T < min_T:
            T = min_T
            # Restart with randomization?
            # If stuck, we can restart the temperature or re-randomize some edges.
            
    return None
