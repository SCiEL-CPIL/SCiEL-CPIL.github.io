import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Try to find valid coloring for n=26 first (known maximum)
    # Then fall back to smaller n if needed
    
    for n in [26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 5:
            break
        result = find_coloring(n, time_remaining=deadline - time.time())
        if result is not None:
            return result
    
    # Fallback: return a trivial valid result
    return {'n': 1, 'colors': []}

def find_coloring(n, time_limit):
    """
    Find a valid 5-coloring of K_n with no C4 in any color class.
    Uses incremental violation counting and simulated annealing.
    """
    if n < 5:
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        return {'n': n, 'colors': colors}
    
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Initialize with random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    
    # Neighbor bitmasks for each color and vertex
    neighbors = [[0] * n for _ in range(5)]
    
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        neighbors[c][i] |= (1 << j)
        neighbors[c][j] |= (1 << i)
    
    # Precompute degree counts and violation contributions
    # For each color c and vertex v, we track the number of C4s involving v
    # A C4 exists if two vertices u,v have at least 2 common neighbors in color c
    # We count violations as sum over all pairs (u,v) of C(common(c,u,v), 2)
    
    def compute_violation_matrix():
        """Compute violation count for each color-vertex pair."""
        # violations[c][v] = number of C4s where v is one of the two non-adjacent vertices
        # Actually, let's just compute total violations incrementally
        return 0
    
    def count_violations():
        """Count total number of monochromatic C4s."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    k = bin(common).bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total
    
    current_violations = count_violations()
    if current_violations == 0:
        return {'n': n, 'colors': colors}
    
    # Simulated Annealing with incremental updates
    T = 5.0
    cooling = 0.99995
    min_T = 0.01
    start_sa = time.time()
    max_time = time_limit - (start_sa - time.time())
    
    best_violations = current_violations
    best_colors = colors[:]
    best_neighbors = [row[:] for row in neighbors]
    
    iterations = 0
    max_iterations = 5000000
    
    while time.time() < start_sa + max_time and iterations < max_iterations:
        iterations += 1
        
        # Pick a random edge
        e_idx = random.randint(0, num_edges - 1)
        u, v = edges[e_idx]
        old_color = colors[e_idx]
        
        # Try changing to a different color
        new_color = (old_color + random.randint(1, 4)) % 5
        
        # Calculate delta in violations using incremental update
        # Remove u from v's neighbors in old_color, add to new_color
        # The change affects violations for all pairs involving u or v in both colors
        
        # For efficiency, compute delta by checking affected pairs
        delta = 0
        
        # Pairs involving u in old_color (remove edge u-v)
        # For each w != u, v: pair (u, w) loses a common neighbor v in old_color
        # Pair (u, v) itself: they were connected in old_color, now not
        # Actually, removing edge (u,v) from old_color:
        # - For pair (u, v): common neighbors in old_color lose v? No, u and v are the pair
        #   common(c, u, v) = neighbors[c][u] & neighbors[c][v]. Removing edge u-v doesn't change
        #   neighbors[c][u] or neighbors[c][v] directly... wait, it does.
        #   neighbors[old_color][u] had bit v set, now it won't.
        #   So for pair (u, w) where w != v: common(old_color, u, w) changes because
        #   neighbors[old_color][u] lost bit v. So if w had v in its neighbors in old_color,
        #   then common lost v.
        # This is getting complex. Let me just do the simple O(n^2) recount but optimize it.
        
        # Actually, let me use a smarter approach: track per-vertex violation contributions
        
        # Simpler: just update and recount, but use bit_count which is fast
        # For n=26, that's 5 * 325 = 1625 bit operations per iteration
        # Should be fast enough in C-level bit operations
        
        # Update neighbors
        neighbors[old_color][u] &= ~(1 << v)
        neighbors[old_color][v] &= ~(1 << u)
        neighbors[new_color][u] |= (1 << v)
        neighbors[new_color][v] |= (1 << u)
        colors[e_idx] = new_color
        
        new_violations = count_violations()
        
        # Acceptance criterion
        diff = new_violations - current_violations
        if diff < 0 or random.random() < math.exp(-diff / T):
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
    
    # If we didn't find a valid coloring, return best if it's valid
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    
    return None
