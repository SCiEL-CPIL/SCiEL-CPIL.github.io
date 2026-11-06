import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target n=26 based on Lazebnik-Woldar
    target_n = 26
    
    # We will try to find a valid coloring for n=26
    # If not found, we fall back to smaller n
    
    best_n = 11
    best_colors = None
    
    # Simple seed for n=11 (from original)
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_n = n_seed
    best_colors = colors_seed
    
    # Local Search for n=26
    n = target_n
    num_edges = n * (n - 1) // 2
    
    # Initialize colors randomly
    colors = np.random.randint(0, 5, size=num_edges)
    
    # Precompute edge indices
    edge_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to get color of edge (u, v)
    def get_color(u, v):
        if u > v: u, v = v, u
        # Calculate index
        # Index of (u, v) in lexicographic order
        # Number of edges with first vertex < u: u * n - u * (u - 1) / 2
        # Actually, simpler:
        # idx = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
        idx = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
        return colors[idx]
    
    def set_color(u, v, c):
        if u > v: u, v = v, u
        idx = u * (n - 1) - u * (u + 1) // 2 + (v - u - 1)
        colors[idx] = c
    
    # Count violations: number of pairs of vertices with >= 2 common neighbors in same color
    # This is expensive if done naively.
    # Optimization: For each color, count common neighbors.
    
    def count_violations():
        violations = 0
        for c in range(5):
            # For each pair of vertices (u, v), count common neighbors in color c
            # This is O(n^3 * 5) which is 26^3 * 5 ~ 90k operations. Very fast.
            for u in range(n):
                for v in range(u + 1, n):
                    common = 0
                    for w in range(n):
                        if w == u or w == v: continue
                        if get_color(u, w) == c and get_color(v, w) == c:
                            common += 1
                            if common >= 2:
                                violations += 1
                                break
        return violations
    
    # Initial check
    if count_violations() == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # Local Search
    # We want to minimize violations.
    # If violations > 0, pick a violating pair (u, v) and color c, and change color of one of their common neighbors' edges.
    
    while time.time() < deadline:
        violations = count_violations()
        if violations == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Find a violating pair
        # We need to find u, v, c such that they have >= 2 common neighbors in color c.
        # Then pick one of the common neighbors w, and change the color of edge (u, w) or (v, w).
        
        found_violation = False
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common_neighbors = []
                    for w in range(n):
                        if w == u or w == v: continue
                        if get_color(u, w) == c and get_color(v, w) == c:
                            common_neighbors.append(w)
                    if len(common_neighbors) >= 2:
                        found_violation = True
                        # Pick a random common neighbor w
                        w = random.choice(common_neighbors)
                        # Change color of edge (u, w) or (v, w)
                        if random.random() < 0.5:
                            old_c = get_color(u, w)
                            # Try all other colors
                            for new_c in range(5):
                                if new_c == old_c: continue
                                set_color(u, w, new_c)
                                new_violations = count_violations()
                                if new_violations < violations:
                                    break
                                else:
                                    set_color(u, w, old_c)
                            # If no improvement, accept a random change to escape local min?
                            # Or just accept the first change if it doesn't increase violations?
                            # Let's use a simple hill climbing: accept if violations decrease.
                            # If stuck, do a random move.
                        else:
                            old_c = get_color(v, w)
                            for new_c in range(5):
                                if new_c == old_c: continue
                                set_color(v, w, new_c)
                                new_violations = count_violations()
                                if new_violations < violations:
                                    break
                                else:
                                    set_color(v, w, old_c)
                        break
                if found_violation: break
            if found_violation: break
        
        if not found_violation:
            # This should not happen if violations > 0, but just in case
            break

    # If we didn't find a valid coloring for n=26, try n=25, etc?
    # Or just return the best valid one we had (n=11).
    # But the local search might have reduced violations to 0 for n=26.
    
    # Re-check final state
    if count_violations() == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # Fallback to seed
    return {'n': best_n, 'colors': best_colors}
