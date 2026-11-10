import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Fallback solution: n=11
    def get_n11():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    # Try to find n=26
    n_target = 26
    # Initialize with random colors
    colors = [random.randint(0, 4) for _ in range(n_target * (n_target - 1) // 2)]
    
    # Helper to get/set color
    def idx(i, j):
        if i > j: i, j = j, i
        return i * (n_target - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    def count_c4s():
        # Count monochromatic C4s
        # A C4 is a set of 4 vertices {a,b,c,d} and a color c such that
        # (a,b), (b,c), (c,d), (d,a) are all color c.
        # Or more efficiently, for each pair of vertices (u,v) with color c,
        # count common neighbors w with (u,w) and (v,w) also color c.
        # If count > 0, there is a C4.
        # We want to minimize the sum of such counts.
        
        # This is O(n^4) if done naively. n=26, 26^4 ~ 456k.
        # We can optimize by iterating over pairs and checking neighbors.
        # For each color c, build adjacency list.
        # For each pair (u,v) in color c, check if they have common neighbors in color c.
        
        total_c4s = 0
        # Build adjacency lists for each color
        adj = [[] for _ in range(5)]
        for i in range(n_target):
            for j in range(i + 1, n_target):
                c = colors[idx(i, j)]
                adj[c].append((i, j))
        
        # For each color, check for C4s
        for c in range(5):
            # Map vertex to set of neighbors in this color
            # Using sets for O(1) intersection check
            neigh = [set() for _ in range(n_target)]
            for i, j in adj[c]:
                neigh[i].add(j)
                neigh[j].add(i)
            
            # For each pair (u, v) in adj[c], check if len(neigh[u] & neigh[v]) > 0
            # To avoid double counting, we can just sum the sizes of intersections.
            # Each C4 is counted multiple times?
            # A C4 (a,b,c,d) has edges (a,b), (b,c), (c,d), (d,a).
            # Pairs with common neighbors:
            # (a,c) have common neighbors b, d. (2)
            # (b,d) have common neighbors a, c. (2)
            # (a,b) have no common neighbors in the C4?
            # Wait, a C4 is a cycle of length 4.
            # Vertices a,b,c,d. Edges ab, bc, cd, da.
            # Common neighbors of a and c: b and d.
            # Common neighbors of b and d: a and c.
            # Common neighbors of a and b: none in the cycle.
            # So each C4 contributes 2 to the sum of intersection sizes for opposite pairs.
            # We just need to detect if any intersection is non-empty.
            
            for u in range(n_target):
                for v in range(u + 1, n_target):
                    if v in neigh[u]: # Only check edges in this color
                        if neigh[u] & neigh[v]:
                            total_c4s += 1
                            # Early exit if we just want to know if valid
                            # But we need a score for search.
        
        return total_c4s

    # Local Search
    best_colors = colors[:]
    best_score = count_c4s()
    
    # If random init is already 0, return
    if best_score == 0:
        return {'n': n_target, 'colors': best_colors}

    while time.time() < deadline:
        # Pick a random edge to change
        e_idx = random.randint(0, len(colors) - 1)
        old_c = colors[e_idx]
        new_c = random.randint(0, 4)
        if old_c == new_c: continue
        
        colors[e_idx] = new_c
        new_score = count_c4s()
        
        if new_score < best_score:
            best_score = new_score
            best_colors = colors[:]
            if best_score == 0:
                return {'n': n_target, 'colors': best_colors}
        else:
            # Revert
            colors[e_idx] = old_c
            
        # If we get stuck, maybe try a random restart?
        # For now, simple hill climbing.
        
    # If no improvement, try to return best found
    if best_score == 0:
        return {'n': n_target, 'colors': best_colors}
    
    # Fallback
    return get_n11()
