"""Improved program for R_4(C6) - 4-colouring of K_n without monochromatic C6."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def has_c6_in_colour(edges, n, colour):
    """Check if there's a 6-cycle in the given colour class."""
    # Build adjacency for this colour
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        if colour == 0:
            adj[u].add(v)
            adj[v].add(u)
    
    # For each pair of non-adjacent vertices, check if they can form a C6
    # A C6: v0-v1-v2-v3-v4-v5-v0 where all edges are in the colour
    # We can check by looking for paths of length 5 between non-adjacent vertices
    for v0 in range(n):
        for v1 in adj[v0]:
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                for v3 in adj[v2]:
                    if v3 == v0 or v3 == v1:
                        continue
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2, v3):
                            continue
                        # Check if v4-v5-v0 forms the rest, i.e., v4 adjacent to some v5 adjacent to v0
                        # v5 must be adjacent to both v4 and v0, and v5 != v0,v1,v2,v3,v4
                        for v5 in adj[v4]:
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if v0 in adj[v5]:
                                return True
    return False

def is_valid_colouring(colors, n):
    """Check if a 4-colouring of K_n has no monochromatic C6."""
    # Build edge list per colour
    edges = [[], [], [], []]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            edges[c].append((i, j))
            idx += 1
    
    for c in range(4):
        if has_c6_in_colour(edges[c], n, c):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # Try to find a valid colouring for increasing n
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Use randomised search with backtracking
    for n in range(10, 21):
        if time.time() > deadline:
            break
        
        num_edges = n * (n - 1) // 2
        found = False
        
        # Try multiple random restarts
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initial colouring
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Try to fix violations using local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Check validity
                if is_valid_colouring(colors, n):
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break
                
                # Find a violating colour class and try to fix
                edges = [[], [], [], []]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = colors[idx]
                        edges[c].append((i, j))
                        idx += 1
                
                # Try changing random edges to eliminate C6s
                for c in range(4):
                    if has_c6_in_colour(edges[c], n, c):
                        # Pick a random edge in this colour and change it
                        if edges[c]:
                            ei = random.randint(0, len(edges[c]) - 1)
                            u, v = edges[c][ei]
                            # Find index in colors
                            idx2 = 0
                            for i2 in range(n):
                                for j2 in range(i2 + 1, n):
                                    if i2 == u and j2 == v:
                                        colors[idx2] = (c + random.randint(1, 3)) % 4
                                        improved = True
                                        break
                                    idx2 += 1
                                if improved:
                                    break
                            if improved:
                                break
                
                if time.time() > deadline:
                    break
        
        if found:
            continue  # Try larger n
        else:
            break  # Can't find valid for this n, stop
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
