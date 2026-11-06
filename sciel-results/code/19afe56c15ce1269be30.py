"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def _has_c6_in_color(colors, n, color):
    """Check if a specific color class contains a 6-cycle (not necessarily induced)."""
    # Build adjacency matrix for this color
    # We need to check if there's a cycle of length 6 in the graph formed by edges of this color
    
    # For efficiency, let's build adjacency lists
    adj = [[] for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if colors[idx] == color:
                adj[i].append(j)
                adj[j].append(i)
            idx += 1
    
    # Now check for 6-cycles. A 6-cycle is v0-v1-v2-v3-v4-v5-v0 where all are distinct.
    # We can do this by trying all 6-tuples of distinct vertices and checking if they form a cycle.
    # But that's O(n^6) which is too slow for n=17+.
    
    # Better approach: use the fact that a 6-cycle in a graph can be found by checking
    # if there exist vertices a,b,c,d,e,f such that ab, bc, cd, de, ef, fa are all edges.
    # 
    # Alternative: For each pair of non-adjacent vertices, check if they have a common neighbor
    # that forms part of a 6-cycle. Actually, let's use a different approach.
    #
    # A graph contains a 6-cycle if and only if there exist three vertices a, b, c (not all the same)
    # such that... hmm, this is getting complicated.
    #
    # Let's use a simpler but potentially slower approach: for each 6-tuple of distinct vertices,
    # check if they form a cycle. For n=17, that's C(17,6) * 5! / 2 which is large but we can prune.
    #
    # Actually, let's use a smarter approach: 
    # A 6-cycle exists iff there are vertices v0,...,v5 such that vi is adjacent to v(i+1) mod 6.
    # We can fix v0 and v1 (adjacent), then try to extend to a path of length 5 and close it.
    
    # For each edge (u, v), try to find a path u - x - y - z - w - v where all vertices are distinct
    # and u, x, y, z, w, v are all distinct.
    
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            # We want to find x, y, z, w such that:
            # u-x, x-y, y-z, z-w, w-v are edges, and u,v,x,y,z,w are all distinct
            for x in adj[u]:
                if x == v:
                    continue
                for y in adj[x]:
                    if y in (u, v, x):
                        continue
                    for z in adj[y]:
                        if z in (u, v, x, y):
                            continue
                        for w in adj[z]:
                            if w in (u, v, x, y, z):
                                continue
                            if w in adj[v]:
                                return True
    return False

def _valid_coloring(colors, n):
    """Check if the coloring is valid (no 6-cycle in any color class)."""
    for c in range(4):
        if _has_c6_in_color(colors, n, c):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with n=9 known good coloring
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to extend to larger n using local search / backtracking
    # Strategy: try n from 10 up to 17, using randomized local search with restarts
    
    for target_n in range(10, 18):
        if time.time() > deadline - 5:
            break
        
        found = False
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a random coloring
            colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Try to fix violations using local search
            # We'll use a simple approach: repeatedly find a 6-cycle in some color and change an edge
            for iteration in range(1000):
                if time.time() > deadline - 5:
                    break
                
                # Check if valid
                valid = True
                violating_color = -1
                for c in range(4):
                    if _has_c6_in_color(colors, target_n, c):
                        valid = False
                        violating_color = c
                        break
                
                if valid:
                    found = True
                    best_n = target_n
                    best_colors = colors[:]
                    break
                
                # If not valid, try to fix by changing a random edge in the violating color
                # Pick a random edge and change its color
                edge_idx = random.randint(0, len(colors) - 1)
                old_color = colors[edge_idx]
                new_color = random.choice([c for c in range(4) if c != old_color])
                colors[edge_idx] = new_color
            
            if found:
                break
        
        if found:
            # Try to extend further
            continue
        else:
            # Couldn't find coloring for this n, try next n
            continue
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
