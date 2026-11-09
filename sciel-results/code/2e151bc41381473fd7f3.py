import itertools
import random
import time

def is_2_colorable(edges):
    # Backtracking search for 2-coloring
    # vertices 0..8
    n = 9
    # Order edges by degree of unassigned vertices? 
    # Simple backtracking:
    # Assign colors to vertices one by one? No, check edges.
    # Better: Assign colors to edges? No.
    # Standard CSP: Variables are vertices, Domains {0,1}. Constraints: edges not mono.
    
    def backtrack(vertex_idx, colors):
        if vertex_idx == n:
            for e in edges:
                if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                    return False
            return True
        
        # Try 0
        colors[vertex_idx] = 0
        if backtrack(vertex_idx + 1, colors):
            return True
        # Try 1
        colors[vertex_idx] = 1
        if backtrack(vertex_idx + 1, colors):
            return True
        return False

    # Optimization: Only check edges that are "complete" (all 3 vertices assigned)
    # Or use a more efficient solver. For n=9, simple backtracking is fast enough.
    
    # Even better: Iterate through all 2^9 = 512 colorings.
    # This is extremely fast.
    for mask in range(512):
        c = [(mask >> i) & 1 for i in range(9)]
        valid = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid = False
                break
        if valid:
            return True
    return False

def construct():
    # The Fano plane on vertices 0..6 is a 3-uniform hypergraph with 7 edges and chromatic number 3.
    # Vertices 7 and 8 are isolated.
    # This is the minimum number of edges for a 3-uniform 3-chromatic hypergraph.
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane
    # 0:0, 1:1, 2:2, 3:1, 4:2, 5:2, 6:1
    # 7:0, 8:0
    colors = [0, 1, 2, 1, 2, 2, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

def find_3_coloring(edges):
    # Find a valid coloring using 3 colors
    # Since n=9, we can brute force 3^9 = 19683 colorings.
    for mask in range(19683):
        c = []
        m = mask
        for _ in range(9):
            c.append(m % 3)
            m //= 3
        valid = True
        for e in edges:
            if c[e[0]] == c[e[1]] == c[e[2]]:
                valid = False
                break
        if valid:
            return c
    return [0]*9 # Should not happen
