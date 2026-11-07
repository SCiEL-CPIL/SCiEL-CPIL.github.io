"""Rewritten program: find a 3-chromatic 3-uniform hypergraph on 9 vertices with minimum edges."""
# EVOLVE-BLOCK-START

import random

def is_2_colorable(edges):
    """
    Returns True if the hypergraph defined by edges is 2-colorable.
    Returns False if it is 3-chromatic (not 2-colorable).
    """
    if not edges:
        return True
        
    # Get unique vertices involved
    vertices = set()
    for e in edges:
        vertices.update(e)
    vertices = sorted(vertices)
    n = len(vertices)
    
    # Map vertex to index
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Convert edges to indices
    edges_idx = [[v_to_idx[v] for v in e] for e in edges]
    
    # Check all 2^n colorings
    # n <= 9, so 2^9 = 512
    for mask in range(1 << n):
        valid = True
        # Assign colors
        # color[i] = (mask >> i) & 1
        for e in edges_idx:
            c0 = (mask >> e[0]) & 1
            c1 = (mask >> e[1]) & 1
            c2 = (mask >> e[2]) & 1
            if c0 == c1 == c2:
                valid = False
                break
        if valid:
            return True
            
    return False

def find_3_coloring(edges):
    """
    Finds a 3-coloring for the hypergraph defined by edges.
    Returns a dict {vertex: color} for vertices in edges.
    """
    if not edges:
        return {}
        
    vertices = set()
    for e in edges:
        vertices.update(e)
    vertices = sorted(vertices)
    n = len(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    edges_idx = [[v_to_idx[v] for v in e] for e in edges]
    
    # Try all 3^n colorings
    # n <= 9, 3^9 = 19683
    # Use base-3 representation
    for mask in range(3 ** n):
        valid = True
        temp = mask
        colors = [0] * n
        for i in range(n):
            colors[i] = temp % 3
            temp //= 3
            
        for e in edges_idx:
            if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                valid = False
                break
        if valid:
            coloring = {}
            for i, v in enumerate(vertices):
                coloring[v] = colors[i]
            return coloring
            
    raise ValueError("No 3-coloring found")

def construct():
    """
    Searches for a 3-chromatic 3-uniform hypergraph on vertices 0..8 with minimum edges.
    Tries 6 edges first, then 7.
    """
    all_possible_edges = []
    for i in range(9):
        for j in range(i+1, 9):
            for k in range(j+1, 9):
                all_possible_edges.append((i, j, k))
                
    # Strategy: Try m=6, then m=7
    for m in [6, 7]:
        # Randomized search
        attempts = 1000
        for _ in range(attempts):
            edges = random.sample(all_possible_edges, m)
            if not is_2_colorable(edges):
                # Found a 3-chromatic hypergraph
                coloring = find_3_coloring(edges)
                colors = [0] * 9
                for v, c in coloring.items():
                    colors[v] = c
                # Ensure types are int and list
                edges_list = [list(e) for e in edges]
                return {
                    'edges': edges_list,
                    'colors': colors
                }
        
        # If random search failed for m=6, it will likely fail for m=7 if we don't bias,
        # but we know m=7 has solutions.
        # To be safe and fast, if m=7, we can construct the Fano plane directly
        # if random search doesn't find one quickly.
        if m == 7:
            # Construct Fano plane on 0..6
            # This is a known 3-chromatic 3-uniform hypergraph
            fano_edges = [
                (0, 1, 2),
                (0, 3, 4),
                (0, 5, 6),
                (1, 3, 5),
                (1, 4, 6),
                (2, 3, 6),
                (2, 4, 5)
            ]
            # Verify it's 3-chromatic (should be)
            if not is_2_colorable(fano_edges):
                coloring = find_3_coloring(fano_edges)
                colors = [0] * 9
                for v, c in coloring.items():
                    colors[v] = c
                edges_list = [list(e) for e in fano_edges]
                return {
                    'edges': edges_list,
                    'colors': colors
            }
            
    # Fallback: Should never reach here
    raise Exception("Failed to find a 3-chromatic hypergraph")

# EVOLVE-BLOCK-END
