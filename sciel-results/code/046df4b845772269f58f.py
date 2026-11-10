"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_2_colorable(n, edges):
    """Check if 5-uniform hypergraph is 2-colorable using backtracking with unit propagation."""
    if not edges:
        return True
    
    # Build incidence: for each vertex, list of edges containing it
    # For unit propagation: if an edge has 4 vertices same color, the 5th must be different
    
    color = [-1] * n  # -1 unassigned, 0/1 colors
    
    # Edge index for quick lookup
    edge_set = set()
    for e in edges:
        edge_set.add(tuple(sorted(e)))
    
    # For each edge, track which vertices are assigned and their colors
    # Unit propagation: if 4 vertices in an edge have same color, 5th must be opposite
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for edge in edges:
                colors = [color[v] for v in edge]
                # Count colors
                c0 = colors.count(0)
                c1 = colors.count(1)
                unassigned = colors.count(-1)
                
                # If 4 are same color, 5th must be opposite
                if c0 == 4 and unassigned == 1:
                    for v in edge:
                        if color[v] == -1:
                            color[v] = 1
                            changed = True
                elif c1 == 4 and unassigned == 1:
                    for v in edge:
                        if color[v] == -1:
                            color[v] = 0
                            changed = True
                
                # Check for conflict: all 5 same color
                if c0 == 5 or c1 == 5:
                    return False
        return True
    
    def backtrack(idx):
        if idx >= n:
            return True
        
        # Try each color
        for c in [0, 1]:
            color[idx] = c
            if propagate():
                if backtrack(idx + 1):
                    return True
            color[idx] = -1
        
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Start with complete hypergraph on 8 vertices
    n = 8
    all_edges = [list(e) for e in combinations(range(n), 5)]
    edges = all_edges[:]
    
    # Greedily remove edges while maintaining non-2-colorability
    # Try removing each edge in order
    i = 0
    while i < len(edges) and time.time() < deadline:
        # Try removing edge i
        edges_without = edges[:i] + edges[i+1:]
        if not _is_2_colorable(n, edges_without):
            # Removing this edge still keeps it non-2-colorable
            edges = edges_without
            # Don't increment i, try next edge
        else:
            # Need this edge
            i += 1
    
    # Verify result
    if not _is_2_colorable(n, edges):
        return {'n': n, 'edges': edges}
    
    # Fallback: complete hypergraph on 9 vertices
    n9 = 9
    return {'n': n9, 'edges': [list(e) for e in combinations(range(n9), 5)]}
# EVOLVE-BLOCK-END
