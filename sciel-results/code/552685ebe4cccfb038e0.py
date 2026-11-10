<<<<<<< SEARCH
"""
5-uniform hypergraph without Property B (m(5)) - combined construction.

Strategy:
  1. Use a fast, array-based 2-colourability checker with unit propagation.
  2. Greedy edge removal from a complete hypergraph on small n.
  3. Local search: random restarts, add edges to break 2-colourability, then minimize.
  4. Hard-coded fallback if search fails.
"""

import time
import random
import itertools
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph on n vertices is 2-colorable (has Property B).
    Uses backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Precompute edge list as tuples of ints
    edge_list = [tuple(e) for e in edges]
    m = len(edge_list)
    
    # For speed, we'll use a list of lists for counts per edge
    # But for small n and m, simple iteration is fine.
    
    # colors: -1 unassigned, 0 or 1 assigned
    colors = [-1] * n
    
    # Precompute: for each vertex, list of edge indices containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_list):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate():
        """
        Apply unit propagation.
        Returns False if contradiction (monochromatic edge), True otherwise.
        Modifies colors in place.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                e = edge_list[i]
                c0 = 0
                c1 = 0
                unassigned = -1
                uncount = 0
                for v in e:
                    c = colors[v]
                    if c == 0:
                        c0 += 1
                    elif c == 1:
                        c1 += 1
                    else:
                        unassigned = v
                        uncount += 1
                
                if c0 == 5 or c1 == 5:
                    return False
                if uncount == 1:
                    if c0 == 4:
                        if colors[unassigned] == -1:
                            colors[unassigned] = 1
                            changed = True
                    elif c1 == 4:
                        if colors[unassigned] == -1:
                            colors[unassigned] = 0
                            changed = True
        return True
    
    def find_unassigned():
        """Find an unassigned vertex with the most constraints."""
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        return best_v
    
    def backtrack():
        """Try to complete the assignment. Returns True if 2-coloring exists."""
        if not propagate():
            return False
        
        v = find_unassigned()
        if v == -1:
            return True
        
        # Try color 0
        colors[v] = 0
        if backtrack():
            return True
        
        # Try color 1
        colors[v] = 1
        if backtrack():
            return True
        
        colors[v] = -1
        return False
    
    return backtrack()
=======
"""
5-uniform hypergraph without Property B (m(5)) - brute force verification.

Strategy:
  1. Use a brute-force 2-colourability checker (iterate all 2^n colorings).
     This is exact and fast for n <= 16.
  2. Random search for a non-2-colorable subset of edges.
  3. Greedy edge removal to minimize edge count, verified by brute force.
"""

import time
import random
import itertools
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph on n vertices is 2-colorable.
    Brute force: iterate all 2^n colorings.
    Returns True if 2-colorable (there exists a coloring with no monochromatic edge).
    Returns False if non-2-colorable (every coloring has a monochromatic edge).
    """
    if not edges:
        return True
    
    edge_list = [tuple(e) for e in edges]
    m = len(edge_list)
    
    # For each coloring mask from 0 to 2^n - 1
    for mask in range(1 << n):
        # Check if this coloring is valid (no monochromatic edge)
        valid = True
        for e in edge_list:
            # Count colors in this edge
            c0 = 0
            c1 = 0
            for v in e:
                if (mask >> v) & 1:
                    c1 += 1
                else:
                    c0 += 1
            if c0 == 5 or c1 == 5:
                valid = False
                break
        if valid:
            # Found a valid coloring
            return True
            
    # No valid coloring found
    return False
>>>>>>> REPLACE

