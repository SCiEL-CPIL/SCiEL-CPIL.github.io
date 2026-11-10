"""
5-uniform hypergraph without Property B (m(5)) - optimized construction.

Strategy:
  1. Hard-code a known small non-2-colorable 5-uniform hypergraph if available.
  2. Otherwise, perform a time-bounded search: start from complete hypergraphs
     on small n, greedily remove edges while maintaining non-2-colorability.
  3. Also try random sampling of edge subsets to find small non-2-colorable instances.
  4. Return the best (fewest edges) result found within the time budget.
"""

import time
import random
from itertools import combinations
from functools import lru_cache

def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph on n vertices is 2-colorable (has Property B).
    Uses backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Build adjacency: for each edge, which vertices are in it
    # We'll use a recursive backtracking approach
    
    # Convert edges to frozensets for fast lookup
    edge_set = [frozenset(e) for e in edges]
    
    # For unit propagation: if an edge has k-1 vertices of same color,
    # the last must be opposite.
    
    # State: assignment dict vertex -> 0 or 1, or None if unassigned
    assignment = {}
    
    # Precompute: for each vertex, which edges contain it
    vertex_edges = {v: [] for v in range(n)}
    for i, e in enumerate(edge_set):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate(assignment):
        """Apply unit propagation. Returns None if contradiction, else updated assignment."""
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_set):
                # Count colors in this edge
                colors = {}
                unassigned = []
                for v in e:
                    if v in assignment:
                        c = assignment[v]
                        colors[c] = colors.get(c, 0) + 1
                    else:
                        unassigned.append(v)
                
                if len(unassigned) == 0:
                    # Check if monochromatic
                    if len(colors) == 1:
                        return None  # contradiction
                elif len(unassigned) == 1:
                    # If 4 of one color, last must be opposite
                    for c, cnt in colors.items():
                        if cnt == 4:
                            v = unassigned[0]
                            if v in assignment:
                                if assignment[v] == c:
                                    return None
                            else:
                                assignment[v] = 1 - c
                                changed = True
                elif len(unassigned) == 2:
                    # If 3 of one color, the two unassigned must both be opposite? No.
                    # If 3 red, then we need at least one blue among the 2 unassigned.
                    # This doesn't force a specific assignment, just a constraint.
                    pass
        
        return assignment
    
    def backtrack(assignment):
        """Try to complete the assignment. Returns True if 2-coloring exists."""
        # Check for contradiction in current assignment
        for i, e in enumerate(edge_set):
            colors = {}
            unassigned_count = 0
            for v in e:
                if v in assignment:
                    c = assignment[v]
                    colors[c] = colors.get(c, 0) + 1
                else:
                    unassigned_count += 1
            
            if unassigned_count == 0:
                if len(colors) == 1:
                    return False  # monochromatic edge
            elif unassigned_count == 1:
                for c, cnt in colors.items():
                    if cnt == 4:
                        v = [x for x in e if x not in assignment][0]
                        if assignment.get(v) == c:
                            return False
        
        # Find an unassigned vertex
        unassigned = [v for v in range(n) if v not in assignment]
        if not unassigned:
            # All assigned, check all edges
            for i, e in enumerate(edge_set):
                colors = set(assignment[v] for v in e)
                if len(colors) == 1:
                    return False
            return True
        
        # Choose vertex with most constraints (in most edges)
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        # Try both colors
        for color in [0, 1]:
            new_assignment = dict(assignment)
            new_assignment[v] = color
            
            # Apply unit propagation
            propagated = propagate(new_assignment)
            if propagated is not None:
                if backtrack(propagated):
                    return True
        
        return False
    
    return backtrack(assignment)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second budget, leave 10s margin
    
    best_result = None
    best_edges = None
    
    # Try to find a small non-2-colorable hypergraph
    # Start with small n values
    
    for n in range(6, 13):
        if time.time() > deadline:
            break
        
        all_edges = list(combinations(range(n), 5))
        num_total = len(all_edges)
        
        if num_total > 200:
            continue
        
        # Try greedy removal: start with all edges, remove one at a time
        # if the hypergraph remains non-2-colorable
        
        # First check if complete hypergraph is non-2-colorable
        if _is_2_colorable(n, all_edges):
            continue
        
        # It's non-2-colorable. Now try to remove edges greedily.
        current_edges = list(all_edges)
        random.shuffle(current_edges)
        
        # Try removing edges in random order
        i = 0
        while i < len(current_edges) and time.time() < deadline:
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            if not _is_2_colorable(n, test_edges):
                # Still non-2-colorable, remove it
                current_edges = test_edges
                # Don't increment i since we removed an element
            else:
                i += 1
        
        if best_edges is None or len(current_edges) < len(best_edges):
            best_edges = current_edges
            best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
        
        # Also try random subset approach
        if time.time() > deadline:
            break
        
        # Try various subset sizes
        for size in range(max(5, len(current_edges) - 10), len(current_edges) + 1):
            if time.time() > deadline:
                break
            # Try a few random subsets of this size
            for attempt in range(5):
                if time.time() > deadline:
                    break
                random.shuffle(current_edges)
                subset = current_edges[:size]
                if not _is_2_colorable(n, subset):
                    if best_edges is None or len(subset) < len(best_edges):
                        best_edges = subset
                        best_result = {'n': n, 'edges': [list(e) for e in subset]}
                    break  # Found one of this size, move to smaller
    
    # If no result found, fall back to complete hypergraph on 9 vertices
    if best_result is None:
        all_edges = list(combinations(range(9), 5))
        best_result = {'n': 9, 'edges': [list(e) for e in all_edges]}
    
    return best_result
