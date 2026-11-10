import time
import random
from itertools import combinations

def is_2_colorable(n, edges, timeout=10):
    """Check if the hypergraph is 2-colourable using backtracking with unit propagation."""
    start_time = time.time()
    
    # Build adjacency: for each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
    
    # Unit propagation: if an edge has 4 vertices of the same colour, the 5th must be different
    # We'll use a simple backtracking with propagation
    
    def propagate(coloring):
        """Apply unit propagation. Returns None if contradiction, else updated coloring."""
        changed = True
        while changed:
            changed = False
            for edge in edges:
                # Count colours in this edge
                counts = [0, 0]
                unknown = -1
                for v in edge:
                    if coloring[v] == -1:
                        unknown = v
                    else:
                        counts[coloring[v]] += 1
                # If 4 of one colour, the 5th must be the other
                if counts[0] == 4 and unknown != -1 and coloring[unknown] == -1:
                    coloring[unknown] = 1
                    changed = True
                elif counts[1] == 4 and unknown != -1 and coloring[unknown] == -1:
                    coloring[unknown] = 0
                    changed = True
        # Check for contradictions
        for edge in edges:
            if all(coloring[v] == 0 for v in edge) or all(coloring[v] == 1 for v in edge):
                return None
        return coloring
    
    def backtrack(coloring):
        if time.time() - start_time > timeout:
            return None  # Timeout, treat as "cannot prove not 2-colourable"
        
        coloring = propagate(coloring[:])
        if coloring is None:
            return None  # Contradiction, this branch fails
        
        # Find unassigned vertex
        unassigned = [v for v in range(n) if coloring[v] == -1]
        if not unassigned:
            return coloring  # Found a valid 2-colouring
        
        # Choose vertex with most constraints
        v = max(unassigned, key=lambda x: len(vertex_edges[x]))
        
        for c in [0, 1]:
            coloring[v] = c
            result = backtrack(coloring)
            if result is not None:
                return result
            coloring[v] = -1
        
        return None
    
    coloring = [-1] * n
    result = backtrack(coloring)
    return result is not None  # True if 2-colourable


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Try to find a small non-2-colourable 5-uniform hypergraph
    # Start with known structures and search
    
    best = None
    
    # First, try the complete 5-uniform hypergraph on 9 vertices as a fallback
    fallback_edges = [list(e) for e in combinations(range(9), 5)]
    best = {'n': 9, 'edges': fallback_edges}
    
    # Try to find smaller constructions
    # Use random search with increasing edge counts
    
    for n in [9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline:
            break
        
        for m in range(30, 60):  # Try edge counts from 30 to 59
            if time.time() > deadline:
                break
            
            # Try a few random hypergraphs with m edges
            for trial in range(5):
                if time.time() > deadline:
                    break
                
                # Generate random 5-uniform hypergraph
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:m]]
                
                # Check if it's non-2-colourable
                if not is_2_colorable(n, edges, timeout=5):
                    if best is None or m < len(best['edges']):
                        best = {'n': n, 'edges': edges}
                        break  # Found one with this m, move to next m
    
    return best
