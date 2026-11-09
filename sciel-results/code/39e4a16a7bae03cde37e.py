import time
import random
import itertools
import sys
from collections import defaultdict

# Known minimal or near-minimal non-2-colorable 5-uniform hypergraphs
# Format: (n, edges)
# We use a known small example if available, or rely on the greedy search.
# The seed returns the complete graph on 9 vertices (126 edges).
# We aim to find something with < 126 edges quickly.

def _check_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False if not.
    """
    # Map vertices to bitmasks for efficiency? 
    # For n <= 26, we can use integers or lists.
    # Let's use a simple recursive backtracking with propagation.
    
    # Preprocess edges: remove duplicates, ensure sorted
    unique_edges = set()
    for e in edges:
        if len(e) != 5:
            return True # Invalid edge, treat as colorable? No, task says 5-uniform.
        unique_edges.add(tuple(sorted(e)))
    
    if not unique_edges:
        return True
        
    # For small n and e, simple backtracking is fine.
    # But we need speed.
    # Let's use a bitmask representation for the edges if n is small enough?
    # n <= 26, so 2^n is too big for full state, but we can use bitmasks for edges.
    
    # Actually, for the feedback loop in greedy search, we need this to be VERY fast.
    # A 5-uniform hypergraph with ~50-100 edges on ~10-20 vertices.
    # Backtracking might be slow if not pruned well.
    
    # Alternative: Use a simple greedy coloring check? No, must be exact.
    # Let's implement a decent backtracking solver.
    
    # Order vertices by degree
    deg = [0] * n
    for e in unique_edges:
        for v in e:
            deg[v] += 1
            
    # Sort vertices by degree descending
    order = sorted(range(n), key=lambda v: -deg[v])
    
    # Map original vertex to position in order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    # Remap edges to positions
    remapped_edges = []
    for e in unique_edges:
        remapped_edges.append(tuple(pos[v] for v in e))
        
    # Sort edges by "difficulty" or just keep them
    # Unit propagation: if an edge has 4 vertices of same color, the 5th must be opposite.
    
    # We will use a simple recursive function.
    # State: colors of vertices 0..k-1 in 'order'.
    # colors: -1 unassigned, 0, 1 assigned.
    
    colors = [-1] * n
    
    def propagate(k):
        """
        Apply unit propagation to colors given assignments up to k.
        Returns False if conflict, True if consistent.
        Modifies colors in place.
        """
        changed = True
        while changed:
            changed = False
            for e in remapped_edges:
                # Count colors in this edge
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if v < k:
                        if colors[v] == 0:
                            c0 += 1
                        elif colors[v] == 1:
                            c1 += 1
                    else:
                        unassigned.append(v)
                        
                # If all assigned, check if monochromatic
                if not unassigned:
                    if c0 == 5 or c1 == 5:
                        return False
                else:
                    # If 4 are same color, the 5th must be opposite
                    if c0 == 4 and len(unassigned) == 1:
                        colors[unassigned[0]] = 1
                        changed = True
                    elif c1 == 4 and len(unassigned) == 1:
                        colors[unassigned[0]] = 0
                        changed = True
                    elif c0 == 5 or c1 == 5:
                        return False
        return True

    def backtrack(k):
        if k == n:
            # All assigned, check if any edge is monochromatic
            for e in remapped_edges:
                c0 = 0
                c1 = 0
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    else:
                        c1 += 1
                if c0 == 5 or c1 == 5:
                    return False
            return True
            
        # Try color 0
        colors[k] = 0
        if propagate(k+1):
            if backtrack(k+1):
                return True
        # Try color 1
        colors[k] = 1
        if propagate(k+1):
            if backtrack(k+1):
                return True
                
        colors[k] = -1
        return False

    return backtrack(0)

def _greedy_search(time_limit=100):
    """
    Greedy search for a non-2-colorable 5-uniform hypergraph with few edges.
    """
    start_time = time.time()
    best_edges = None
    best_n = None
    
    # Try different n values
    for n in range(9, 15):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with empty edges
        edges = []
        
        # Greedy addition of edges
        # We want to force non-2-colorability.
        # Randomly add edges, check if 2-colorable.
        # If it becomes non-2-colorable, we have a candidate.
        # To minimize edges, we should add edges that "break" colorings efficiently.
        
        # Strategy: Start with a random set of edges, grow it until non-2-colorable.
        # Or, start with a known structure?
        
        # Let's try a random walk approach.
        # Pick random edges, add them. Check 2-colorability.
        # If non-2-colorable, record it.
        # Try to prune? No, just find one.
        
        # To be efficient, we can use a heuristic:
        # Add edges that are "hard" to color?
        
        # Simple approach: Random sampling of edge sets.
        # For a fixed n, try to build a minimal non-2-colorable graph.
        
        # Let's try a different heuristic:
        # Start with a complete graph on a small subset?
        
        # Actually, let's just use a simple local search or random restarts.
        # Given the time limit, we can afford a few thousand checks.
        
        # Let's try constructing a graph that contains a known non-2-colorable core.
        # The smallest known non-2-colorable 5-uniform hypergraph has 51 edges (per task description).
        # We don't have that hardcoded.
        
        # Let's try to find one with random edges.
        # For n=10, C(10,5) = 252 edges.
        # For n=11, C(11,5) = 462 edges.
        
        # We will run a loop:
        # 1. Pick random edges until non-2-colorable.
        # 2. Record the number of edges.
        # 3. Try to reduce n?
        
        # Better: Use a "maximal" random set and then try to remove edges?
        # No, adding is easier.
        
        # Let's try a few random instances for each n.
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
                
            current_edges = []
            # Pick a random subset of all possible edges
            all_edges = list(itertools.combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Add edges one by one
            for e in all_edges:
                current_edges.append(e)
                # Check 2-colorability
                # This is the bottleneck.
                if not _check_2_colorable(n, current_edges):
                    # Found a non-2-colorable graph
                    if best_edges is None or len(current_edges) < len(best_edges):
                        best_edges = [list(e) for e in current_edges]
                        best_n = n
                    break
                    
    return {'n': best_n, 'edges': best_edges} if best_edges else None

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 140  # 10s margin
    
    # 1. Try to return a known small non-2-colorable hypergraph if we have one.
    # We don't have one hardcoded in this code, so we rely on search.
    # However, the seed returns the complete graph on 9 vertices (126 edges).
    # If we find anything with < 126 edges, it's an improvement.
    
    result = None
    
    # 2. Run greedy search
    try:
        result = _greedy_search(time_limit=130)
    except Exception as e:
        pass
        
    # 3. If search failed or didn't find anything, fall back to seed
    if result is None or len(result['edges']) > 126:
        # Fall back to complete graph on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in itertools.combinations(range(9), 5)]}
        
    return result
