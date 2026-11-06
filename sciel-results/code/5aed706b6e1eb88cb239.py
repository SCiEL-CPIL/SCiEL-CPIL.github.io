import time
import random
from itertools import combinations

def is_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False # Empty hypergraph is 2-colorable
    
    # Pre-process edges to lists of indices for faster access
    edge_list = [list(e) for e in edges]
    
    # State
    color = [-1] * n
    # active_edges: edges that are not yet monochromatic
    # We can just iterate all edges each time for n=9, E=126. It's fast enough.
    
    def propagate():
        # Returns False if contradiction, True if consistent
        # Also assigns forced colors
        changed = True
        while changed:
            changed = False
            for e in edge_list:
                c0 = 0
                c1 = 0
                unassigned = -1
                for v in e:
                    if color[v] == 0: c0 += 1
                    elif color[v] == 1: c1 += 1
                    else: unassigned = v
                
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and unassigned != -1:
                    if color[unassigned] == -1:
                        color[unassigned] = 1
                        changed = True
                    elif color[unassigned] == 0:
                        return False
                if c1 == 4 and unassigned != -1:
                    if color[unassigned] == -1:
                        color[unassigned] = 0
                        changed = True
                    elif color[unassigned] == 1:
                        return False
        return True

    def backtrack():
        # Find unassigned vertex
        v = -1
        for i in range(n):
            if color[i] == -1:
                v = i
                break
        
        if v == -1:
            return True # All assigned and consistent
        
        # Try color 0
        color[v] = 0
        if propagate():
            if backtrack():
                return True
        color[v] = -1
        
        # Try color 1
        color[v] = 1
        if propagate():
            if backtrack():
                return True
        color[v] = -1
        
        return False

    # Initial propagation
    if not propagate():
        return True # Contradiction immediately? No, just means no assignment works? 
                    # Actually if propagate fails, it means current partial assignment is bad.
                    # But we start with empty assignment, so it should pass.
                    
    if backtrack():
        return False # 2-coloring found
    else:
        return True # No 2-coloring

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Start with complete 5-uniform hypergraph on 9 vertices
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    best_edges = all_edges
    best_count = len(all_edges)
    
    # Try multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        current_edges = set(map(tuple, all_edges))
        
        # Shuffle edges to remove
        edges_to_try = list(current_edges)
        random.shuffle(edges_to_try)
        
        for edge in edges_to_try:
            if time.time() > deadline:
                break
            
            # Try removing edge
            current_edges.discard(edge)
            
            # Check if still non-2-colorable
            if is_non_2_colorable(n, list(current_edges)):
                # Keep it removed
                continue
            else:
                # Restore edge
                current_edges.add(edge)
        
        # If we have fewer edges, update best
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
            # If we hit the theoretical lower bound or a very good number, we could stop
            # But let's keep trying to see if we can go lower.
            
    return {'n': n, 'edges': [list(e) for e in best_edges]}
