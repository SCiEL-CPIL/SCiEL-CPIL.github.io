import time
import random
from itertools import combinations

def is_not_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Returns False if it IS 2-colorable.
    Uses backtracking with unit propagation.
    """
    # edges is a list of lists/tuples of 5 vertices
    
    def propagate(colors):
        # Returns False if contradiction, True if consistent
        # Modifies colors in place
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                        
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and len(unassigned) == 1:
                    u = unassigned[0]
                    if colors[u] != -1:
                        if colors[u] != 1:
                            return False
                    else:
                        colors[u] = 1
                        changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    u = unassigned[0]
                    if colors[u] != -1:
                        if colors[u] != 0:
                            return False
                    else:
                        colors[u] = 0
                        changed = True
        return True

    def backtrack(idx):
        if idx == n:
            return True
            
        v = idx
        if colors[v] != -1:
            return backtrack(v + 1)
            
        # Try 0
        colors[v] = 0
        if propagate(colors):
            if backtrack(v + 1):
                return True
        colors[v] = -1
        
        # Try 1
        colors[v] = 1
        if propagate(colors):
            if backtrack(v + 1):
                return True
        colors[v] = -1
        
        return False

    colors = [-1] * n
    if propagate(colors):
        if backtrack(0):
            return False
    return True

def construct():
    deadline = time.time() + 110
    best_edges = None
    best_count = float('inf')
    
    # Start with n=9
    n = 9
    all_edges = [list(e) for e in combinations(range(n), 5)]
    
    # Initial check: is it non-2-colorable?
    if not is_not_2_colorable(n, all_edges):
        # Should not happen for complete graph, but just in case
        return {'n': n, 'edges': all_edges}
        
    best_edges = list(all_edges)
    best_count = len(all_edges)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        current_edges = list(all_edges)
        random.shuffle(current_edges)
        
        i = 0
        while i < len(current_edges) and time.time() < deadline:
            e = current_edges[i]
            # Try to remove e
            # Create a new list without e
            # Optimization: Since we are checking repeatedly, 
            # and n is small, list comprehension is fine.
            remaining = [x for x in current_edges if x != e]
            
            if is_not_2_colorable(n, remaining):
                # Successful removal
                current_edges.pop(i)
                # Do not increment i
                if len(current_edges) < best_count:
                    best_count = len(current_edges)
                    best_edges = list(current_edges)
            else:
                i += 1
                
        if time.time() > deadline:
            break

    return {'n': n, 'edges': best_edges}
