import random
import time
from itertools import combinations
import math

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable.
    Returns True if a proper 2-coloring exists, False otherwise.
    """
    if not edges:
        return True
        
    # Convert edges to tuples for faster iteration
    edge_tuples = [tuple(e) for e in edges]
    
    color = [-1] * n
    
    def propagate():
        """
        Performs unit propagation.
        Returns False if a contradiction is found.
        """
        changed = True
        while changed:
            changed = False
            for e in edge_tuples:
                c0 = 0
                c1 = 0
                unassigned = -1
                for v in e:
                    c = color[v]
                    if c == -1:
                        unassigned = v
                    elif c == 0:
                        c0 += 1
                    elif c == 1:
                        c1 += 1
                
                # Check for conflict: all assigned and monochromatic
                if unassigned == -1:
                    if c0 == 5 or c1 == 5:
                        return False
                else:
                    # Propagate
                    if c0 == 4:
                        color[unassigned] = 1
                        changed = True
                    elif c1 == 4:
                        color[unassigned] = 0
                        changed = True
        return True

    def backtrack(idx):
        if idx == n:
            # All vertices assigned. Check if any edge is monochromatic.
            # Actually, propagation should have caught conflicts.
            # But let's double check.
            for e in edge_tuples:
                c0 = 0
                c1 = 0
                for v in e:
                    if color[v] == 0:
                        c0 += 1
                    elif color[v] == 1:
                        c1 += 1
                if c0 == 5 or c1 == 5:
                    return False
            return True
        
        # Heuristic: Try 0 then 1
        color[idx] = 0
        if propagate():
            if backtrack(idx + 1):
                return True
        color[idx] = -1
        
        color[idx] = 1
        if propagate():
            if backtrack(idx + 1):
                return True
        color[idx] = -1
        
        return False

    return backtrack(0)

def construct():
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    best_edges = None
    best_n = None
    best_m = 1000
    
    # Search for n from 9 to 18
    # Search for m from 29 to best_m
    for n in range(9, 19):
        if time.time() - start_time > time_limit:
            break
            
        # Max possible edges
        try:
            max_possible = math.comb(n, 5)
        except:
            max_possible = 126
            
        # We only care about m < best_m
        # And m >= 29
        upper_m = min(best_m - 1, max_possible)
        
        for m in range(29, upper_m + 1):
            if time.time() - start_time > time_limit:
                break
                
            # Try to find a non-2-colorable graph with n vertices and m edges
            # Number of trials depends on time remaining
            # Start with 50 trials, increase if time permits
            trials = 50
            found = False
            
            all_combos = list(combinations(range(n), 5))
            if len(all_combos) < m:
                continue
                
            for _ in range(trials):
                if time.time() - start_time > time_limit:
                    break
                    
                edges = random.sample(all_combos, m)
                edges = [list(e) for e in edges]
                
                if not is_2_colorable(n, edges):
                    best_m = m
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            # If found, we don't need to try larger m for this n
            # because we want to minimize m.
            if found:
                # Continue to next n to see if a smaller m exists there
                pass
        
        # End m loop
        
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to seed
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
