import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    # Map vertex to list of edges containing it
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
    
    # Edge active status
    active = [True] * len(edges)
    
    # Color: -1 uncolored, 0, 1
    color = [-1] * n
    
    def propagate(vertex, val):
        """
        Assigns color `val` to `vertex`.
        Returns False if conflict, True otherwise.
        Uses unit propagation: if an edge has 4 vertices of one color, the 5th must be the other.
        """
        stack = [(vertex, val)]
        while stack:
            v, c = stack.pop()
            if color[v] != -1:
                if color[v] != c:
                    return False
                else:
                    continue
            
            color[v] = c
            
            # Propagate to edges containing v
            for e_idx in vertex_to_edges[v]:
                if not active[e_idx]:
                    continue
                
                # Count colors in this edge
                counts = [0, 0]
                uncolored = []
                for u in edges[e_idx]:
                    if color[u] == 0:
                        counts[0] += 1
                    elif color[u] == 1:
                        counts[1] += 1
                    else:
                        uncolored.append(u)
                
                # If 4 are same color, 5th must be opposite
                if counts[0] == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 1))
                elif counts[1] == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 0))
                elif counts[0] == 5 or counts[1] == 5:
                    return False
        
        return True

    def backtrack(vertex):
        if vertex == n:
            return True
        
        # Try color 0
        if color[vertex] == -1:
            old_color = color[vertex]
            # We don't actually need to save/restore if we use a stack or copy, 
            # but for efficiency, we can just try and backtrack manually.
            # However, propagate modifies `color` and `active`? 
            # My propagate only modifies `color`. It doesn't mark edges inactive yet?
            # Actually, if an edge becomes monochromatic, we return False.
            # If an edge has 4 of one color, we force the 5th.
            
            # Let's use a simple recursive backtracking with state copying for simplicity/safety in small N
            # N is small (<= 26), edges <= 200.
            
            pass
        return False

    # Better backtracking implementation
    def solve(idx):
        if idx == n:
            return True
        
        if color[idx] != -1:
            return solve(idx + 1)
        
        for c in [0, 1]:
            # Save state
            saved_color = color[:]
            # Try assignment
            if propagate(idx, c):
                if solve(idx + 1):
                    return True
            # Restore
            color[:] = saved_color
        
        return False

    return solve(0)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = None
    best_count = 1000
    
    # Fallback: K9(5)
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    # Try to find a smaller one on n=11 or n=12
    # Strategy: Random edges, check, prune.
    
    for n in [11, 12]:
        if time.time() > deadline:
            break
            
        # Start with a random set of edges
        # Estimate needed edges: ~100-150
        for trial in range(50):
            if time.time() > deadline:
                break
            
            random.seed(trial)
            # Generate random edges
            all_possible = list(combinations(range(n), 5))
            random.shuffle(all_possible)
            
            # Add edges until non-2-colorable or max reached
            edges = []
            for e in all_possible:
                if len(edges) > 200:
                    break
                edges.append(list(e))
                if not is_2_colorable(n, edges):
                    break
            
            if is_2_colorable(n, edges):
                continue
            
            # Now prune
            # Try to remove each edge if possible
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for i in range(len(edges) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    # Remove edge i
                    candidate = edges[:i] + edges[i+1:]
                    if not is_2_colorable(n, candidate):
                        edges = candidate
                        changed = True
                        break # Restart loop as indices changed
            
            if len(edges) < best_count:
                best_count = len(edges)
                best_edges = edges
                best_n = n
                
        if best_count < 100: # Good enough, break
            break
            
    if best_edges is not None:
        return {'n': best_n, 'edges': best_edges}
    else:
        return fallback
