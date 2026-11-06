import random
from itertools import combinations
import time

def is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable.
    Returns True if a proper 2-coloring exists (no monochromatic edge).
    Returns False otherwise.
    Uses backtracking with unit propagation.
    """
    # Pre-process edges for faster access
    # For each vertex, list of edges containing it
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # State: colors[i] is -1 (unassigned), 0, or 1
    colors = [-1] * n
    
    # Queue for unit propagation
    # We need to check if adding a color causes a conflict or forces another color.
    # An edge is 'critical' if it has k vertices of one color and 0 of the other,
    # and k+1 < 5. If k=4, the last one is forced.
    # If k=5, conflict (monochromatic edge found, so this branch is invalid for PROPER coloring).
    
    def propagate(start_v, color):
        # BFS/DFS from start_v
        queue = [(start_v, color)]
        visited = set()
        while queue:
            v, c = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            
            if colors[v] != -1:
                if colors[v] != c:
                    return False # Conflict
                else:
                    continue
            
            colors[v] = c
            
            # Check all edges containing v
            for e_idx in vertex_edges[v]:
                e = edges[e_idx]
                # Count colors in e
                count_0 = 0
                count_1 = 0
                unassigned = None
                for u in e:
                    if colors[u] == 0:
                        count_0 += 1
                    elif colors[u] == 1:
                        count_1 += 1
                    else:
                        unassigned = u
                
                # If we have 5 of one color, it's a conflict (monochromatic edge)
                if count_0 == 5 or count_1 == 5:
                    return False
                
                # If we have 4 of one color, the 5th is forced to the other
                if count_0 == 4:
                    if unassigned is not None:
                        if colors[unassigned] == 1:
                            pass # Already consistent
                        elif colors[unassigned] == -1:
                            queue.append((unassigned, 1))
                        else: # colors[unassigned] == 0? Impossible as count_0==4
                            return False
                    else:
                        # All assigned, count_0=4, count_1=1. No conflict, no force.
                        pass
                elif count_1 == 4:
                    if unassigned is not None:
                        if colors[unassigned] == 0:
                            pass
                        elif colors[unassigned] == -1:
                            queue.append((unassigned, 0))
                        else:
                            return False
                    else:
                        pass
        return True

    def backtrack(idx):
        # Find next unassigned vertex
        # Heuristic: pick vertex with most constraints? Or just first unassigned?
        # For small n, first unassigned is fine.
        next_v = -1
        for i in range(n):
            if colors[i] == -1:
                next_v = i
                break
        
        if next_v == -1:
            return True # All assigned, no conflict found, so proper coloring exists
        
        # Try color 0
        # Save state
        old_colors = colors[:]
        
        if propagate(next_v, 0):
            if backtrack(idx + 1):
                return True
        
        # Restore state
        colors[:] = old_colors
        
        # Try color 1
        old_colors = colors[:]
        
        if propagate(next_v, 1):
            if backtrack(idx + 1):
                return True
        
        # Restore state
        colors[:] = old_colors
        
        return False

    # Initial state: all -1
    # We can start with any vertex.
    # To speed up, we can pick a vertex in the largest edge?
    # Just start with vertex 0.
    if not propagate(0, 0):
        # Try 1? No, propagate(0,0) returns False if conflict.
        # If conflict, try 1.
        colors = [-1] * n
        if not propagate(0, 1):
            return False
        # Wait, propagate modifies colors. If it returns True, colors are set.
        # If it returns False, we need to reset and try the other color?
        # Actually, my propagate function sets colors. If it fails, it might have partially set them.
        # Better to encapsulate state changes properly or just reset.
        
    # Let's rewrite the backtracking to be cleaner with state snapshots.
    
    def solve():
        # Use a stack for backtracking to avoid deep recursion if n is large, but n<=26 so recursion is fine.
        # However, we need to manage the 'colors' array carefully.
        
        # Heuristic: MRV (Minimum Remaining Values) is hard here as domain is always {0,1} until assigned.
        # Let's just pick the first unassigned vertex.
        
        def bt():
            # Find first unassigned
            v = -1
            for i in range(n):
                if colors[i] == -1:
                    v = i
                    break
            if v == -1:
                return True
            
            # Try 0
            snapshot = colors[:]
            if propagate(v, 0):
                if bt():
                    return True
            colors[:] = snapshot
            
            # Try 1
            snapshot = colors[:]
            if propagate(v, 1):
                if bt():
                    return True
            colors[:] = snapshot
            
            return False
        
        # Initial call
        # We can start with any vertex.
        # To be safe, let's try 0 and 1 for vertex 0.
        # But bt() handles the choice.
        # We just need to call bt().
        # But bt() picks the first unassigned, which is 0 initially.
        return bt()

    return solve()

def find_coloring(n, edges):
    """
    Finds a valid 2-coloring if one exists.
    Returns list of colors (0/1) or None.
    """
    # Similar to is_2_colorable but returns the coloring.
    # We can reuse the logic but store the solution.
    
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
            
    colors = [-1] * n
    solution = None

    def propagate(start_v, color):
        queue = [(start_v, color)]
        visited = set()
        while queue:
            v, c = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            colors[v] = c
            
            for e_idx in vertex_edges[v]:
                e = edges[e_idx]
                count_0 = 0
                count_1 = 0
                unassigned = None
                for u in e:
                    if colors[u] == 0:
                        count_0 += 1
                    elif colors[u] == 1:
                        count_1 += 1
                    else:
                        unassigned = u
                
                if count_0 == 5 or count_1 == 5:
                    return False
                
                if count_0 == 4:
                    if unassigned is not None:
                        if colors[unassigned] == 1:
                            pass
                        elif colors[unassigned] == -1:
                            queue.append((unassigned, 1))
                        else:
                            return False
                elif count_1 == 4:
                    if unassigned is not None:
                        if colors[unassigned] == 0:
                            pass
                        elif colors[unassigned] == -1:
                            queue.append((unassigned, 0))
                        else:
                            return False
        return True

    def bt():
        nonlocal solution
        v = -1
        for i in range(n):
            if colors[i] == -1:
                v = i
                break
        if v == -1:
            solution = colors[:]
            return True
        
        snapshot = colors[:]
        if propagate(v, 0):
            if bt():
                return True
        colors[:] = snapshot
        
        snapshot = colors[:]
        if propagate(v, 1):
            if bt():
                return True
        colors[:] = snapshot
        
        return False

    if bt():
        return solution
    return None

def construct():
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    # Try different n values
    best_result = None
    
    for n in range(9, 13): # 9, 10, 11, 12
        if time.time() > deadline - 10:
            break
            
        # Start with empty edges
        edges = []
        
        # Randomized Greedy Construction
        # We want to add edges until it's not 2-colorable.
        # To avoid getting stuck in a local minimum or taking too long,
        # we can set a max number of steps.
        
        # Initialize with some random edges? Or start empty.
        # Starting empty, it's 2-colorable.
        # We need to add edges.
        
        # To make it faster, we can pick a random coloring of the empty graph (any coloring works),
        # and add a monochromatic edge from it.
        
        # Let's use a fixed random seed for reproducibility in local tests, but random in prod?
        # The task doesn't specify reproducibility, but consistent results are better.
        random.seed(42)
        
        for _ in range(1000): # Max iterations
            if time.time() > deadline - 5:
                break
                
            # Check if currently 2-colorable
            if not is_2_colorable(n, edges):
                break
                
            # Find a coloring
            coloring = find_coloring(n, edges)
            if coloring is None:
                # Should not happen if is_2_colorable returned True
                break
                
            # Find monochromatic 5-sets in this coloring
            mono_edges = []
            for combo in combinations(range(n), 5):
                # Check if all have same color
                c0 = coloring[combo[0]]
                if all(coloring[v] == c0 for v in combo):
                    mono_edges.append(list(combo))
            
            if not mono_edges:
                # This shouldn't happen if the coloring is valid (no mono edges)
                # Wait, find_coloring returns a coloring with NO mono edges.
                # So there are NO mono edges in this coloring.
                # This means we CANNOT add an edge from this coloring to break it?
                # No, if we add an edge that IS monochromatic in this coloring,
                # then this specific coloring becomes invalid.
                # But find_coloring returns a coloring where NO edge in `edges` is monochromatic.
                # The new edge we add is NOT in `edges` yet.
                # So we need to find a set of 5 vertices that are monochromatic in `coloring`.
                # And add that set as a new edge.
                
                # If `coloring` has no monochromatic 5-sets, it means all 5-sets are mixed.
                # This is impossible if n is small?
                # If n=9, and we have a 4-5 split, there are monochromatic 5-sets (the 5 of the majority color).
                # If we have a 4-4 split (n=8), no mono 5-sets.
                # For n=9, any coloring has at least 5 vertices of one color.
                # So there is always at least one monochromatic 5-set.
                
                pass
            
            # Pick a random mono edge
            if mono_edges:
                new_edge = random.choice(mono_edges)
                # Avoid duplicates
                if new_edge not in edges:
                    edges.append(new_edge)
            else:
                # Fallback: add a random edge
                new_edge = list(random.sample(range(n), 5))
                edges.append(new_edge)
        
        # Now edges is non-2-colorable (hopefully)
        if is_2_colorable(n, edges):
            continue
            
        # Minimize edges
        # Try to remove each edge
        i = 0
        while i < len(edges):
            if time.time() > deadline - 5:
                break
                
            e = edges[i]
            # Temporarily remove
            edges.remove(e)
            
            # Check if still non-2-colorable
            if not is_2_colorable(n, edges):
                # Keep removed
                i += 1 # Don't increment i, as list shifted
                # Wait, if we remove, indices shift.
                # Better to iterate in reverse or use a set.
                # Let's restart the removal loop from 0 to be safe and simple.
                i = 0
            else:
                # Restore
                edges.append(e)
                i += 1
        
        # Update best
        if best_result is None or len(edges) < len(best_result['edges']):
            best_result = {'n': n, 'edges': edges}
            
    if best_result is None:
        # Fallback to K9^5
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
        
    return best_result
