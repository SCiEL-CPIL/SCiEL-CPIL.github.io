import time
import random
from itertools import combinations

def is_2_colorable(n, edges, timeout=None):
    """Check if hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False if not."""
    if timeout is not None:
        start = time.time()
    
    # Convert edges to frozensets for fast lookup
    edge_sets = [frozenset(e) for e in edges]
    
    # color[i] = -1 (unassigned), 0 or 1
    color = [-1] * n
    
    def unit_propagate():
        """Apply unit propagation: if an edge has all but one vertex colored the same,
        the last must be the opposite color to avoid monochromatic edge.
        Also detect conflicts."""
        changed = True
        while changed:
            changed = False
            for e in edge_sets:
                verts = list(e)
                colors_in_e = [color[v] for v in verts]
                # Count unassigned
                unassigned = [v for v in verts if color[v] == -1]
                if len(unassigned) == 1:
                    # The one unassigned vertex must be colored opposite to the others
                    # if the others are all the same color
                    assigned_colors = [color[v] for v in verts if color[v] != -1]
                    if len(set(assigned_colors)) == 1:
                        # All assigned have same color, unassigned must be opposite
                        required = 1 - assigned_colors[0]
                        v = unassigned[0]
                        if color[v] != -1 and color[v] != required:
                            return False  # conflict
                        if color[v] == -1:
                            color[v] = required
                            changed = True
                # Check for immediate conflict: if all vertices in edge are assigned
                # and all same color -> not 2-colorable with current assignment
                if all(c != -1 for c in colors_in_e):
                    if len(set(colors_in_e)) == 1:
                        return False  # monochromatic edge -> conflict
        return True
    
    def backtrack(idx):
        if timeout is not None:
            if time.time() - start > timeout:
                return None  # timeout
        if idx == n:
            # All vertices colored, check no monochromatic edge
            for e in edge_sets:
                colors_in_e = [color[v] for v in e]
                if len(set(colors_in_e)) == 1:
                    return None  # not 2-colorable with this coloring
            return True  # found a valid 2-coloring
        
        # Try color 0
        color[idx] = 0
        if unit_propagate():
            result = backtrack(idx + 1)
            if result is not None:
                return result
        # Try color 1
        color[idx] = 1
        if unit_propagate():
            result = backtrack(idx + 1)
            if result is not None:
                return result
        color[idx] = -1
        return None
    
    if not unit_propagate():
        return False
    
    result = backtrack(0)
    if result is True:
        return True
    elif result is False:
        return False
    else:
        # Timeout or inconclusive - assume 2-colorable to be safe? 
        # Actually if we can't prove non-2-colorable, we should assume it might be 2-colorable
        return True

def construct():
    deadline = time.time() + 110  # safety margin
    n = 9
    # Start with complete 5-uniform hypergraph on 9 vertices
    # This is known to be non-2-colorable
    initial_edges = list(combinations(range(n), 5))
    
    best_edges = initial_edges[:]
    best_count = len(best_edges)
    
    # Perform multiple rounds of greedy edge removal
    # Each round shuffles the edges and tries to remove them
    # If removing an edge makes the hypergraph 2-colorable, we keep it (revert)
    # If removing it keeps it non-2-colorable, we remove it
    
    max_rounds = 200 # Limit rounds to stay within time
    current_edges = initial_edges[:]
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
            
        random.shuffle(current_edges)
        i = 0
        while i < len(current_edges):
            if time.time() > deadline:
                break
            
            # Try removing edge i
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            # Check if remaining hypergraph is 2-colorable
            # If it IS 2-colorable, we must keep the edge
            # If it is NOT 2-colorable, we can remove it
            if is_2_colorable(n, test_edges, timeout=5):
                # It became 2-colorable, so we need this edge. Skip it.
                i += 1
            else:
                # It remains non-2-colorable, so we remove the edge
                current_edges = test_edges
                # Do not increment i, as the next edge is now at position i
                
        # Update best if improved
        if len(current_edges) < best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]
            
        # If we reached the theoretical lower bound or close to it, we can stop
        # m(5) >= 29, best known <= 51
        if best_count <= 51:
            break
            
    return {'n': n, 'edges': [list(e) for e in best_edges]}
