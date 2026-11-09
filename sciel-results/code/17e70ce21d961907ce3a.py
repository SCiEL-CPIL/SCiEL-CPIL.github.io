import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation and trail-based undo.
    """
    m = len(edges)
    if m == 0:
        return True
    
    # Map vertex to color: -1 (unassigned), 0, 1
    colors = [-1] * n
    
    # Precompute for each vertex, which edges it belongs to
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
            
    # Count of 0s and 1s in each edge
    count_0 = [0] * m
    count_1 = [0] * m
    
    # Trail for undoing
    trail = []

    def propagate(idx, color):
        # Assign colors[idx] = color
        # Returns False if contradiction, True if okay
        if colors[idx] != -1:
            return colors[idx] == color
        
        colors[idx] = color
        trail.append((idx, color))
        
        for e_idx in vertex_edges[idx]:
            if color == 0:
                count_0[e_idx] += 1
                if count_0[e_idx] == 5:
                    return False # Monochromatic 0
            else:
                count_1[e_idx] += 1
                if count_1[e_idx] == 5:
                    return False # Monochromatic 1
            
            # Unit propagation check
            # If 4 are 0, the last must be 1
            # If 4 are 1, the last must be 0
            if count_0[e_idx] == 4:
                # Find the unassigned vertex in this edge
                unassigned = None
                for v in edges[e_idx]:
                    if colors[v] == -1:
                        if unassigned is not None:
                            return False # Should not happen if consistent
                        unassigned = v
                if unassigned is not None:
                    if not propagate(unassigned, 1):
                        return False
            elif count_1[e_idx] == 4:
                unassigned = None
                for v in edges[e_idx]:
                    if colors[v] == -1:
                        if unassigned is not None:
                            return False # Should not happen if consistent
                        unassigned = v
                if unassigned is not None:
                    if not propagate(unassigned, 0):
                        return False
        return True

    def undo_to(length):
        while len(trail) > length:
            idx, color = trail.pop()
            colors[idx] = -1
            for e_idx in vertex_edges[idx]:
                if color == 0:
                    count_0[e_idx] -= 1
                else:
                    count_1[e_idx] -= 1

    def backtrack():
        # Heuristic: pick vertex with most unassigned neighbors in constrained edges?
        # Simple heuristic: pick first unassigned. 
        # Better: pick vertex in an edge with highest count of assigned?
        # For now, keep it simple but efficient.
        idx = -1
        for i in range(n):
            if colors[i] == -1:
                idx = i
                break
        
        if idx == -1:
            return True # All assigned, no contradiction found
        
        trail_len = len(trail)
        
        # Try 0
        if propagate(idx, 0):
            if backtrack():
                return True
            undo_to(trail_len)
        
        # Try 1
        if propagate(idx, 1):
            if backtrack():
                return True
            undo_to(trail_len)
            
        return False

    return backtrack()

def construct():
    # We want to find a small non-2-colorable 5-uniform hypergraph.
    # Known bounds: m(5) <= 51.
    # We try to find one with fewer edges via random search.
    
    best_edges = None
    best_count = 1000
    best_n = 0
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s for safety
    
    # Try n from 9 to 15
    for n in range(9, 16):
        if time.time() - start_time > timeout:
            break
            
        all_edges = list(combinations(range(n), 5))
        total_possible = len(all_edges)
        
        # We want to find a subset of size E that is not 2-colorable.
        # Start E from 29 (known lower bound-ish) up to 51.
        # If we find one, we record it.
        # We should prioritize smaller E.
        
        for E in range(29, 52):
            if E > total_possible:
                continue
                
            if time.time() - start_time > timeout:
                break
                
            # Limit random tries
            # For smaller E, fewer tries might be needed if we are lucky, 
            # but probability is low. 
            # For larger E, probability is high.
            
            # Heuristic: try a few random subsets
            max_tries = 200 if E < 40 else 50
            
            found = False
            for _ in range(max_tries):
                if time.time() - start_time > timeout:
                    break
                    
                # Select E random edges
                # Using random.sample is O(E) if we pre-shuffle, or O(E log E) if not?
                # random.sample from a list is O(E).
                candidate_indices = random.sample(range(total_possible), E)
                candidate = [list(all_edges[i]) for i in candidate_indices]
                
                if not is_2_colorable(n, candidate):
                    if E < best_count:
                        best_count = E
                        best_edges = candidate
                        best_n = n
                    found = True
                    break # Stop trying this E for this n
                    
            if found:
                # If we found a solution for this n, we can stop checking larger E for this n
                # because we want minimum E.
                # But we might find a smaller E with a different n?
                # Yes, so we continue outer loop.
                # However, if we found E=29, we can break everything.
                if best_count == 29:
                    break

        if best_count == 29:
            break

    if best_edges:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: The complete graph on 9 vertices (126 edges)
        # This is guaranteed to be non-2-colorable.
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
