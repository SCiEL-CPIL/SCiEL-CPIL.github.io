"""Combined program for finding minimal non-2-colorable 5-uniform hypergraphs."""
import time
import random
from itertools import combinations

def is_2_colorable_fast(edges, n):
    """
    Fast 2-colorability check using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Build vertex-to-edge index
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    # For each edge, store as a tuple for fast access
    edge_tuples = [tuple(e) for e in edges]
    
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    
    def propagate(v, c):
        """Assign color c to vertex v with unit propagation. Returns False on contradiction."""
        if colors[v] != -1:
            return colors[v] == c
        
        colors[v] = c
        
        # Process all edges containing v
        for e_idx in vertex_edges[v]:
            e = edge_tuples[e_idx]
            count_c = 0
            count_1c = 0
            unassigned = -1
            
            for u in e:
                cu = colors[u]
                if cu == c:
                    count_c += 1
                elif cu == 1 - c:
                    count_1c += 1
                else:
                    unassigned = u
            
            # Contradiction check
            if count_c == 5 or count_1c == 5:
                return False
            
            # Forced assignment
            if count_c == 4:
                if unassigned != -1 and colors[unassigned] == -1:
                    if not propagate(unassigned, 1 - c):
                        return False
            elif count_1c == 4:
                if unassigned != -1 and colors[unassigned] == -1:
                    if not propagate(unassigned, c):
                        return False
        
        return True
    
    def backtrack():
        # Find unassigned vertex with highest degree
        best_v = -1
        best_deg = -1
        for v in range(n):
            if colors[v] == -1:
                deg = len(vertex_edges[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True  # All assigned, no contradictions
        
        # Try color 0
        old_colors = colors[:]
        if propagate(best_v, 0):
            if backtrack():
                return True
        colors[:] = old_colors
        
        # Try color 1
        old_colors = colors[:]
        if propagate(best_v, 1):
            if backtrack():
                return True
        
        return False
    
    return backtrack()


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Best known: 51 edges. Let's try to find a construction.
    # Strategy: For various n, greedily add edges in random order until non-2-colorable.
    
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_count = len(best_edges)  # 126
    
    # Try to find better solutions
    # Known constructions often use n around 10-15
    
    for n in range(9, 16):
        if time.time() > deadline - 10.0:
            break
        
        all_edges = list(combinations(range(n), 5))
        
        # Multiple random trials for this n
        max_trials = 50 if n <= 12 else 20
        for trial in range(max_trials):
            if time.time() > deadline - 10.0:
                break
            
            random.shuffle(all_edges)
            current_edges = []
            
            for e in all_edges:
                if time.time() > deadline - 5.0:
                    break
                if len(current_edges) >= best_count:
                    break
                
                current_edges.append(e)
                
                # Check if non-2-colorable
                if not is_2_colorable_fast(current_edges, n):
                    count = len(current_edges)
                    if count < best_count:
                        best_count = count
                        best_n = n
                        best_edges = [list(e) for e in current_edges]
                    break  # Found one for this ordering, try next
        
        # Early exit if we hit the known minimum
        if best_count <= 51:
            break
    
    return {'n': best_n, 'edges': best_edges}
