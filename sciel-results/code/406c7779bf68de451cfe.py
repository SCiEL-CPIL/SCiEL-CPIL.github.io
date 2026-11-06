import time
import itertools
import random

def _is_2_colorable(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    assignment = [-1] * n
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for e in edges:
                c0 = 0
                c1 = 0
                unassigned = []
                for v in e:
                    if assignment[v] == 0:
                        c0 += 1
                    elif assignment[v] == 1:
                        c1 += 1
                    else:
                        unassigned.append(v)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(unassigned) == 1:
                    assignment[unassigned[0]] = 1
                    changed = True
                elif c1 == 4 and len(unassigned) == 1:
                    assignment[unassigned[0]] = 0
                    changed = True
        return True

    def backtrack():
        if not propagate():
            return False
        
        # Find unassigned vertex
        unassigned_verts = [v for v in range(n) if assignment[v] == -1]
        if not unassigned_verts:
            return True
        
        # Heuristic: Pick the vertex that is in the edge with the fewest unassigned vertices.
        # This helps prune the tree.
        best_v = unassigned_verts[0]
        min_unassigned_in_edge = 100
        
        for e in edges:
            unassigned_in_e = [v for v in e if assignment[v] == -1]
            if len(unassigned_in_e) < min_unassigned_in_edge:
                min_unassigned_in_edge = len(unassigned_in_e)
                # Pick the first unassigned vertex in this edge
                best_v = unassigned_in_e[0]
                if min_unassigned_in_edge == 1:
                    break # Can't get better than 1
        
        v = best_v
        
        # Try 0
        assignment[v] = 0
        if backtrack():
            return True
        
        # Try 1
        assignment[v] = 1
        if backtrack():
            return True
        
        assignment[v] = -1
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 9
    all_edges = [tuple(e) for e in itertools.combinations(range(n), 5)]
    current_edges = list(all_edges)
    
    removed = True
    while removed and time.time() < deadline:
        removed = False
        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            candidate_edges = current_edges[:i] + current_edges[i+1:]
            
            if not _is_2_colorable(n, candidate_edges):
                current_edges = candidate_edges
                removed = True

    edges_out = [list(e) for e in current_edges]
    return {'n': n, 'edges': edges_out}
