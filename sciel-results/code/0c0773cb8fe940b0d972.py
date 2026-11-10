import itertools
import random
import time
from collections import deque

def construct():
    """
    Return a 5-uniform hypergraph on n vertices with no proper 2-colouring,
    minimizing the number of edges.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_edges = None
    best_n = 0
    best_count = 1000
    
    def is_not_2_colorable(n, edges):
        """
        Check if hypergraph is NOT 2-colorable (i.e., every 2-coloring has a monochromatic edge).
        Returns True if NOT 2-colorable.
        """
        m = len(edges)
        if m == 0:
            return False
        
        # Precompute
        edge_vertices = [set(e) for e in edges]
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        # Sort vertices by degree descending
        order = sorted(range(n), key=lambda v: len(vertex_edges[v]), reverse=True)
        
        assignment = [-1] * n
        # edge_state: 0 = unassigned, 1 = has 0s, 2 = has 1s, 3 = mixed (both 0 and 1 seen)
        # Actually, we need to track if it's "dead" (monochromatic).
        # If an edge has only 0s so far, and we assign another 0, it's dead.
        # So for each edge, track:
        # count_0, count_1
        # If count_0 == 5 or count_1 == 5, it is monochromatic -> Conflict.
        
        edge_c0 = [0] * m
        edge_c1 = [0] * m
        edge_assigned = [0] * m
        
        def propagate(v, c):
            """
            Assign color c to vertex v.
            Return False if conflict.
            Also perform unit propagation: if an edge has 4 of one color, force the 5th.
            """
            assignment[v] = c
            for ei in vertex_edges[v]:
                if c == 0:
                    edge_c0[ei] += 1
                else:
                    edge_c1[ei] += 1
                edge_assigned[ei] += 1
                
                # Check if monochromatic
                if edge_c0[ei] == 5 or edge_c1[ei] == 5:
                    return False
                
                # Unit Propagation
                if edge_assigned[ei] == 4:
                    # 4 vertices assigned, 1 unassigned
                    if edge_c0[ei] == 4:
                        # Must be 1
                        # Find the unassigned vertex in this edge
                        for u in edge_vertices[ei]:
                            if assignment[u] == -1:
                                if c == 1:
                                    return False # Conflict: forced 1, but we are assigning 0? No, we are assigning v.
                                    # The forced vertex is u.
                                    # If u is already assigned 0, conflict.
                                    # If u is unassigned, force it to 1.
                                    # But we are in propagate(v,c). 
                                    # We need to return the forced assignment.
                                    # This simple propagate doesn't handle chains well.
                                    # Let's just check conflict here.
                                    # If edge has 4 zeros, the last must be 1.
                                    # If the last vertex is already 0, conflict.
                                    # If unassigned, we don't force it here, but the search will pick it.
                                    # However, if the last vertex is v? No, v is the one we just assigned.
                                    # So the 4 assigned include v?
                                    # If edge_c0[ei] == 4, it means 4 vertices in this edge are 0.
                                    # One of them is v.
                                    # The 5th vertex u is unassigned.
                                    # It MUST be 1.
                                    # We don't assign it here, but if the search assigns it 0 later, it will fail.
                                    # To be faster, we should force it.
                                    pass
            return True

        # Simpler Backtracking without complex propagation, but with good ordering
        def search(idx):
            if idx == len(order):
                return True # All assigned, no conflict found -> Colorable
            
            v = order[idx]
            # Try 0
            if assignment[v] == -1:
                # Assign 0
                assignment[v] = 0
                ok = True
                for ei in vertex_edges[v]:
                    edge_c0[ei] += 1
                    edge_assigned[ei] += 1
                    if edge_c0[ei] == 5:
                        ok = False
                        break
                if ok:
                    if search(idx + 1):
                        return True
                # Undo
                for ei in vertex_edges[v]:
                    edge_c0[ei] -= 1
                    edge_assigned[ei] -= 1
                assignment[v] = -1
                
            # Try 1
            if assignment[v] == -1:
                assignment[v] = 1
                ok = True
                for ei in vertex_edges[v]:
                    edge_c1[ei] += 1
                    edge_assigned[ei] += 1
                    if edge_c1[ei] == 5:
                        ok = False
                        break
                if ok:
                    if search(idx + 1):
                        return True
                # Undo
                for ei in vertex_edges[v]:
                    edge_c1[ei] -= 1
                    edge_assigned[ei] -= 1
                assignment[v] = -1
            
            return False
        
        return not search(0)

    def reduce_edges(n, edges):
        """
        Try to remove edges while maintaining non-2-colorability.
        """
        edges = list(edges)
        i = 0
        while i < len(edges):
            new_edges = edges[:i] + edges[i+1:]
            if is_not_2_colorable(n, new_edges):
                edges = new_edges
            else:
                i += 1
        return edges

    # Search for small non-2-colorable hypergraphs
    # Try different n and edge counts
    for n in [15, 18, 20, 22, 25]:
        if time.time() - start_time > time_limit:
            break
        for E in range(29, 60):
            if time.time() - start_time > time_limit:
                break
            # Try a few random instances
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                # Random edges
                all_possible = list(itertools.combinations(range(n), 5))
                random.shuffle(all_possible)
                edges = [list(e) for e in all_possible[:E]]
                
                if is_not_2_colorable(n, edges):
                    # Found one! Try to reduce
                    reduced = reduce_edges(n, edges)
                    if len(reduced) < best_count:
                        best_count = len(reduced)
                        best_edges = reduced
                        best_n = n
                        # If we found 29, we can stop (theoretical min)
                        if best_count == 29:
                            return {'n': best_n, 'edges': best_edges}

    # If no good one found, return a known valid one (e.g., the 126 edge one or a hardcoded 51)
    # Let's return the 126 edge one as fallback, but hopefully we found better.
    # Actually, to ensure we beat the seed (126), we should try harder or use a fallback < 126.
    # Let's use a hardcoded 51-edge construction if available?
    # I don't have a hardcoded 51-edge one memorized.
    # But the random search on n=20, E=40 should find non-2-colorable ones easily.
    # And reduction should bring it down significantly.
    
    if best_edges is None:
        # Fallback: Complete 5-uniform on 9 vertices
        n_start = 9
        all_edges_5 = list(itertools.combinations(range(n_start), 5))
        return {'n': n_start, 'edges': [list(e) for e in all_edges_5]}
        
    return {'n': best_n, 'edges': best_edges}
