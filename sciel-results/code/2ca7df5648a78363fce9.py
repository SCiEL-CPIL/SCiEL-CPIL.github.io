"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 5-uniform hypergraph with no proper 2-colouring (no Property B).
    We use a fast bitset-based backtracking with unit propagation to check
    2-colourability, and we search for minimal non-2-colourable hypergraphs.
    
    The best known construction has 51 edges. We aim to find one with <= 51 edges.
    We'll use a deterministic construction based on known results, or a random search
    if needed.
    """
    import random
    import time
    from itertools import combinations
    
    # First, try a known construction. The Petersen graph based construction or
    # other known minimal non-2-colourable 5-uniform hypergraphs.
    
    # Let's try to find a small non-2-colourable hypergraph by random search
    # with backtracking verification.
    
    def is_2_colorable(n, edges):
        """
        Check if a 5-uniform hypergraph on n vertices has a proper 2-colouring.
        A proper 2-colouring means no edge is monochromatic.
        We use backtracking with unit propagation.
        
        Returns True if 2-colourable, False otherwise.
        """
        if not edges:
            return True
        
        # Build edge list as frozensets for quick lookup
        edge_set = [frozenset(e) for e in edges]
        
        # We'll use a recursive backtracking with unit propagation.
        # State: assignment dict {vertex: 0 or 1}, and list of unassigned vertices.
        # For efficiency, we track for each edge how many vertices are assigned and
        # what colours they have. If an edge has all but one vertex assigned to the
        # same colour, the last must be the other colour (unit propagation).
        # If an edge is fully assigned and monochromatic, contradiction.
        
        n_edges = len(edges)
        
        # Precompute for each vertex, which edges contain it
        vertex_to_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_to_edges[v].append(i)
        
        # We'll do DFS with unit propagation.
        # assignment: list of -1 (unassigned), 0, or 1
        assignment = [-1] * n
        
        # For each edge, track: count of 0s, count of 1s
        edge_zero_count = [0] * n_edges
        edge_one_count = [0] * n_edges
        
        def propagate(assignment, edge_zero_count, edge_one_count):
            """
            Apply unit propagation. Returns (success, changed) where success is
            whether no contradiction was found, and changed is whether any new
            assignments were made.
            """
            changed = True
            while changed:
                changed = False
                for i, e in enumerate(edges):
                    z = edge_zero_count[i]
                    o = edge_one_count[i]
                    # If all 5 vertices assigned and all same colour -> contradiction
                    if z + o == 5:
                        if z == 5 or o == 5:
                            return False, False
                        # Mixed, so fine
                        continue
                    # Unit propagation: if 4 vertices are 0 and 1 unassigned, that one must be 1
                    # If 4 vertices are 1 and 1 unassigned, that one must be 0
                    unassigned = [v for v in e if assignment[v] == -1]
                    if len(unassigned) == 1:
                        v = unassigned[0]
                        if z == 4:
                            # Must be 1
                            if assignment[v] != -1:
                                if assignment[v] != 1:
                                    return False, False
                            else:
                                assignment[v] = 1
                                edge_one_count[i] += 1
                                for ei in vertex_to_edges[v]:
                                    if ei != i:
                                        edge_one_count[ei] += 1
                                changed = True
                        elif o == 4:
                            # Must be 0
                            if assignment[v] != -1:
                                if assignment[v] != 0:
                                    return False, False
                            else:
                                assignment[v] = 0
                                edge_zero_count[i] += 1
                                for ei in vertex_to_edges[v]:
                                    if ei != i:
                                        edge_zero_count[ei] += 1
                                changed = True
                    elif len(unassigned) == 0:
                        # All assigned, check not monochromatic
                        if z == 5 or o == 5:
                            return False, False
                    # If 3 or more unassigned, no unit propagation for this edge
            return True, changed
        
        def backtrack(assignment, edge_zero_count, edge_one_count, unassigned_list):
            """
            Recursive backtracking. unassigned_list is a list of unassigned vertices.
            Returns True if a valid 2-colouring exists.
            """
            if not unassigned_list:
                return True
            
            # Pick a vertex with the most constraints (most edges)
            # For simplicity, pick the first one, but we can be smarter
            # Let's pick the vertex involved in the most unassigned edges
            best_idx = 0
            best_count = -1
            for idx, v in enumerate(unassigned_list):
                count = len(vertex_to_edges[v])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            
            v = unassigned_list[best_idx]
            remaining = unassigned_list[:best_idx] + unassigned_list[best_idx+1:]
            
            for colour in (0, 1):
                # Try assigning colour to v
                new_assignment = assignment[:]
                new_edge_zero = edge_zero_count[:]
                new_edge_one = edge_one_count[:]
                
                new_assignment[v] = colour
                if colour == 0:
                    for ei in vertex_to_edges[v]:
                        new_edge_zero[ei] += 1
                else:
                    for ei in vertex_to_edges[v]:
                        new_edge_one[ei] += 1
                
                # Check for immediate contradiction
                valid = True
                for ei in vertex_to_edges[v]:
                    z = new_edge_zero[ei]
                    o = new_edge_one[ei]
                    if z == 5 or o == 5:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Apply unit propagation
                ok, _ = propagate(new_assignment, new_edge_zero, new_edge_one)
                if not ok:
                    continue
                
                # Find remaining unassigned vertices
                new_unassigned = [x for x in remaining if new_assignment[x] == -1]
                
                if backtrack(new_assignment, new_edge_zero, new_edge_one, new_unassigned):
                    return True
            
            return False
        
        # Initial propagation
        ok, _ = propagate(assignment, edge_zero_count, edge_one_count)
        if not ok:
            return False
        
        unassigned = [v for v in range(n) if assignment[v] == -1]
        return backtrack(assignment, edge_zero_count, edge_one_count, unassigned)
    
    # Known minimal constructions:
    # The best known has 51 edges. Let's try to find one with random search.
    
    # Actually, let's use a known explicit construction. 
    # One approach: take the complete 5-uniform hypergraph on 9 vertices (126 edges) 
    # and find a minimal subset that is still non-2-colourable.
    
    # But that's 126 edges. We need <= 51.
    
    # Let's try random search with small n.
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    # Try random hypergraphs on n=8,9,10,11 vertices
    for n in range(8, 13):
        if time.time() - start_time > time_limit:
            break
        all_edges = list(combinations(range(n), 5))
        random.shuffle(all_edges)
        
        # Try increasing number of edges
        for num_edges in range(10, min(len(all_edges), 55)):
            if time.time() - start_time > time_limit:
                break
            # Try several random subsets
            for attempt in range(50):
                if time.time() - start_time > time_limit:
                    break
                edges = [list(e) for e in all_edges[:num_edges]]
                random.shuffle(edges)
                edges = edges[:num_edges]
                
                if not is_2_colorable(n, edges):
                    # Found a non-2-colourable hypergraph
                    return {'n': n, 'edges': edges}
    
    # Fallback: return the complete hypergraph on 9 vertices (126 edges)
    # This is guaranteed to be non-2-colourable
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
# EVOLVE-BLOCK-END
