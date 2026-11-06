import time
from itertools import combinations

def _is_not_2_colorable(n, edges):
    """
    Check if hypergraph is NOT 2-colorable.
    Returns True if no proper 2-coloring exists.
    """
    if not edges:
        return False

    m = len(edges)
    # Sort edges and vertices for consistency
    edges = [tuple(sorted(e)) for e in edges]
    
    # Incidence list: vertex -> list of edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)

    # State
    assignment = [-1] * n
    # For each edge: count of 0s, count of 1s, count of unassigned
    edge_count0 = [0] * m
    edge_count1 = [0] * m
    edge_unassigned = [5] * m
    
    # Symmetry breaking: Vertex 0 must be 0
    # This reduces search space by half
    assignment[0] = 0
    for ei in vertex_edges[0]:
        edge_count0[ei] += 1
        edge_unassigned[ei] -= 1
        # If an edge has 5 zeros, it's monochromatic -> invalid path
        if edge_count0[ei] == 5:
            return True # Wait, if we assume v0=0 and it leads to contradiction, 
            # it means there is NO coloring with v0=0.
            # Due to symmetry, if no coloring with v0=0, no coloring with v0=1 either?
            # Not exactly, but if a hypergraph is non-2-colorable, both branches fail.
            # If a branch fails immediately, it might mean this specific assignment is bad,
            # but we need to check if ANY assignment works.
            # Actually, if v0=0 leads to a contradiction (monochromatic edge forced),
            # it doesn't mean the graph is non-2-colorable, it just means v0=0 is a bad start?
            # No, if v0=0 forces a monochromatic edge, that specific partial assignment is invalid.
            # We must backtrack.
            # However, if the graph is non-2-colorable, ALL assignments lead to contradiction.
            # If v0=0 leads to contradiction, we must check v0=1?
            # Yes. So we cannot just return True here. We must return False for this branch.
            # But wait, if I fix v0=0, and it fails, I haven't proven non-2-colorability.
            # I've only proven that no coloring with v0=0 exists.
            # I must also check v0=1.
            # So symmetry breaking is only safe if we are sure the graph is non-2-colorable?
            # No, we are checking if it IS.
            # If it IS non-2-colorable, both v0=0 and v0=1 branches will fail.
            # If it IS 2-colorable, at least one branch will succeed.
            # So we can try v0=0 first. If it succeeds, return False (it is 2-colorable).
            # If it fails, try v0=1. If it succeeds, return False.
            # If both fail, return True.
            
            # Let's structure the function to try both if needed.
            # But for speed, we can just run the full search without fixing v0, 
            # or fix v0=0 and if it returns "no solution", we must check v0=1.
            
            # Let's stick to a standard backtracking that tries 0 and 1 for the first unassigned vertex.
            # To speed up, we pick the vertex with the highest degree or most constrained edges.
            
            pass # Placeholder for logic below

    def backtrack():
        # Find most constrained unassigned vertex
        # Heuristic: Vertex in the most edges with low unassigned count?
        # Or simply first unassigned.
        # Let's use first unassigned for simplicity, but check monochromatic edges first.
        
        # 1. Check for contradictions (monochromatic edges)
        for i in range(m):
            if edge_count0[i] == 5 or edge_count1[i] == 5:
                return False # This branch is invalid

        # 2. Unit Propagation
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if edge_unassigned[i] == 1:
                    if edge_count0[i] == 4:
                        # Last vertex must be 1
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 1
                                edge_count1[i] += 1
                                edge_unassigned[i] -= 1
                                # Update other edges containing v
                                for ei in vertex_edges[v]:
                                    if ei != i:
                                        edge_count1[ei] += 1
                                        edge_unassigned[ei] -= 1
                                        if edge_count1[ei] == 5 or edge_count0[ei] == 5:
                                            return False
                                changed = True
                                break
                    elif edge_count1[i] == 4:
                        # Last vertex must be 0
                        for v in edges[i]:
                            if assignment[v] == -1:
                                assignment[v] = 0
                                edge_count0[i] += 1
                                edge_unassigned[i] -= 1
                                # Update other edges containing v
                                for ei in vertex_edges[v]:
                                    if ei != i:
                                        edge_count0[ei] += 1
                                        edge_unassigned[ei] -= 1
                                        if edge_count1[ei] == 5 or edge_count0[ei] == 5:
                                            return False
                                changed = True
                                break
        if not changed:
            # No unit propagation possible, find next variable
            # Find first unassigned vertex
            idx = -1
            for i in range(n):
                if assignment[i] == -1:
                    idx = i
                    break
            
            if idx == -1:
                # All assigned, check for monochromatic edges
                for i in range(m):
                    if edge_count0[i] == 5 or edge_count1[i] == 5:
                        return False
                return True # Proper coloring found

            # Branch on idx
            # Try 0
            assignment[idx] = 0
            # Apply changes
            backup_e0 = edge_count0[:]
            backup_e1 = edge_count1[:]
            backup_eu = edge_unassigned[:]
            
            valid = True
            for ei in vertex_edges[idx]:
                edge_count0[ei] += 1
                edge_unassigned[ei] -= 1
                if edge_count0[ei] == 5:
                    valid = False
                    break
            
            if valid and backtrack():
                return True # Found a valid coloring (so graph IS 2-colorable)
            
            # Undo changes
            edge_count0[:] = backup_e0
            edge_count1[:] = backup_e1
            edge_unassigned[:] = backup_eu
            assignment[idx] = -1
            
            # Try 1
            assignment[idx] = 1
            backup_e0 = edge_count0[:]
            backup_e1 = edge_count1[:]
            backup_eu = edge_unassigned[:]
            
            valid = True
            for ei in vertex_edges[idx]:
                edge_count1[ei] += 1
                edge_unassigned[ei] -= 1
                if edge_count1[ei] == 5:
                    valid = False
                    break
            
            if valid and backtrack():
                return True # Found a valid coloring
            
            # Undo changes
            edge_count0[:] = backup_e0
            edge_count1[:] = backup_e1
            edge_unassigned[:] = backup_eu
            assignment[idx] = -1
            
            return False # No valid coloring in this branch

    # Start backtracking
    # We don't fix vertex 0 to avoid the complexity of checking both branches manually if one fails.
    # The backtracking function handles branching.
    return not backtrack()

def _greedy_reduce(n, edges, max_time=10.0):
    """
    Try to remove edges while maintaining non-2-colorability.
    """
    edges = list(edges)
    start_time = time.time()
    changed = True
    while changed and time.time() - start_time < max_time:
        changed = False
        for i in range(len(edges)):
            if time.time() - start_time > max_time:
                break
            candidate = edges[:i] + edges[i+1:]
            if _is_not_2_colorable(n, candidate):
                edges = candidate
                changed = True
                break # Restart loop as indices changed
    return edges

def construct():
    start_time = time.time()
    deadline = 145 # Safety margin for 150s limit
    
    best_edges = 999
    best_result = None

    # Start with K9(5) which is non-2-colorable
    # n=9, all 126 edges
    n = 9
    all_edges = list(combinations(range(n), 5))
    
    # Verify K9 is non-2-colorable (it should be)
    if _is_not_2_colorable(n, all_edges):
        # Greedy reduction
        reduced = _greedy_reduce(n, all_edges, max_time=60.0)
        if len(reduced) < best_edges:
            best_edges = len(reduced)
            best_result = {'n': n, 'edges': [list(e) for e in reduced]}
            
    # Try other n values if time permits
    # n=10, random dense hypergraphs
    if time.time() < deadline - 30:
        for n in [10, 11]:
            # Try to find a non-2-colorable hypergraph with fewer edges
            # Randomly generate edges, check, then reduce
            for _ in range(5):
                if time.time() > deadline - 30:
                    break
                # Start with a random subset of ~80 edges
                random.shuffle(all_edges) # Reuse list? No, different n.
                import random
                all_edges_n = list(combinations(range(n), 5))
                random.shuffle(all_edges_n)
                # Take enough edges to likely be non-2-colorable
                # Threshold for 5-uniform is around n/2 * log n?
                # For n=10, maybe 50-60 edges.
                edges = all_edges_n[:70]
                if _is_not_2_colorable(n, edges):
                    reduced = _greedy_reduce(n, edges, max_time=20.0)
                    if len(reduced) < best_edges:
                        best_edges = len(reduced)
                        best_result = {'n': n, 'edges': [list(e) for e in reduced]}

    # Fallback
    if best_result is None:
        # Return K9 if reduction failed or wasn't run
        return {'n': 9, 'edges': [list(e) for e in all_edges]}
        
    return best_result
