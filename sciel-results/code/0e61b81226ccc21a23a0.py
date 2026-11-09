import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False if not.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Pre-process edges to lists for faster access
    edge_lists = [list(e) for e in edges]
    
    # colors: -1 for unknown, 0, 1
    colors = [-1] * n
    
    # Helper to check if an edge is violated or satisfied
    # Returns:
    # 0: satisfied (mixed)
    # 1: violated (monochromatic)
    # 2: unknown
    # 3: unit propagation needed (4 of one color, 1 unknown)
    # Also returns the vertex to assign and the color if unit propagation
    def check_edge(e):
        c0 = 0
        c1 = 0
        unknowns = []
        for v in e:
            if colors[v] == 0:
                c0 += 1
            elif colors[v] == 1:
                c1 += 1
            else:
                unknowns.append(v)
        
        if c0 == 5 or c1 == 5:
            return 1, None, None
        if not unknowns:
            return 0, None, None
        
        if c0 == 4:
            return 3, unknowns[0], 1
        if c1 == 4:
            return 3, unknowns[0], 0
            
        return 2, None, None

    def backtrack(node_limit):
        # We use a queue for unit propagation
        # For simplicity, we do recursive backtracking with MRV
        # But to be fast, we do a quick pass for unit propagation first
        
        # Find a variable to assign
        # MRV: pick vertex in an edge with fewest unknowns?
        # Or just pick first unassigned vertex in an unsatisfied edge?
        
        # Let's iterate edges to find one that is not satisfied
        # And check for unit propagation
        for i, e in enumerate(edge_lists):
            status, v, c = check_edge(e)
            if status == 1:
                return False
            if status == 3:
                # Unit propagation
                if colors[v] != -1:
                    if colors[v] != c:
                        return False
                else:
                    colors[v] = c
                    # Recurse immediately?
                    # This is a simplification. Proper UP would process all.
                    # Let's just recurse.
                    if backtrack(node_limit):
                        return True
                    colors[v] = -1
                    return False
        
        # No unit propagation, no violations.
        # Find an edge that is not satisfied (has unknowns)
        # Pick a vertex from it to assign
        target_edge = None
        for e in edge_lists:
            status, _, _ = check_edge(e)
            if status == 2:
                target_edge = e
                break
        
        if target_edge is None:
            # All edges satisfied
            return True
            
        # Pick a vertex from target_edge that is unknown
        # Heuristic: pick vertex with highest degree? 
        # For now, pick the first unknown.
        v = None
        for u in target_edge:
            if colors[u] == -1:
                v = u
                break
                
        if v is None:
            return True

        # Try 0
        colors[v] = 0
        if backtrack(node_limit):
            return True
        colors[v] = -1
        
        # Try 1
        colors[v] = 1
        if backtrack(node_limit):
            return True
        colors[v] = -1
        
        return False

    # Limit nodes to avoid long runs on 2-colorable instances
    # If we exceed, we assume it's likely 2-colorable and return True (discard candidate)
    # This is a heuristic for speed.
    # However, we MUST be correct for "Not 2-colorable".
    # If we time out, we don't know.
    # So we only return False if the search space is exhausted.
    
    # To ensure we don't hang, we rely on the fact that small n (13-15) 
    # and small e (29-50) should be fast enough for a basic solver 
    # if we add a node counter.
    
    # Let's add a node counter to the closure
    state = {'nodes': 0}
    LIMIT = 100000
    
    def backtrack_limited():
        state['nodes'] += 1
        if state['nodes'] > LIMIT:
            return None # Unknown
            
        # Check all edges for UP/Violation
        for i in range(len(edge_lists)):
            e = edge_lists[i]
            c0 = 0
            c1 = 0
            unknowns = []
            for v in e:
                if colors[v] == 0: c0 += 1
                elif colors[v] == 1: c1 += 1
                else: unknowns.append(v)
                
            if c0 == 5 or c1 == 5:
                return False
            if not unknowns:
                continue
                
            if c0 == 4:
                v = unknowns[0]
                if colors[v] != -1:
                    if colors[v] != 1:
                        return False
                else:
                    colors[v] = 1
                    res = backtrack_limited()
                    if res is not None:
                        return res
                    colors[v] = -1
                    return None
            if c1 == 4:
                v = unknowns[0]
                if colors[v] != -1:
                    if colors[v] != 0:
                        return False
                else:
                    colors[v] = 0
                    res = backtrack_limited()
                    if res is not None:
                        return res
                    colors[v] = -1
                    return None

        # No UP
        # Find unknown vertex
        v = None
        for e in edge_lists:
            for u in e:
                if colors[u] == -1:
                    v = u
                    break
            if v is not None:
                break
                
        if v is None:
            return True

        # Try 0
        colors[v] = 0
        res = backtrack_limited()
        if res is not None:
            return res
        colors[v] = -1
        
        # Try 1
        colors[v] = 1
        res = backtrack_limited()
        if res is not None:
            return res
        colors[v] = -1
        
        return False

    res = backtrack_limited()
    if res is None:
        # Timeout/Node limit, assume colorable to be safe (discard)
        return True
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: Complete 5-uniform hypergraph on 9 vertices
    # 126 edges
    fallback_edges = [list(e) for e in combinations(range(9), 5)]
    best_result = {'n': 9, 'edges': fallback_edges}
    best_edge_count = len(fallback_edges)
    
    # Try to find smaller instances
    # n=13 is a good candidate
    for n in [13, 14, 15]:
        if time.time() > deadline:
            break
            
        # Try increasing edge counts
        for e_count in range(29, 55):
            if time.time() > deadline:
                break
                
            # Try a few random samples
            for _ in range(20):
                if time.time() > deadline:
                    break
                    
                # Generate random edges
                all_possible = list(combinations(range(n), 5))
                random.shuffle(all_possible)
                edges = [list(e) for e in all_possible[:e_count]]
                
                # Check if 2-colorable
                if not _is_2_colorable(n, edges):
                    # Found a non-2-colorable hypergraph
                    current_result = {'n': n, 'edges': edges}
                    if e_count < best_edge_count:
                        best_result = current_result
                        best_edge_count = e_count
                    # We can break inner loops as we found a valid one for this e_count
                    # But we want the MINIMUM, so we continue to next e_count?
                    # No, we are iterating e_count upwards. 
                    # The first one found is the minimum for this n in our search.
                    # We should break the e_count loop? 
                    # No, maybe a smaller e_count was missed? 
                    # We iterate 29, 30... so if 29 failed, 30 is next.
                    # If 30 succeeds, it's better than 126.
                    # We break the sample loop.
                    break
            
            # If we found a valid one for this e_count, we can stop trying higher e_counts for this n
            # Because we want to minimize edges.
            if best_edge_count == e_count:
                break
                
    return best_result
