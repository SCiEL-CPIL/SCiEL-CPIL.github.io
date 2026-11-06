import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """Check if hypergraph is 2-colorable via backtracking with unit propagation.
    Returns True if a proper 2-coloring exists (every edge has both colors)."""
    m = len(edges)
    if m == 0:
        return True
    
    edge_sets = [frozenset(e) for e in edges]
    # Precompute for each vertex, which edges contain it
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edge_sets):
        for v in e:
            v_edges[v].append(ei)
    
    color = [-1] * n
    edge_count = [[0, 0] for _ in range(m)]
    
    # Order vertices by degree descending
    vertex_degree = [len(v_edges[v]) for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -vertex_degree[v])
    
    def backtrack(pos):
        if pos == n:
            return True
        
        v = vertex_order[pos]
        # Try both colors
        for c in [0, 1]:
            # Check if this color is possible
            possible = True
            for ei in v_edges[v]:
                if edge_count[ei][c] == 5:
                    possible = False
                    break
            if not possible:
                continue
            
            # Assign
            color[v] = c
            affected_edges = v_edges[v][:]
            for ei in affected_edges:
                edge_count[ei][c] += 1
            
            # Unit propagation
            forced = []
            ok = True
            for ei in affected_edges:
                if edge_count[ei][0] == 5 or edge_count[ei][1] == 5:
                    ok = False
                    break
                if edge_count[ei][0] == 4:
                    # Force remaining unassigned vertex to color 1
                    for w in edge_sets[ei]:
                        if color[w] == -1:
                            if c == 1:  # already assigned
                                ok = False
                                break
                            color[w] = 1
                            forced.append((w, 1))
                            for ej in v_edges[w]:
                                edge_count[ej][1] += 1
                                if edge_count[ej][1] == 5 or edge_count[ej][0] == 5:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
                elif edge_count[ei][1] == 4:
                    for w in edge_sets[ei]:
                        if color[w] == -1:
                            if c == 0:
                                ok = False
                                break
                            color[w] = 0
                            forced.append((w, 0))
                            for ej in v_edges[w]:
                                edge_count[ej][0] += 1
                                if edge_count[ej][0] == 5 or edge_count[ej][1] == 5:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
            
            if not ok:
                # Undo
                for ei in affected_edges:
                    edge_count[ei][c] -= 1
                color[v] = -1
                for w, fc in forced:
                    color[w] = -1
                    for ej in v_edges[w]:
                        edge_count[ej][fc] -= 1
                continue
            
            if backtrack(pos + 1):
                return True
            
            # Undo
            for ei in affected_edges:
                edge_count[ei][c] -= 1
            color[v] = -1
            for w, fc in forced:
                color[w] = -1
                for ej in v_edges[w]:
                    edge_count[ej][fc] -= 1
        
        return False
    
    return backtrack(0)


def _try_remove_edge(n, edges, time_limit=0.5):
    """Try to remove one edge while maintaining non-2-colorability."""
    start = time.time()
    m = len(edges)
    indices = list(range(m))
    random.shuffle(indices)
    for i in indices:
        if time.time() - start > time_limit:
            break
        new_edges = edges[:i] + edges[i+1:]
        if not _is_2_colorable(n, new_edges):
            return new_edges
    return None


def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_result = None
    best_edge_count = 201
    
    def update_best(n, edges):
        nonlocal best_result, best_edge_count
        if len(edges) < best_edge_count:
            best_edge_count = len(edges)
            best_result = {'n': n, 'edges': [sorted(e) for e in edges]}
    
    random.seed(42)
    
    # Try multiple n values and random constructions
    for n in [13, 14, 15, 16, 17, 18, 19, 20]:
        if time.time() - start_time > deadline - 8:
            break
        
        for trial in range(8):
            if time.time() - start_time > deadline - 8:
                break
            
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Start with moderate density
            for num_edges in [max(20, n * 2), max(30, n * 3), max(40, n * 4)]:
                if num_edges > len(all_edges):
                    continue
                if time.time() - start_time > deadline - 5:
                    break
                
                edges = [list(e) for e in all_edges[:num_edges]]
                
                # If 2-colorable, add edges until not
                if _is_2_colorable(n, edges):
                    for e in all_edges[num_edges:]:
                        if time.time() - start_time > deadline - 5:
                            break
                        edges.append(list(e))
                        if not _is_2_colorable(n, edges):
                            break
                    if _is_2_colorable(n, edges):
                        continue
                
                # Greedy removal
                improved = True
                while improved and time.time() - start_time < deadline - 5:
                    improved = False
                    new_edges = _try_remove_edge(n, edges, time_limit=0.3)
                    if new_edges is not None:
                        edges = new_edges
                        improved = True
                
                update_best(n, edges)
    
    # Also try K9^5 based approach
    if time.time() - start_time < deadline - 5:
        n = 9
        all_k9 = list(combinations(range(9), 5))
        random.shuffle(all_k9)
        edges = [list(e) for e in all_k9]
        
        improved = True
        while improved and time.time() - start_time < deadline - 3:
            improved = False
            new_edges = _try_remove_edge(n, edges, time_limit=0.5)
            if new_edges is not None:
                edges = new_edges
                improved = True
        
        update_best(n, edges)
    
    if best_result is not None:
        return best_result
    
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
