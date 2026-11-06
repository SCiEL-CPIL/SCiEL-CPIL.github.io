"""Optimized program for finding few-edge non-2-colorable 5-uniform hypergraphs."""
import time
import random
from itertools import combinations

def _is_not_2_colorable(n, edges, timeout=8.0):
    """Check if hypergraph is NOT 2-colorable using optimized backtracking with unit propagation."""
    start_time = time.time()
    edges_set = [tuple(sorted(e)) for e in edges]
    m = len(edges_set)
    
    # Precompute edge membership for fast propagation
    # For each vertex, which edges contain it
    vertex_to_edges = {}
    for i, e in enumerate(edges_set):
        for v in e:
            if v not in vertex_to_edges:
                vertex_to_edges[v] = []
            vertex_to_edges[v].append(i)
    
    def propagate(assign):
        """Unit propagation: if an edge has 4 vertices same color, 5th must be opposite."""
        changed = True
        while changed:
            changed = False
            for ei, e in enumerate(edges_set):
                colors = [assign.get(v) for v in e]
                num_0 = sum(1 for c in colors if c == 0)
                num_1 = sum(1 for c in colors if c == 1)
                num_un = sum(1 for c in colors if c is None)
                
                if num_0 == 5 or num_1 == 5:
                    return False
                if num_un == 0:
                    continue
                if num_0 == 4 and num_un == 1:
                    for v in e:
                        if assign.get(v) is None:
                            assign[v] = 1
                            changed = True
                elif num_1 == 4 and num_un == 1:
                    for v in e:
                        if assign.get(v) is None:
                            assign[v] = 0
                            changed = True
        return True
    
    def backtrack(assign):
        if time.time() - start_time > timeout:
            return None  # timeout
        
        if not propagate(assign):
            return False
        
        # Check if all vertices assigned
        unassigned = [v for v in range(n) if v not in assign]
        if not unassigned:
            # Verify all edges have both colors
            for e in edges_set:
                colors = [assign[v] for v in e]
                if len(set(colors)) < 2:
                    return False
            return True
        
        # Pick vertex with most constraints (heuristic)
        v = min(unassigned, key=lambda x: -len(vertex_to_edges.get(x, [])))
        
        # Try color 0
        assign[v] = 0
        result = backtrack(assign)
        if result is not None:
            return result
        del assign[v]
        
        # Try color 1
        assign[v] = 1
        result = backtrack(assign)
        if result is not None:
            return result
        del assign[v]
        
        return None  # no valid coloring from this state
    
    result = backtrack({})
    if result is None and time.time() - start_time <= timeout:
        return True  # no coloring found, so not 2-colorable
    return False  # found coloring or timeout

def _greedy_removal(n, edges, timeout=60.0):
    """Try to remove edges one at a time while maintaining non-2-colorability."""
    start_time = time.time()
    edges = [tuple(sorted(e)) for e in edges]
    m = len(edges)
    
    # Try removing each edge and check if still not 2-colorable
    removed = 0
    changed = True
    while changed and time.time() - start_time < timeout:
        changed = False
        for i in range(len(edges)):
            if time.time() - start_time > timeout:
                break
            # Try removing edge i
            candidate = edges[:i] + edges[i+1:]
            if _is_not_2_colorable(n, candidate, timeout=3.0):
                edges = candidate
                removed += 1
                changed = True
                break
    
    return [list(e) for e in edges]

def _find_small_non_2colorable(max_edges=55, n_start=8, n_end=15, timeout_per=4.0):
    """Search for small non-2-colorable 5-uniform hypergraphs with greedy removal."""
    best = None
    start_time = time.time()
    
    for n in range(n_start, n_end + 1):
        if time.time() - start_time > 100:
            break
        max_possible = len(list(combinations(range(n), 5)))
        
        for e_count in range(29, min(max_edges + 1, max_possible + 1)):
            if time.time() - start_time > 100:
                break
            
            all_edges = list(combinations(range(n), 5))
            attempts = 0
            max_attempts = 30
            while attempts < max_attempts and time.time() - start_time < 100:
                random.shuffle(all_edges)
                subset = all_edges[:e_count]
                subset_list = [list(x) for x in subset]
                
                if _is_not_2_colorable(n, subset_list, timeout_per):
                    # Found a non-2-colorable hypergraph; try to minimize
                    reduced = _greedy_removal(n, subset_list, timeout=20.0)
                    if best is None or len(reduced) < len(best['edges']):
                        best = {'n': n, 'edges': reduced}
                    break
                attempts += 1
        
        if best is not None and len(best['edges']) <= 32:
            break
    
    return best

def construct():
    # Try to find a small non-2-colorable hypergraph
    result = _find_small_non_2colorable(max_edges=55, n_start=8, n_end=14, timeout_per=3.0)
    
    if result is not None:
        return result
    
    # Fallback: complete hypergraph on 9 vertices (known non-2-colorable)
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
