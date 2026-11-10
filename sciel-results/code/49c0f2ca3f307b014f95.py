import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k != 3:
        return _lnw_construct(k, g)
        
    # Try voltage graph construction first (structural change from LUW)
    best_result = None
    
    # Try various base graphs and voltage assignments
    base_graphs = [
        # (edges, n_base) - small cubic graphs
        _get_k4(),           # K4: 4 vertices, girth 3
        _get_k5_minus(),     # K5 minus an edge won't be cubic
        _get_octahedron(),   # Octahedron: 6 vertices, girth 3
        _get_prism6(),       # Prism over C3: 6 vertices
        _get_cubic12(),      # A 12-vertex cubic graph
    ]
    
    for base_n, base_edges in base_graphs:
        if time.time() > deadline - 30:
            break
        # Try different voltage group sizes
        for m in range(2, 100):
            if time.time() > deadline - 30:
                break
            # Try random voltage assignments
            for trial in range(20):
                if time.time() > deadline - 30:
                    break
                n, edges = _voltage_lift(base_n, base_edges, m, seed=trial*1000+m)
                if n < 200 or n > 5000:
                    continue
                adj = _edges_to_adj(edges, n)
                girth = _calculate_girth_fast(adj, n, g)
                if girth >= g:
                    if best_result is None or n < best_result[0]:
                        best_result = (n, edges)
                    break  # Found valid for this m, try smaller m
            if best_result is not None:
                break  # Found valid, stop trying larger m for this base
    
    if best_result is not None:
        return best_result
    
    # Fallback to LUW
    for d in range(6, 16):
        if time.time() > deadline - 10:
            break
        try:
            n, edges = _lnw_construct_params(3, d)
        except Exception:
            continue
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth_fast(adj, n, g)
        if girth >= g:
            return n, edges
    
    return _lnw_construct(3, g)
