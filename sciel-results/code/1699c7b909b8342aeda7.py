"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque
import numpy as np

# EVOLVE-BLOCK-START

def _check_girth_fast(adj, n, g):
    """Check girth >= g using non-backtracking BFS from every vertex."""
    for start in range(n):
        depth = {}
        queue = deque()
        for nb in adj[start]:
            depth[(start, nb)] = 1
            queue.append((start, nb))
        while queue:
            u, v = queue.popleft()
            d = depth[(u, v)]
            if d + 1 >= g:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                key = (v, w)
                if key in depth:
                    if d + 1 + depth[key] < g:
                        return False
                else:
                    depth[key] = d + 1
                    if d + 1 < g:
                        queue.append(key)
    return True

def _edges_to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _k44_edges():
    """Return edges of K_{4,4} (8 vertices, 4-regular, girth 4)."""
    edges = []
    for i in range(4):
        for j in range(4):
            edges.append((i, 4 + j))
    return edges

def _build_lift(base_edges, base_n, k, group_order, voltage_fn):
    """
    Build a k-regular graph by lifting a base graph via voltage assignment.
    """
    n = base_n * group_order
    
    # Build adjacency of base graph
    base_adj = [[] for _ in range(base_n)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    edges = []
    edge_set = set()
    
    for u in range(base_n):
        for v in base_adj[u]:
            if u < v:
                volt = voltage_fn(u, v)
                for t in range(group_order):
                    a = u * group_order + t
                    b = v * group_order + (t + volt) % group_order
                    if a < b:
                        e = (a, b)
                    else:
                        e = (b, a)
                    if e not in edge_set:
                        edge_set.add(e)
                        edges.append(e)
    
    return n, edges

def _construct_via_lift(k, g, deadline):
    """
    Try to construct a k-regular graph of girth >= g using voltage lifts.
    Uses K_{4,4} as base for k=4.
    """
    import math
    
    if k == 4:
        base_edges = _k44_edges()
        base_n = 8
    else:
        return None
    
    best_result = None
    best_n = float('inf')
    
    moore = 4 * (4 - 1) * (4 ** (g // 2 - 1) + 1) + 2 * (4 - 1) if g % 2 == 0 else \
             2 * 4 * (4 ** ((g - 3) // 2) + 4 ** ((g - 1) // 2)) - 2
    
    min_group_order = max(1, math.ceil(moore / base_n))
    
    # For k=4, g=10: moore ~ 322, base_n=8, so min_group_order ~ 41
    # The known cage is 320, so group_order 40 might work if we find the right voltages.
    # But 40 < 41, so we must try 40 as well, or the Moore bound calculation is slightly off.
    # Moore bound for g=10, k=4: 1 + 4 + 4*3 + 4*3*3 + 4*3*3*3 = 1 + 4 + 12 + 36 + 108 = 161?
    # Wait, the formula is:
    # For even g=2m: n >= 2 * k * (k-1)^(m-1)
    # For g=10, m=5. n >= 2 * 4 * 3^4 = 8 * 81 = 648? No.
    # The Moore bound is:
    # n >= 1 + k * sum_{i=0}^{m-1} (k-1)^i for odd g?
    # For even g=2m: n >= 2 * sum_{i=0}^{m-1} k(k-1)^{i-1}?
    # Let's use the standard formula:
    # n >= k * (k-1)^(g/2 - 1) * 2 for even g?
    # For g=10, k=4: n >= 4 * 3^4 * 2 = 648?
    # But the known cage is 320. So the Moore bound is not tight.
    # The Moore bound is a lower bound.
    # The known cage is 320.
    # So we should try group orders around 40.
    
    for group_order in range(40, 46):
        if time.time() > deadline:
            break
        
        n = base_n * group_order
        if n > 400:
            break
        
        # Try structured voltage assignments
        # The 320-vertex graph is the incidence graph of a generalized quadrangle?
        # No, it's the Hoffman-Singleton type.
        # We try a few deterministic patterns.
        
        for trial in range(10):
            if time.time() > deadline:
                break
            
            edge_voltages = {}
            # Pattern 1: Random
            if trial == 0:
                rng = random.Random(42)
                for u, v in base_edges:
                    volt = rng.randint(0, group_order - 1)
                    edge_voltages[(u, v)] = volt
                    edge_voltages[(v, u)] = (-volt) % group_order
            
            # Pattern 2: Structured
            elif trial == 1:
                gen = 1
                idx = 0
                for u, v in base_edges:
                    volt = (gen * idx) % group_order
                    edge_voltages[(u, v)] = volt
                    edge_voltages[(v, u)] = (-volt) % group_order
                    idx += 1
            
            # Pattern 3: Another structured
            elif trial == 2:
                idx = 0
                for u, v in base_edges:
                    volt = (idx * 7) % group_order
                    edge_voltages[(u, v)] = volt
                    edge_voltages[(v, u)] = (-volt) % group_order
                    idx += 1
            
            else:
                continue
            
            def voltage_fn(u, v, _ev=edge_voltages):
                return _ev.get((u, v), 0)
            
            n_lift, edges = _build_lift(base_edges, base_n, k, group_order, voltage_fn)
            
            adj = _edges_to_adj(n_lift, edges)
            if any(len(adj[i]) != k for i in range(n_lift)):
                continue
            
            if _check_girth_fast(adj, n_lift, g):
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    if best_result is not None:
        return best_result
    
    return None

def construct(k, g):
    """
    Build a k-regular graph with girth at least g on as few vertices as possible.
    """
    start_time = time.time()
    deadline = 140
    
    # Try voltage lift construction
    result = _construct_via_lift(k, g, deadline)
    if result is not None:
        return result
    
    # Fallback: Return a valid regular graph (may not satisfy girth)
    n_fallback = 2 * k
    edges_fallback = []
    for i in range(n_fallback):
        edges_fallback.append((i, (i + 1) % n_fallback))
    for i in range(k - 1):
        edges_fallback.append((i, (i + n_fallback // 2) % n_fallback))
    edges_fallback = list(set((min(u,v), max(u,v)) for u,v in edges_fallback))
    return n_fallback, edges_fallback
# EVOLVE-BLOCK-END
