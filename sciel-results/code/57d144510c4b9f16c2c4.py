import math
import random
import time

def construct():
    """
    Finds a C4-free graph on n vertices with minimum degree at least n - 39.
    Maximizes n.
    """
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        # Check theoretical bound
        # e_max ~ n/4 * (1 + sqrt(4n-3))
        e_max = (n / 4.0) * (1.0 + math.sqrt(4 * n - 3))
        e_min_needed = (n * min_deg) / 2.0
        if e_min_needed > e_max:
            continue
            
        # Try to construct
        start_time = time.time()
        graph = find_graph(n, min_deg, start_time)
        
        if graph is not None:
            return graph
            
    # Fallback
    return {'n': 39, 'edges': []}

def find_graph(n, min_deg, start_time):
    """
    Backtracking search for a C4-free graph on n vertices with min degree >= min_deg.
    """
    adj = [set() for _ in range(n)]
    deg = [0] * n
    edges = []
    
    # Max degree allowed to keep search space small
    # We need min_deg, so we can allow up to min_deg + 2
    max_deg = min_deg + 2
    
    def creates_c4(u, v):
        # Check if adding (u,v) creates a C4
        # C4 is u-w-x-v
        # w in adj[u], x in adj[v], w-x in adj
        # Optimization: check intersection of adj[u] and adj[v] neighbors?
        # Path length 3: u->w->x->v
        # For each w in adj[u]:
        #   For each x in adj[v]:
        #     if w == x: continue (triangle, not C4, but wait, if w==x, it's a triangle u-w-v. Triangles are allowed in C4-free? Yes. C4 is a 4-cycle. Triangles don't violate C4-free.)
        #     if x in adj[w]: return True
        # Note: if w==x, it's a triangle. Triangles are fine.
        # So we just need to check if there is a path of length 3.
        
        # To be faster, we can check:
        # N(u) and N(v)
        # We need w in N(u) and x in N(v) such that (w,x) in E.
        # This is equivalent to:
        # There exists w in N(u) such that N(w) intersects N(v).
        
        for w in adj[u]:
            if w == v:
                continue
            # Check if any neighbor of w is in adj[v]
            # adj[w] & adj[v]
            if adj[w] & adj[v]:
                return True
        return False

    def is_valid_state():
        if min(deg) >= min_deg:
            return True
        return False

    def backtrack():
        if time.time() - start_time > 110: # 10s per n, total limit 120s
            return None
            
        if is_valid_state():
            return edges.copy()
            
        # Find vertex with minimum degree
        min_d = min(deg)
        if min_d >= min_deg:
            return edges.copy()
            
        # Candidates for u
        # Pick u with min degree
        u_candidates = [i for i in range(n) if deg[i] == min_d]
        random.shuffle(u_candidates)
        
        for u in u_candidates:
            # Find valid v for u
            # v must have deg < max_deg
            # (u,v) not in adj
            # not creates_c4(u,v)
            v_candidates = []
            for v in range(n):
                if v == u: continue
                if deg[v] >= max_deg: continue
                if v in adj[u]: continue
                if creates_c4(u, v): continue
                v_candidates.append(v)
                
            if not v_candidates:
                continue
                
            random.shuffle(v_candidates)
            
            for v in v_candidates:
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
                
                res = backtrack()
                if res is not None:
                    return res
                    
                # Undo
                edges.pop()
                deg[u] -= 1
                deg[v] -= 1
                adj[u].remove(v)
                adj[v].remove(u)
                
        return None

    return backtrack()
