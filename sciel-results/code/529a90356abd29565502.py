import time
import random
from itertools import combinations

def is_2_colorable(n, edges, node_limit=10000):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    Returns None if node_limit is exceeded (unknown).
    """
    edge_sets = [set(e) for e in edges]
    colors = [-1] * n
    
    vertex_to_edges = {i: [] for i in range(n)}
    for idx, e in enumerate(edge_sets):
        for v in e:
            vertex_to_edges[v].append(idx)
    
    nodes_visited = 0
    
    def propagate(colors, changed):
        nonlocal nodes_visited
        queue = list(changed)
        while queue:
            v = queue.pop()
            c = colors[v]
            for e_idx in vertex_to_edges[v]:
                e = edge_sets[e_idx]
                count_0 = 0
                count_1 = 0
                unknowns = []
                for u in e:
                    cu = colors[u]
                    if cu == 0:
                        count_0 += 1
                    elif cu == 1:
                        count_1 += 1
                    else:
                        unknowns.append(u)
                
                if count_0 == 5 or count_1 == 5:
                    return False
                
                if len(unknowns) == 1:
                    u = unknowns[0]
                    if count_0 == 4:
                        if colors[u] == 1:
                            return False
                        if colors[u] == -1:
                            colors[u] = 1
                            queue.append(u)
                    elif count_1 == 4:
                        if colors[u] == 0:
                            return False
                        if colors[u] == -1:
                            colors[u] = 0
                            queue.append(u)
        return True

    def backtrack():
        nonlocal nodes_visited
        nodes_visited += 1
        if nodes_visited > node_limit:
            return None  # Timeout/limit exceeded
        
        unassigned = [i for i in range(n) if colors[i] == -1]
        if not unassigned:
            return True
        
        def score(v):
            s = 0
            for e_idx in vertex_to_edges[v]:
                e = edge_sets[e_idx]
                for u in e:
                    if u != v and colors[u] != -1:
                        s += 1
            return s
        
        v = max(unassigned, key=score)
        
        for c in [0, 1]:
            old_colors = colors.copy()
            colors[v] = c
            if propagate(colors, [v]):
                res = backtrack()
                if res is True:
                    return True
                if res is None:
                    return None
            colors[:] = old_colors
            
        return False

    if not propagate(colors, []):
        return False
    
    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try to find a non-2-colorable hypergraph
    # Start with n=12, m=51 (known minimum)
    # If not found, try other configurations
    
    # We want to MINIMIZE edges.
    # So we try small m first.
    
    # However, if m < 51, it's likely 2-colorable.
    # We will search m from 30 to 60.
    
    for n in [12, 13]:
        for m in range(30, 61):
            if time.time() > deadline - 10.0:
                break
            
            # Increase node limit for larger m (harder to prove non-colorable?)
            # Actually, if it's non-colorable, propagation might find contradiction fast.
            # If it's colorable, we need to find a coloring.
            # For m < 51, likely colorable -> solver finds coloring fast.
            # For m >= 51, likely non-colorable -> solver finds contradiction fast?
            
            max_trials = 100
            for _ in range(max_trials):
                if time.time() > deadline - 10.0:
                    break
                
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = all_edges[:m]
                
                res = is_2_colorable(n, edges, node_limit=5000)
                if res is False:
                    # Found a non-2-colorable hypergraph
                    return {'n': n, 'edges': [list(e) for e in edges]}
                # If res is True or None, continue
    
    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
