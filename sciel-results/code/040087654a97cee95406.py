import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Fallback: 30-vertex graph made of 10 triangles
    def _get_30_vertex_edges():
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a, b])
            edges.append([a, c])
            edges.append([b, c])
        return edges

    def _creates_c4(adj, n, u, v):
        # Check if adding edge (u,v) creates a C4
        # A C4 is created if there exist x, y such that:
        # x is adjacent to both u and v, AND y is adjacent to both u and v,
        # with x != y. Actually, the standard check: 
        # If u and v have >= 2 common neighbors, adding edge (u,v) doesn't create C4 by itself,
        # but the graph already has a C4. Wait, the condition for C4-freeness is:
        # No two vertices have 2+ common neighbors.
        # So if we add (u,v), we must ensure:
        # 1. u and v have < 2 common neighbors in the current graph (otherwise C4 exists already)
        # 2. For any neighbor w of u, w and v must not have 2+ common neighbors after adding (u,v)
        #    Since (u,v) is new, u is now a common neighbor of w and v (if w~u and v~u... wait v~u is the new edge)
        #    So common neighbors of w and v now include u (since w~u and u~v).
        #    They must not have any OTHER common neighbor.
        #    So: (N(w) ∩ N(v)) must be empty (in the graph BEFORE adding (u,v)), 
        #    because u is not yet in N(v) but will be, and u IS in N(w).
        #    Actually: after adding (u,v), N(v) gains u. 
        #    Common neighbors of w and v = (N(w) ∩ N(v)_old) ∪ (N(w) ∩ {u}) if u in N(w)
        #    Since w is neighbor of u, u in N(w). So u is a common neighbor of w and v after adding.
        #    We need |N(w) ∩ N(v)_new| <= 1.
        #    N(v)_new = N(v)_old ∪ {u}
        #    So N(w) ∩ N(v)_new = (N(w) ∩ N(v)_old) ∪ (N(w) ∩ {u})
        #    Since w ~ u, u ∈ N(w), so N(w) ∩ {u} = {u}.
        #    So we need |N(w) ∩ N(v)_old ∪ {u}| <= 1
        #    Since u ∉ N(v)_old (edge not added yet), and u ∉ N(w) ∩ N(v)_old (as u not in N(v)_old),
        #    these are disjoint. So we need |N(w) ∩ N(v)_old| + 1 <= 1, i.e., |N(w) ∩ N(v)_old| = 0.
        #    Similarly for neighbors of v.
        
        # Check 1: u and v must have < 2 common neighbors
        common_uv = adj[u] & adj[v]
        if len(common_uv) >= 2:
            return True
        
        # Check 2: For each neighbor w of u, w and v must have 0 common neighbors in current graph
        for w in adj[u]:
            if w == v:
                continue
            if adj[w] & adj[v]:
                return True
        
        # Check 3: For each neighbor w of v, w and u must have 0 common neighbors in current graph
        for w in adj[v]:
            if w == u:
                continue
            if adj[w] & adj[u]:
                return True
        
        return False

    def _is_c4_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                if len(adj[u] & adj[v]) >= 2:
                    return False
        return True

    def _max_independent_set_bound(adj, n, target):
        # Try to find an independent set of size 'target'
        # Use greedy with backtracking
        if n < target:
            return False
        
        # Greedy coloring bound: if chromatic number > target, no independent set of size target
        # Simpler: use a simple backtracking search
        
        vertices = list(range(n))
        random.shuffle(vertices)
        
        found = [False]
        
        def backtrack(remaining, current_set):
            if found[0]:
                return
            if len(current_set) == target:
                found[0] = True
                return
            if len(remaining) < target - len(current_set):
                return
            if time.time() > deadline:
                return
            
            # Pick a vertex
            v = remaining[0]
            rem_rest = remaining[1:]
            
            # Option 1: Include v
            new_remaining = [x for x in rem_rest if x not in adj[v]]
            backtrack(new_remaining, current_set + [v])
            
            # Option 2: Exclude v
            if not found[0]:
                backtrack(rem_rest, current_set)
        
        backtrack(vertices, [])
        return found[0]

    def _verify_graph(n, edges):
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check C4-free
        if not _is_c4_free(adj, n):
            return False
        
        # Check no independent set of size 11
        if _max_independent_set_bound(adj, n, 11):
            return False
        
        return True

    # Try to construct a C4-free graph on 39 vertices with alpha < 11
    # Use random greedy construction with restarts
    
    best_result = None
    best_n = 0
    
    for n in [39, 38, 37, 36, 35]:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 12345 + n)
            
            # Start with empty graph
            adj = [set() for _ in range(n)]
            edges = []
            
            # Generate all possible edges
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            # Greedy: add edges if C4-free
            for u, v in all_edges:
                if not _creates_c4(adj, n, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Check independence number
            if not _max_independent_set_bound(adj, n, 11):
                # Valid!
                best_result = {'n': n, 'edges': edges}
                best_n = n
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to 30-vertex graph
    edges = _get_30_vertex_edges()
    return {'n': 30, 'edges': edges}
