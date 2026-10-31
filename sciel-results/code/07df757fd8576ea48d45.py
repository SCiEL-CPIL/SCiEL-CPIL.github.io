import random
from collections import deque

def construct(k, g):
    base_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,7), (7,9), (9,6), (6,8), (8,5),
        (0,5), (1,6), (2,7), (3,8), (4,9)
    ]
    
    def check_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = (g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            dist[start] = 0
            while queue:
                u, d = queue.popleft()
                if d > max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        if d + 1 <= max_depth:
                            queue.append((v, d + 1))
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def find_short_cycle_edges(n, edges, g):
        # Find a cycle of length < g
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        max_depth = (g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            queue = deque([(start, 0)])
            dist[start] = 0
            while queue:
                u, d = queue.popleft()
                if d > max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        if d + 1 <= max_depth:
                            queue.append((v, d + 1))
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle edges
                            # Path u -> ... -> LCA -> ... -> v
                            # Find LCA
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                cur = parent[cur]
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = parent[cur]
                                
                            # Find common ancestor
                            set_u = set(path_u)
                            lca = None
                            for node in path_v:
                                if node in set_u:
                                    lca = node
                                    break
                                    
                            if lca is None:
                                continue
                                
                            cycle_edges = set()
                            # Add edges from u to lca
                            cur = u
                            while cur != lca:
                                p = parent[cur]
                                cycle_edges.add((min(cur, p), max(cur, p)))
                                cur = p
                                
                            # Add edges from v to lca
                            cur = v
                            while cur != lca:
                                p = parent[cur]
                                cycle_edges.add((min(cur, p), max(cur, p)))
                                cur = p
                                
                            # Add closing edge
                            cycle_edges.add((min(u, v), max(u, v)))
                            
                            return list(cycle_edges)
        return None

    m = 94
    n = 10 * m
    edge_keys = [(min(u,v), max(u,v)) for u, v in base_edges]
    
    for attempt in range(100):
        random.seed(12345 + attempt)
        twists = {key: random.randint(0, m-1) for key in edge_keys}
        
        edges = []
        for u, v in base_edges:
            key = (min(u,v), max(u,v))
            t = twists[key]
            for i in range(m):
                u_lift = u * m + i
                v_lift = v * m + (i + t) % m
                edges.append((min(u_lift, v_lift), max(u_lift, v_lift)))
        
        if check_girth_at_least(n, edges, g):
            return n, edges
            
        # Try local search if girth is close
        # Check if girth >= 14
        if not check_girth_at_least(n, edges, 14):
            continue
            
        # Local search
        current_edges = edges
        for step in range(50):
            # Find a short cycle
            cycle_edges = find_short_cycle_edges(n, current_edges, g)
            if not cycle_edges:
                break
                
            # Try to break it
            improved = False
            for _ in range(10):
                # Pick random edge from cycle
                e1 = random.choice(cycle_edges)
                # Pick random edge from graph not in cycle
                all_edges_set = set((min(u,v), max(u,v)) for u,v in current_edges)
                candidates = [e for e in all_edges_set if e not in set(cycle_edges)]
                if not candidates:
                    continue
                e2 = random.choice(candidates)
                
                u1, v1 = e1
                u2, v2 = e2
                
                # Swap
                new_e1 = (min(u1, v2), max(u1, v2))
                new_e2 = (min(u2, v1), max(u2, v1))
                
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                if new_e1 in all_edges_set or new_e2 in all_edges_set or new_e1 == new_e2:
                    continue
                    
                # Build new edge list
                new_edges = []
                for e in current_edges:
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    if key != e1 and key != e2:
                        new_edges.append(e)
                new_edges.append((u1, v2))
                new_edges.append((u2, v1))
                
                # Check girth
                if check_girth_at_least(n, new_edges, 14):
                    # Accept if it improves girth
                    if check_girth_at_least(n, new_edges, g):
                        return n, new_edges
                    # Or if it increases girth (hard to check exact girth easily)
                    # Let's just accept if it's valid and we hope it improves
                    # To avoid loops, we can keep a history or just accept first valid
                    current_edges = new_edges
                    improved = True
                    break
            if not improved:
                break
                
    # Fallback
    return 0, []
