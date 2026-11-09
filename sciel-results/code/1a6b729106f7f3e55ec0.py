import random
import itertools

def is_valid(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Check all pairs with same degree
    for s in range(n):
        for t in range(s + 1, n):
            if degrees[s] != degrees[t]:
                continue
            
            # Check if there is a simple path of length 3 between s and t
            # s - u - v - t
            found = False
            for u in adj[s]:
                if u == t:
                    continue
                for v in adj[u]:
                    if v == s or v == t or v == u:
                        continue
                    if t in adj[v]:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def count_conflicts(edges, n):
    """
    Returns a list of conflicts. Each conflict is a tuple (s, t, path_edges)
    where path_edges is a set of edges in the path s-u-v-t.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    conflicts = []
    
    for s in range(n):
        for t in range(s + 1, n):
            if degrees[s] != degrees[t]:
                continue
            
            # Find one witness path
            for u in adj[s]:
                if u == t:
                    continue
                for v in adj[u]:
                    if v == s or v == t or v == u:
                        continue
                    if t in adj[v]:
                        # Path s-u-v-t
                        # Edges: (s,u), (u,v), (v,t)
                        e1 = (min(s,u), max(s,u))
                        e2 = (min(u,v), max(u,v))
                        e3 = (min(v,t), max(v,t))
                        conflicts.append((s, t, {e1, e2, e3}))
                        break
                if conflicts and conflicts[-1][0] == s and conflicts[-1][1] == t:
                    break
    return conflicts

def construct():
    n = 15
    # Start with K_15
    edges = [[u, v] for u in range(n) for v in range(u + 1, n)]
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    # Greedy removal based on conflict resolution
    # To avoid getting stuck in a bad local minimum, we can add some randomness
    # or use a smarter scoring.
    
    # Heuristic:
    # 1. Find all conflicts.
    # 2. For each edge, count how many conflicts it is part of.
    # 3. Also consider the "degree impact": removing an edge (u,v) reduces deg(u) and deg(v).
    #    This might resolve conflicts where u or v were the equal-degree endpoints.
    #    So we score an edge by:
    #    - Number of paths it is in (direct conflict resolution)
    #    - Plus a bonus if it is incident to a vertex involved in many conflicts.
    
    for _ in range(1000): # Max iterations
        conflicts = count_conflicts(edges, n)
        if not conflicts:
            break
        
        # Score each edge in the current graph
        edge_scores = {}
        for e in edge_set:
            edge_scores[e] = 0
            
        # Direct path involvement
        for s, t, path_edges in conflicts:
            for e in path_edges:
                edge_scores[e] += 10 # High weight for being in a witness path
        
        # Indirect involvement (incident to conflict endpoints)
        # If we remove edge (u,v), deg(u) and deg(v) change.
        # This might break equalities for u or v with other nodes.
        # Let's count how many conflicts involve u or v.
        vertex_conflict_count = [0] * n
        for s, t, path_edges in conflicts:
            vertex_conflict_count[s] += 1
            vertex_conflict_count[t] += 1
            
        for e in edge_set:
            u, v = e
            # Bonus for removing edges incident to highly conflicted vertices
            edge_scores[e] += vertex_conflict_count[u] + vertex_conflict_count[v]
            
        # Find the edge with the highest score
        if not edge_scores:
            break
            
        best_edge = max(edge_scores, key=edge_scores.get)
        
        # Remove it
        edges.remove([best_edge[0], best_edge[1]])
        edge_set.remove(best_edge)
        
    return edges
