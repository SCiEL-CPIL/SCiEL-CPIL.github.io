import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = []
    best_count = 0
    deadline = time.time() + 50.0
    
    # Helper to check validity of a specific edge addition given current state
    def can_add(u, v, adj, deg):
        # Simulate degree change
        deg[u] += 1
        deg[v] += 1
        
        # 1. Check new paths created by edge (u, v)
        # Path s-u-v-t
        for s in adj[u]:
            if s == v: continue
            for t in adj[v]:
                if t == u or t == s: continue
                if deg[s] == deg[t]:
                    deg[u] -= 1
                    deg[v] -= 1
                    return False
        
        # 2. Check existing paths where u is an endpoint
        # Path u-a-b-c (u is start) or c-b-a-u (u is end)
        # In both cases, we check if deg(u) == deg(other_end)
        # The other_end is a neighbor of a neighbor of a neighbor of u?
        # Path: u - a - b - c. 
        # a in N(u), b in N(a)\{u}, c in N(b)\{a}
        for a in adj[u]:
            for b in adj[a]:
                if b == u: continue
                for c in adj[b]:
                    if c == a: continue
                    # c is the other endpoint
                    if deg[u] == deg[c]:
                        deg[u] -= 1
                        deg[v] -= 1
                        return False
        
        # 3. Check existing paths where v is an endpoint
        for a in adj[v]:
            for b in adj[a]:
                if b == v: continue
                for c in adj[b]:
                    if c == a: continue
                    if deg[v] == deg[c]:
                        deg[u] -= 1
                        deg[v] -= 1
                        return False
        
        # If we reach here, it's safe
        deg[u] -= 1
        deg[v] -= 1
        return True

    # Main loop
    while time.time() < deadline:
        random.shuffle(all_pairs)
        adj = [[] for _ in range(n)]
        deg = [0] * n
        current_edges = []
        
        for u, v in all_pairs:
            if can_add(u, v, adj, deg):
                current_edges.append((u, v))
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
        # Local Search: Try to add more edges
        # We can try all missing edges
        current_set = set(current_edges)
        missing = [p for p in all_pairs if p not in current_set]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline: break
            if can_add(u, v, adj, deg):
                current_edges.append((u, v))
                adj[u].append(v)
                adj[v].append(u)
                deg[u] += 1
                deg[v] += 1
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = [list(e) for e in current_edges]

    return best_edges
