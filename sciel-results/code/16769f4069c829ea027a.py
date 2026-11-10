"""K3,3-free graph on 20 vertices - maximize edges via greedy + local search."""
import time

def construct():
    deadline = time.time() + 135.0
    n = 20
    
    adj = [0] * n
    
    # Start with K_{2,18}
    for a in range(2):
        for b in range(2, n):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    edge_set = set()
    for a in range(2):
        for b in range(2, n):
            edge_set.add((a, b))
    
    def can_add(u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        new_u = adj[u] | (1 << v)
        new_v = adj[v] | (1 << u)
        
        # Check triples involving u
        for x in range(n):
            if x == u:
                continue
            nx = adj[x]
            for y in range(x + 1, n):
                if y == u:
                    continue
                if (new_u & nx & adj[y]).bit_count() >= 3:
                    return False
        
        # Check triples involving v  
        for x in range(n):
            if x == v:
                continue
            nx = adj[x]
            for y in range(x + 1, n):
                if y == v:
                    continue
                if (new_v & nx & adj[y]).bit_count() >= 3:
                    return False
        
        return True
    
    # Greedy: add edges in order of least constraint
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edge_set:
                continue
            if can_add(u, v):
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    # Local search
    for _ in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    continue
                edge_set.discard((i, j))
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                
                found = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if (u, v) in edge_set:
                            continue
                        if can_add(u, v):
                            edge_set.add((u, v))
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            found = True
                            improved = True
                            break
                    if found:
                        break
                if not found:
                    edge_set.add((i, j))
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if not improved:
            break
    
    return [[u, v] for (u, v) in edge_set]
