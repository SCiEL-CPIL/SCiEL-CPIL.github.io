import random
import time

def construct():
    start_time = time.time()
    timeout = 115
    
    def make_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def has_c5_bitset(n, adj):
        """Check C5 using bitsets. For each u, for each pair (a,d) in N(u), check path a-b-c-d in G-{u}."""
        for u in range(n):
            neighbors = [v for v in range(n) if v != u and (adj[u] >> v) & 1]
            if len(neighbors) < 2:
                continue
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    a = neighbors[i]
                    d = neighbors[j]
                    # Need path a-b-c-d in G-{u}: a~b, b~c, c~d, all distinct from u,a,d and from each other
                    # b in N(a) \ {u, a, d}, c in N(d) \ {u, a, b, d}, and b~c
                    for b in range(n):
                        if b == u or b == a or b == d:
                            continue
                        if not ((adj[a] >> b) & 1):
                            continue
                        # c in N(d) & N(b), excluding u,a,b,d
                        common = adj[d] & adj[b]
                        mask = ~((1 << u) | (1 << a) | (1 << b) | (1 << d))
                        if common & mask:
                            return True
        return False
    
    def find_large_is(n, adj, target=9):
        """Try to find an independent set of size >= target. Return it or None."""
        # Greedy with randomization
        for _ in range(50):
            remaining = list(range(n))
            current = []
            while remaining:
                # Pick vertex with min degree in remaining (greedy)
                min_deg = float('inf')
                min_v = -1
                for v in remaining:
                    deg = 0
                    for w in remaining:
                        if w != v and (adj[v] >> w) & 1:
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                if min_v == -1:
                    break
                current.append(min_v)
                if len(current) >= target:
                    return current
                # Remove min_v and its neighbors
                remaining = [v for v in remaining if v != min_v and not ((adj[min_v] >> v) & 1)]
        return None
    
    def alpha_lower_bound(n, adj):
        """Get a lower bound on alpha via greedy."""
        remaining = list(range(n))
        current = []
        while remaining:
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = 0
                for w in remaining:
                    if w != v and (adj[v] >> w) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            if min_v == -1:
                break
            current.append(min_v)
            remaining = [v for v in remaining if v != min_v and not ((adj[min_v] >> v) & 1)]
        return len(current)
    
    # Strategy: Start with a C5-free graph and add edges to reduce alpha
    # Use complete bipartite as base, then add intra-part edges carefully
    
    for n in [32, 33, 34, 35]:
        if time.time() - start_time > timeout - 10:
            break
        
        n1 = n // 2
        n2 = n - n1
        
        # Start with complete bipartite
        edges = []
        for i in range(n1):
            for j in range(n1, n):
                edges.append([i, j])
        
        adj = make_adj(n, edges)
        
        # Add intra-part edges to reduce alpha
        for iter_num in range(500):
            if time.time() - start_time > timeout:
                break
            
            # Check if alpha < 9
            lb = alpha_lower_bound(n, adj)
            if lb < 9:
                # Verify more carefully
                is_found = find_large_is(n, adj, 9)
                if is_found is None:
                    return {'n': n, 'edges': edges}
                # If we found one, continue adding edges
            else:
                is_found = find_large_is(n, adj, 9)
                if is_found is None:
                    return {'n': n, 'edges': edges}
            
            # Try to add an edge within a part
            added = False
            for _ in range(10):
                part_idx = random.randint(0, 1)
                if part_idx == 0:
                    part = list(range(n1))
                else:
                    part = list(range(n1, n))
                if len(part) < 2:
                    continue
                u, v = random.sample(part, 2)
                if (adj[u] >> v) & 1:
                    continue
                
                # Try adding edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if has_c5_bitset(n, adj):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    continue
                
                edges.append([u, v])
                added = True
                break
            
            if not added:
                break
    
    # Try smaller n if needed
    for n in range(31, 27, -1):
        if time.time() - start_time > timeout:
            break
        
        n1 = n // 2
        n2 = n - n1
        
        edges = []
        for i in range(n1):
            for j in range(n1, n):
                edges.append([i, j])
        
        adj = make_adj(n, edges)
        
        for iter_num in range(800):
            if time.time() - start_time > timeout:
                break
            
            is_found = find_large_is(n, adj, 9)
            if is_found is None:
                return {'n': n, 'edges': edges}
            
            added = False
            for _ in range(15):
                part_idx = random.randint(0, 1)
                if part_idx == 0:
                    part = list(range(n1))
                else:
                    part = list(range(n1, n))
                if len(part) < 2:
                    continue
                u, v = random.sample(part, 2)
                if (adj[u] >> v) & 1:
                    continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if has_c5_bitset(n, adj):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    continue
                
                edges.append([u, v])
                added = True
                break
            
            if not added:
                break
    
    # Fallback
    return {'n': 28, 'edges': [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]}
