import random
import itertools

def construct():
    n = 16
    vertices = list(range(n))
    subsets = list(itertools.combinations(vertices, 8))
    
    # Pre-compute subsets containing each pair (u, v)
    # subsets_with_pair[(u, v)] = list of subsets
    subsets_with_pair = {}
    for sub in subsets:
        for i in range(8):
            for j in range(i+1, 8):
                u, v = sub[i], sub[j]
                if u > v: u, v = v, u
                key = (u, v)
                if key not in subsets_with_pair:
                    subsets_with_pair[key] = []
                subsets_with_pair[key].append(sub)
                
    def is_q3(sub, adj):
        mask = 0
        for v in sub:
            mask |= (1 << v)
        
        # Check degrees
        for v in sub:
            deg = bin(adj[v] & mask).count('1')
            if deg != 3:
                return False
        
        # Check bipartite
        # Simple BFS
        color = {}
        for v in sub:
            if v not in color:
                color[v] = 0
                stack = [v]
                while stack:
                    u = stack.pop()
                    # Iterate neighbors in sub
                    # adj[u] & mask
                    neighbors = adj[u] & mask
                    while neighbors:
                        lsb = neighbors & -neighbors
                        w = lsb.bit_length() - 1
                        neighbors ^= lsb
                        if w not in color:
                            color[w] = color[u] ^ 1
                            stack.append(w)
                        else:
                            if color[w] == color[u]:
                                return False
        return True

    def count_q3(adj):
        count = 0
        for sub in subsets:
            if is_q3(sub, adj):
                count += 1
        return count

    def num_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    best_m = 0
    best_adj_final = [0]*n
    
    # Use time limit logic implicitly by limiting restarts
    # 120s limit.
    # Let's run 20 restarts to be safe.
    
    for restart in range(20):
        adj = [0] * n
        # Random init
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Remove Q3s
        c = count_q3(adj)
        while c > 0:
            # Find first Q3
            found = False
            for sub in subsets:
                if is_q3(sub, adj):
                    # Remove random edge from sub
                    edges_in_sub = []
                    mask = 0
                    for v in sub: mask |= (1 << v)
                    for u in sub:
                        neighbors = adj[u] & mask
                        while neighbors:
                            lsb = neighbors & -neighbors
                            w = lsb.bit_length() - 1
                            neighbors ^= lsb
                            if u < w:
                                edges_in_sub.append((u, w))
                    if edges_in_sub:
                        u, v = random.choice(edges_in_sub)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    found = True
                    break
            if not found: break
            c = count_q3(adj)
            
        # Local Search: Add edges
        # Potential edges
        potential = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
        random.shuffle(potential)
        
        added = True
        while added:
            added = False
            for u, v in potential:
                if (adj[u] >> v) & 1: continue # Already added in previous iteration? No, we don't update potential list dynamically efficiently.
                # Actually, if we add (u,v), it's no longer potential.
                # But we just iterate through the static list. If it's already in adj, skip.
                
                # Try add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check Q3s containing (u,v)
                has_q3 = False
                key = (min(u,v), max(u,v))
                if key in subsets_with_pair:
                    for sub in subsets_with_pair[key]:
                        if is_q3(sub, adj):
                            has_q3 = True
                            break
                            
                if has_q3:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    added = True
                    # Remove from potential? Not necessary, we check adj state.
                    # But to save time, we could mark it.
                    pass
                    
        m = num_edges(adj)
        if m > best_m:
            best_m = m
            best_adj_final = adj[:]
            
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_final[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
