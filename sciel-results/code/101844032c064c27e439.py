import random
import time
import math

def count_k4(adj, n):
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                # Count edges in subgraph induced by common
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    # w is in common, so w is neighbor of u and v.
                    # Check if w is connected to other nodes in common
                    # We need to count edges (w, x) where x in common, x > w to avoid double counting?
                    # Or just count all edges and divide by 2?
                    # Let's count edges (w, x) with x in common and x > w.
                    edges_in_common = 0
                    # Iterate neighbors of w that are in common
                    neighbors_in_common = adj[w] & common
                    # Count bits in neighbors_in_common that are > w?
                    # Or just count all and divide by 2 at the end for the pair (u,v)?
                    # No, each K4 is counted 6 times (once per edge).
                    # For a fixed edge (u,v), the number of K4s containing it is the number of edges in the induced subgraph on common neighbors.
                    # Let E_common be the number of edges in G[common].
                    # Then we add E_common to count.
                    # Total K4s = (1/6) * sum_{(u,v) in E} E_{N(u) \cap N(v)}.
                    
                    # Efficiently count edges in G[common]:
                    # Sum over w in common of (deg(w) in common) / 2
                    deg_sum = 0
                    cc = common
                    while cc:
                        lsb2 = cc & -cc
                        w2 = lsb2.bit_length() - 1
                        deg_sum += (adj[w2] & common).bit_count()
                        cc ^= lsb2
                    edges_in_common = deg_sum // 2
                    count += edges_in_common
                    c ^= lsb
    return count // 6

def greedy_independent_set_size(adj, n):
    remaining = (1 << n) - 1
    size = 0
    while remaining:
        min_deg = n
        best_v = -1
        r = remaining
        while r:
            lsb = r & -r
            v = lsb.bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            r ^= lsb
        size += 1
        remaining &= ~adj[best_v]
        remaining &= ~(1 << best_v)
    return size

def has_independent_set_6_exact(adj, n):
    comp = [0] * n
    for i in range(n):
        mask = (1 << n) - 1
        mask ^= (1 << i)
        comp[i] = mask & ~adj[i]
        
    def dfs(candidates_mask, size):
        if size == 0:
            return True
        if candidates_mask == 0:
            return False
        if size == 1:
            return True
        if bin(candidates_mask).count('1') < size:
            return False
        
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        new_cands = candidates_mask & comp[v]
        if dfs(new_cands, size - 1):
            return True
        
        candidates_mask ^= lsb
        if dfs(candidates_mask, size):
            return True
        
        return False

    full_mask = (1 << n) - 1
    return dfs(full_mask, 6)

def construct():
    random.seed(42)
    start_time = time.time()
    time_limit = 110.0
    
    # Try n=36 first, then 35
    for n in [36, 35]:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_k4(adj, n) + 10 * max(0, greedy_independent_set_size(adj, n) - 5)
        
        T = 1.0
        T_min = 0.001
        cooling = 0.999
        
        for step in range(50000):
            if time.time() - start_time > time_limit:
                break
            
            if current_cost == 0:
                if not has_independent_set_6_exact(adj, n):
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                new_cost = count_k4(adj, n) + 10 * max(0, greedy_independent_set_size(adj, n) - 5)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_cost = count_k4(adj, n) + 10 * max(0, greedy_independent_set_size(adj, n) - 5)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_cost = new_cost
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            T *= cooling
            if T < T_min:
                T = T_min

    # Fallback
    return {'n': 1, 'edges': []}
