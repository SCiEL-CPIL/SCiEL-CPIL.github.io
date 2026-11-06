import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            m = mask
            while m:
                lb = m & -m
                j = lb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lb
        return edges

    def find_independent_set_exact(adj, n, target_size=10):
        def search(candidates, depth, current_set):
            if depth == 0:
                return current_set[:]
            if candidates.bit_count() < depth:
                return None
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            new_candidates = candidates & ~adj[v] & ~(1 << v)
            current_set.append(v)
            result = search(new_candidates, depth - 1, current_set)
            if result:
                return result
            current_set.pop()
            return None
        return search((1 << n) - 1, target_size, [])

    def find_independent_set_greedy(adj, n):
        remaining = (1 << n) - 1
        indep = []
        while remaining:
            best_v = -1
            min_deg = n + 1
            m = remaining
            while m:
                lb = m & -m
                v = lb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                m ^= lb
            
            if best_v == -1:
                break
                
            indep.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
            
        return indep

    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        
        # Try a few restarts for this n
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Init: Random Bipartite with random split
            # Shuffle vertices
            verts = list(range(n))
            random.shuffle(verts)
            mid = n // 2
            adj = [0] * n
            for i in range(mid):
                for j in range(mid, n):
                    u = verts[i]
                    v = verts[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Local Search
            local_start = time.time()
            while time.time() - local_start < 2.0:
                # Check validity periodically
                if find_independent_set_exact(adj, n, 10) is None:
                    return {'n': n, 'edges': get_edges_from_adj(adj, n)}
                
                indep = find_independent_set_greedy(adj, n)
                
                if len(indep) < 10:
                    # Double check with exact
                    if find_independent_set_exact(adj, n, 10) is None:
                        return {'n': n, 'edges': get_edges_from_adj(adj, n)}
                    indep = find_independent_set_exact(adj, n, 10)
                
                if indep is None or len(indep) < 2:
                    continue
                    
                u, v = random.sample(indep, 2)
                
                common = adj[u] & adj[v]
                
                if common == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    w_bit = common & -common
                    w = w_bit.bit_length() - 1
                    if random.random() < 0.5:
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                    else:
                        adj[v] &= ~(1 << w)
                        adj[w] &= ~(1 << v)
                    
                    if random.random() < 0.05:
                        ru, rv = random.sample(range(n), 2)
                        if adj[ru] & (1 << rv):
                            adj[ru] &= ~(1 << rv)
                            adj[rv] &= ~(1 << ru)

    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
