import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # 5s margin for 120s limit

    def time_check():
        return time.time() - start_time

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def max_independent_set_size(n, adj_masks):
        best = [0]
        
        def pick_vertex(candidates):
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                neigh = adj_masks[v] & candidates
                deg = neigh.bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c &= c - 1
            return best_v

        def dfs(candidates, size):
            if time_check() > DEADLINE - 2:
                return
            
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            if size + candidates.bit_count() <= best[0]:
                return
            
            v = pick_vertex(candidates)
            if v == -1:
                return
                
            lsb = 1 << v
            
            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)
            
            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def get_max_independent_set(n, adj_masks):
        best_set = [None]
        
        def pick_vertex(candidates):
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                neigh = adj_masks[v] & candidates
                deg = neigh.bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c &= c - 1
            return best_v

        def dfs(candidates, size, current_set):
            if time_check() > DEADLINE - 2:
                return
            
            if not candidates:
                if best_set[0] is None or size > len(best_set[0]):
                    best_set[0] = current_set[:]
                return
            
            if best_set[0] is not None and size + candidates.bit_count() <= len(best_set[0]):
                return
            
            v = pick_vertex(candidates)
            if v == -1:
                return
                
            lsb = 1 << v
            
            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            current_set.append(v)
            dfs(new_candidates, size + 1, current_set)
            current_set.pop()
            
            # Exclude v
            dfs(candidates & ~lsb, size, current_set)

        dfs((1 << n) - 1, 0, [])
        return best_set[0] if best_set[0] else []

    def try_find_graph(n, seed):
        random.seed(seed)
        
        edges = []
        adj_masks = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if adj_masks[u] & (1 << v):
                continue
            if adj_masks[u] & adj_masks[v]:
                continue
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            edges.append([u, v])
            
        max_iters = 1000
        for i in range(max_iters):
            if time_check() > DEADLINE - 5:
                return None
            
            alpha = max_independent_set_size(n, adj_masks)
            if alpha <= 9:
                return {'n': n, 'edges': edges}
                
            indep_set = get_max_independent_set(n, adj_masks)
            
            added = False
            random.shuffle(indep_set)
            for idx_u in range(len(indep_set)):
                for idx_v in range(idx_u + 1, len(indep_set)):
                    u = indep_set[idx_u]
                    v = indep_set[idx_v]
                    
                    if adj_masks[u] & (1 << v):
                        continue
                        
                    if adj_masks[u] & adj_masks[v]:
                        continue
                    
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.append([u, v])
                    added = True
                    break
                if added:
                    break
            
            if not added:
                if edges:
                    e_idx = random.randint(0, len(edges) - 1)
                    u, v = edges[e_idx]
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)
                    edges.pop(e_idx)
                else:
                    return None
                    
        return None

    for seed in range(2000):
        if time_check() > DEADLINE - 10:
            break
        result = try_find_graph(39, seed)
        if result:
            return result
            
    for seed in range(2000):
        if time_check() > DEADLINE - 10:
            break
        result = try_find_graph(38, seed)
        if result:
            return result
            
    # Fallback And(10)
    k = 10
    n = 3 * k - 2
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if 1 <= diff <= k - 1:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
