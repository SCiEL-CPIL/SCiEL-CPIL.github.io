import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def compute_alpha(adj_mask, n):
        """Compute maximum independent set size using Branch and Bound with good heuristics."""
        best = [0]
        
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            if size + candidates.bit_count() <= best[0]:
                return
            
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) maintains C4-free property."""
        common = adj_mask[u] & adj_mask[v]
        return common.bit_count() <= 1

    def try_extend(start_n, target_n, seed):
        """Start with a graph on start_n vertices, try to extend to target_n."""
        random.seed(seed)
        
        # Start with 10 disjoint triangles
        adj_mask = [0] * start_n
        for t in range(start_n // 3):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            adj_mask[a] |= (1 << b) | (1 << c)
            adj_mask[b] |= (1 << a) | (1 << c)
            adj_mask[c] |= (1 << a) | (1 << b)
        
        # Extend one vertex at a time
        for new_v in range(start_n, target_n):
            if time.time() > deadline:
                return None
            adj_mask.append(0)
            
            # Try to connect new_v to existing vertices in random order
            candidates = list(range(new_v))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                if can_add_edge(adj_mask, new_v, j):
                    adj_mask[new_v] |= (1 << j)
                    adj_mask[j] |= (1 << new_v)
        
        if time.time() > deadline - 5:
            return None
        alpha = compute_alpha(adj_mask, target_n)
        if alpha < 11:
            return adj_mask
        return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to extend from 30 vertices to larger n
    for target_n in range(43, 30, -1):
        if time.time() > deadline - 10:
            break
        
        for seed in range(100):
            if time.time() > deadline - 10:
                break
                
            adj_mask = try_extend(30, target_n, seed)
            if adj_mask is not None:
                edges = []
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': target_n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
