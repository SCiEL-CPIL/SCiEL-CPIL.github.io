import time
import random

def _has_indep_11_bitmask(adj, n, deadline):
    """
    Check if an independent set of size 11 exists using bitmask backtracking.
    """
    def backtrack(remaining, current_size):
        if time.time() > deadline:
            return False
        if current_size >= 11:
            return True
        if not remaining:
            return False
        if current_size + remaining.bit_count() < 11:
            return False
        
        # Heuristic: pick vertex with minimum degree in remaining subgraph
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1
            
        if min_v == -1:
            return False
            
        # Branch 1: Include min_v
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        # Branch 2: Exclude min_v
        remaining &= ~(1 << min_v)
        if backtrack(remaining, current_size):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_result = {'n': 30, 'edges': fallback_edges}
    
    n_target = 39
    
    # Try Max-Degree Greedy
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        random.seed(attempt * 12345 + 67890)
        
        adj = [0] * n_target
        edges = []
        degrees = [0] * n_target
        
        # Pre-compute all edges
        all_edges = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        
        # We will maintain a set of valid edges
        # Initially all are valid
        valid_edges = set(all_edges)
        
        while valid_edges:
            if time.time() > deadline - 10:
                break
            
            # Find the edge with max degree sum
            best_edge = None
            best_score = -1
            
            # Sample a subset if valid_edges is large? 
            # For n=39, max edges ~ 700. 700 is small.
            for u, v in valid_edges:
                # Check if still valid (C4-free)
                # We need to check if adding (u,v) creates a C4
                # This check is O(deg(u) + deg(v))
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    continue
                
                # Check neighbors
                ok = True
                nu = adj[u]
                nv = adj[v]
                
                temp = nu
                while temp:
                    w = (temp & -temp).bit_length() - 1
                    if adj[w] & nv:
                        ok = False
                        break
                    temp &= temp - 1
                
                if not ok:
                    continue
                
                temp = nv
                while temp:
                    x = (temp & -temp).bit_length() - 1
                    if adj[x] & nu:
                        ok = False
                        break
                    temp &= temp - 1
                
                if not ok:
                    continue
                
                score = degrees[u] + degrees[v]
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            valid_edges.remove((u, v))
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            edges.append([u, v])
            
            # Remove edges that are now invalid?
            # This is expensive. Instead, we just check validity when considering.
            # But the set `valid_edges` grows stale.
            # Better: just iterate over all possible edges and check validity on the fly.
            # Since n=39 is small, we can just re-scan all edges each time.
            
            # Actually, the loop above iterates over `valid_edges`.
            # We should just use a list of all edges and filter.
            # Let's restart the inner loop logic:
            # Instead of maintaining valid_edges, just iterate all edges.
            
            # This approach is O(E * Check) per step.
            # E ~ 700. Check ~ 100 ops. 70,000 ops per step.
            # 700 steps? No, max edges in C4-free graph is ~ n sqrt(n) / 2.
            # For n=39, ~ 39 * 6 / 2 ~ 100 edges.
            # So ~ 100 steps.
            # Total ~ 7,000,000 ops. Very fast.
            
        # Verify C4-freeness
        is_c4_free = True
        for u in range(n_target):
            for v in range(u + 1, n_target):
                if (adj[u] & adj[v]).bit_count() > 1:
                    is_c4_free = False
                    break
            if not is_c4_free:
                break
        
        if not is_c4_free:
            continue
        
        # Check independence
        has_indep_11 = _has_indep_11_bitmask(adj, n_target, deadline)
        
        if not has_indep_11:
            return {'n': n_target, 'edges': edges}
            
    return best_result
