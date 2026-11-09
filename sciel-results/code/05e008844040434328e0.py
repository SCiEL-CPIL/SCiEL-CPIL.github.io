import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin (90s limit)
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible = len(all_edges)  # 105
    
    # Fast validity check using degree groups
    def is_valid(adj):
        degs = [len(adj[i]) for i in range(n)]
        # Group vertices by degree
        deg_groups = defaultdict(list)
        for i in range(n):
            deg_groups[degs[i]].append(i)
        
        # For each edge (u,v), check all s in N(u)\{v} and t in N(v)\{u}
        # where s != t and deg(s) == deg(t)
        for u in range(n):
            nu = adj[u]
            for v in nu:
                if v <= u:
                    continue
                nv = adj[v]
                # For each degree group, check if there exist s in nu\{v} and t in nv\{u}
                # with s != t and both in the same degree group
                for d, group in deg_groups.items():
                    if len(group) < 2:
                        continue
                    # Find s in nu (s != v) and t in nv (t != u) with s != t, both in group
                    s_candidates = [x for x in group if x in nu and x != v]
                    t_candidates = [x for x in group if x in nv and x != u]
                    if not s_candidates or not t_candidates:
                        continue
                    # Check if there exist s in s_candidates and t in t_candidates with s != t
                    if len(s_candidates) >= 2 or len(t_candidates) >= 2:
                        # If either set has >= 2 elements, we can always find s != t
                        # unless s_candidates == t_candidates and both have exactly 1 element
                        if len(s_candidates) >= 2:
                            return False
                        if len(t_candidates) >= 2:
                            return False
                        # Both have exactly 1 element
                        if s_candidates[0] != t_candidates[0]:
                            return False
                    else:
                        # Both have exactly 1 element
                        if s_candidates[0] != t_candidates[0]:
                            return False
        return True
    
    # Even faster: use bitmasks for degree groups
    def is_valid_fast(adj):
        degs = [len(adj[i]) for i in range(n)]
        # Build bitmask for each degree value
        deg_masks = {}
        for i in range(n):
            d = degs[i]
            if d not in deg_masks:
                deg_masks[d] = 0
            deg_masks[d] |= (1 << i)
        
        for u in range(n):
            nu_mask = 0
            for x in adj[u]:
                nu_mask |= (1 << x)
            for v in adj[u]:
                if v <= u:
                    continue
                nv_mask = 0
                for x in adj[v]:
                    nv_mask |= (1 << x)
                
                # s must be in nu_mask, not v, not u (since s != u for simple path)
                # Actually s is in adj[u], s != v. s can be u? No, u is not in adj[u].
                # s_candidates: in nu_mask, not v
                s_mask = nu_mask & ~(1 << v)
                # t_candidates: in nv_mask, not u
                t_mask = nv_mask & ~(1 << u)
                
                # For each degree group with mask m, check if (s_mask & m) and (t_mask & m)
                # have elements that are not equal
                for d, m in deg_masks.items():
                    if m & ~(m - 1) == m:  # only one bit set
                        continue
                    s_in = s_mask & m
                    t_in = t_mask & m
                    if not s_in or not t_in:
                        continue
                    # Check if there exist s in s_in and t in t_in with s != t
                    # This fails only if s_in == t_in and s_in has exactly one bit
                    if s_in == t_in:
                        # Only one bit in s_in
                        if s_in & (s_in - 1) == 0:
                            continue
                    return False
        return True
    
    # Greedy construction with multiple restarts
    random.seed(12345)
    best_edge_count = 0
    best_adj = None
    
    def try_greedy(edge_order, max_time):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_order:
            if time.time() > max_time:
                break
            adj[u].add(v)
            adj[v].add(u)
            if not is_valid_fast(adj):
                adj[u].discard(v)
                adj[v].discard(u)
        cnt = sum(len(adj[i]) for i in range(n)) // 2
        return adj, cnt
    
    def try_remove_then_add(adj_init, max_time):
        """Start from a valid graph, try to add edges, if stuck remove some and retry"""
        adj = [set(x) for x in adj_init]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for (u, v) in all_edges:
                if time.time() > max_time:
                    break
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    if is_valid_fast(adj):
                        improved = True
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)
        cnt = sum(len(adj[i]) for i in range(n)) // 2
        return adj, cnt
    
    # Multiple restarts with different edge orderings
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        edge_order = all_edges[:]
        random.shuffle(edge_order)
        
        adj, cnt = try_greedy(edge_order, deadline - 5)
        if cnt > best_edge_count:
            best_edge_count = cnt
            best_adj = [set(x) for x in adj]
        
        # Try to improve by adding more edges
        if time.time() < deadline - 3:
            adj2, cnt2 = try_remove_then_add(adj, deadline - 3)
            if cnt2 > best_edge_count:
                best_edge_count = cnt2
                best_adj = [set(x) for x in adj2]
    
    # Final improvement pass
    if best_adj and time.time() < deadline - 2:
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 2:
                    break
                if v not in best_adj[u]:
                    best_adj[u].add(v)
                    best_adj[v].add(u)
                    if is_valid_fast(best_adj):
                        improved = True
                    else:
                        best_adj[u].discard(v)
                        best_adj[v].discard(u)
            if not improved:
                break
    
    # Build result
    if best_adj is None:
        # Fallback: star graph
        edges = [[0, v] for v in range(1, n)]
    else:
        edges = []
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    edges.append([u, v])
    
    # Verify
    if best_adj:
        if not is_valid_fast(best_adj):
            edges = [[0, v] for v in range(1, n)]
    
    # Ensure types
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
