import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin for 90s limit
    
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Fast validity check using bitmasks
    def is_valid(adj):
        degs = [len(adj[i]) for i in range(n)]
        # Build bitmask for each degree value
        deg_masks = {}
        for i in range(n):
            d = degs[i]
            if d not in deg_masks:
                deg_masks[d] = 0
            deg_masks[d] |= (1 << i)
        
        # Precompute neighbor masks
        nb_masks = []
        for u in range(n):
            m = 0
            for x in adj[u]:
                m |= (1 << x)
            nb_masks.append(m)
        
        for u in range(n):
            nu_mask = nb_masks[u]
            for v in adj[u]:
                if v <= u:
                    continue
                nv_mask = nb_masks[v]
                
                # s must be in nu_mask, not v
                s_mask = nu_mask & ~(1 << v)
                # t must be in nv_mask, not u
                t_mask = nv_mask & ~(1 << u)
                
                # For each degree group with mask m, check if there exist
                # s in (s_mask & m) and t in (t_mask & m) with s != t
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
                        if s_in & (s_in - 1) == 0:
                            continue
                    return False
        return True
    
    # Start from K_{7,8} bipartite graph (provably valid, 56 edges)
    def make_bipartite_base():
        adj = [set() for _ in range(n)]
        left = list(range(7))
        right = list(range(7, 15))
        for u in left:
            for v in right:
                adj[u].add(v)
                adj[v].add(u)
        return adj
    
    # Local search: try to add intra-part edges
    def local_search_add(adj, max_time):
        adj = [set(x) for x in adj]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try adding all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > max_time:
                        return adj
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        if is_valid(adj):
                            improved = True
                        else:
                            adj[u].discard(v)
                            adj[v].discard(u)
        return adj
    
    # Local search: try swapping edges (remove one, add another)
    def local_search_swap(adj, max_time):
        adj = [set(x) for x in adj]
        current_count = sum(len(adj[i]) for i in range(n)) // 2
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try removing an edge and adding another
            edges_to_remove = [(u, v) for u in range(n) for v in adj[u] if u < v]
            random.shuffle(edges_to_remove)
            for (u1, v1) in edges_to_remove:
                if time.time() > max_time:
                    return adj
                # Remove edge (u1, v1)
                adj[u1].discard(v1)
                adj[v1].discard(u1)
                
                # Try adding a new edge
                for u2 in range(n):
                    for v2 in range(u2 + 1, n):
                        if time.time() > max_time:
                            adj[u1].add(v1)
                            adj[v1].add(u1)
                            return adj
                        if (u2, v2) == (u1, v1):
                            continue
                        if v2 not in adj[u2]:
                            adj[u2].add(v2)
                            adj[v2].add(u2)
                            if is_valid(adj):
                                improved = True
                                # Keep this swap
                                break
                            else:
                                adj[u2].discard(v2)
                                adj[v2].discard(u2)
                    if improved:
                        break
                if not improved:
                    # Restore the removed edge
                    adj[u1].add(v1)
                    adj[v1].add(u1)
        return adj
    
    # Multiple restarts with different strategies
    random.seed(42)
    best_adj = None
    best_count = 0
    
    # First, start from bipartite base and do local search
    base_adj = make_bipartite_base()
    base_count = sum(len(base_adj[i]) for i in range(n)) // 2
    best_adj = [set(x) for x in base_adj]
    best_count = base_count
    
    if is_valid(base_adj):
        improved_adj = local_search_add(base_adj, deadline - 20)
        improved_count = sum(len(improved_adj[i]) for i in range(n)) // 2
        if improved_count > best_count:
            best_adj = improved_adj
            best_count = improved_count
    
    # Try multiple random restarts with greedy + local search
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random initial graph: start with bipartite and randomly add some edges
        adj = [set(x) for x in base_adj]
        
        # Randomly try to add some intra-part edges
        left = list(range(7))
        right = list(range(7, 15))
        random.shuffle(left)
        random.shuffle(right)
        
        added = 0
        for u in left:
            for v in left:
                if u < v and random.random() < 0.3:
                    adj[u].add(v)
                    adj[v].add(u)
                    added += 1
        for u in right:
            for v in right:
                if u < v and random.random() < 0.3:
                    adj[u].add(v)
                    adj[v].add(u)
                    added += 1
        
        # Check validity, if invalid, remove problematic edges
        if not is_valid(adj):
            # Remove edges until valid
            changed = True
            while changed and time.time() < deadline - 10:
                changed = False
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                random.shuffle(edges_list)
                for (u, v) in edges_list:
                    if time.time() > deadline - 10:
                        break
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if is_valid(adj):
                        changed = True
                        break
        
        # Local search to add more edges
        adj = local_search_add(adj, deadline - 10)
        cnt = sum(len(adj[i]) for i in range(n)) // 2
        
        if cnt > best_count:
            best_adj = [set(x) for x in adj]
            best_count = cnt
    
    # Final improvement: try more swaps
    if best_adj and time.time() < deadline - 5:
        improved = local_search_swap(best_adj, deadline - 5)
        cnt = sum(len(improved[i]) for i in range(n)) // 2
        if cnt > best_count:
            best_adj = improved
            best_count = cnt
    
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
    
    # Verify final result
    if best_adj:
        if not is_valid(best_adj):
            # Fallback to bipartite
            edges = []
            left = list(range(7))
            right = list(range(7, 15))
            for u in left:
                for v in right:
                    edges.append([u, v])
    
    # Ensure types are correct
    result = []
    for e in edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
