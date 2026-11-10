"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115  # safety margin before 150s kill
    n = 40
    
    # Build adjacency as bitmasks for fast operations
    adj_mask = [0] * n
    
    def add_edge(u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    def degree(u):
        return adj_mask[u].bit_count()
    
    def total_edges():
        return sum(adj_mask[u].bit_count() for u in range(n)) // 2
    
    def has_c6_fast(u, v):
        """Check if adding edge (u,v) creates a C6 using bitmasks.
        
        Cycle C6: u - a - b - c - d - v - u
        Need path of length 5 from u to v in G - {u,v edge}.
        """
        nu = adj_mask[u] & ~(1 << v)
        nv = adj_mask[v] & ~(1 << u)
        
        a = nu
        while a:
            la = a & (-a)
            a ^= la
            ai = la.bit_length() - 1
            
            nb = adj_mask[ai] & ~(1 << u) & ~(1 << v) & ~(1 << ai)
            b = nb
            while b:
                lb = b & (-b)
                b ^= lb
                bi = lb.bit_length() - 1
                
                nc = adj_mask[bi] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << bi)
                c = nc
                while c:
                    lc = c & (-c)
                    c ^= lc
                    ci = lc.bit_length() - 1
                    
                    # Precompute intersection once
                    inter = adj_mask[ci] & nv
                    if inter:
                        # Check exclusions: d != u, ai, bi, ci
                        inter &= ~(1 << u) & ~(1 << ai) & ~(1 << bi) & ~(1 << ci)
                        if inter:
                            return True
        
        return False
    
    def get_all_edges(mask):
        """Get list of edges from adjacency mask."""
        edges = []
        for u in range(n):
            m = mask[u]
            while m:
                lb = m & (-m)
                m ^= lb
                v = lb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
        return edges
    
    def greedy_add_pass(mask, time_limit):
        """Try to add as many edges as possible to mask within time_limit."""
        adj_mask[:] = list(mask)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        
        for (u, v) in all_pairs:
            if time.time() > time_limit:
                break
            if (1 << v) & adj_mask[u]:
                continue
            if not has_c6_fast(u, v):
                add_edge(u, v)
        
        return list(adj_mask), total_edges()
    
    def local_search(initial_mask, time_limit):
        """Perform local search starting from a given state."""
        adj_mask[:] = list(initial_mask)
        
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        best_adj_mask = list(adj_mask)
        best_count = total_edges()
        
        while time.time() < time_limit - 1:
            random.shuffle(all_pairs)
            added_any = False
            
            for (u, v) in all_pairs:
                if time.time() > time_limit - 1:
                    break
                if (1 << v) & adj_mask[u]:
                    continue
                if not has_c6_fast(u, v):
                    add_edge(u, v)
                    added_any = True
            
            current_count = total_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj_mask = list(adj_mask)
            
            if not added_any:
                # Perturbation: remove 2-5 random edges to escape local optimum
                if time.time() > time_limit - 2:
                    break
                
                current_edges = get_all_edges(adj_mask)
                
                if not current_edges:
                    break
                
                # Remove 2-5 random edges with bias toward more removals
                k = random.randint(2, 5)
                k = min(k, len(current_edges))
                edges_to_remove = random.sample(current_edges, k)
                for (u, v) in edges_to_remove:
                    remove_edge(u, v)
                
                # Try to re-add greedily
                random.shuffle(all_pairs)
                for (u, v) in all_pairs:
                    if time.time() > time_limit - 1:
                        break
                    if (1 << v) & adj_mask[u]:
                        continue
                    if not has_c6_fast(u, v):
                        add_edge(u, v)
                
                current_count = total_edges()
                if current_count > best_count:
                    best_count = current_count
                    best_adj_mask = list(adj_mask)
        
        return best_adj_mask, best_count

    # Initial state: 8 disjoint K5s
    init_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                init_adj[5 * k + a] |= (1 << (5 * k + b))
                init_adj[5 * k + b] |= (1 << (5 * k + a))
    
    best_overall_mask = list(init_adj)
    best_overall_count = total_edges()
    
    # Phase 1: Quick greedy build from initial state
    t1_end = time.time() + 5
    mask1, count1 = greedy_add_pass(init_adj, t1_end)
    if count1 > best_overall_count:
        best_overall_count = count1
        best_overall_mask = mask1
    
    # Phase 2: Multiple local search runs with perturbations
    remaining_time = deadline - time.time() - 5
    # Allocate time adaptively: give more to later restarts
    num_restarts = 4
    # Weights for time allocation (later restarts get more time)
    weights = [0.15, 0.2, 0.3, 0.35]
    total_weight = sum(weights)
    time_allocs = [w / total_weight * remaining_time for w in weights]
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            start_mask = list(best_overall_mask)
        else:
            start_mask = list(best_overall_mask)
            # Remove a few random edges to escape local optimum
            current_edges = get_all_edges(start_mask)
            if current_edges:
                edges_to_remove = random.sample(current_edges, min(4, len(current_edges)))
                for (u, v) in edges_to_remove:
                    start_mask[u] &= ~(1 << v)
                    start_mask[v] &= ~(1 << u)
        
        t_end = time.time() + time_allocs[r]
        if t_end > deadline - 5:
            t_end = deadline - 5
        if t_end < time.time() + 1:
            break
            
        mask, count = local_search(start_mask, t_end)
        
        if count > best_overall_count:
            best_overall_count = count
            best_overall_mask = mask
    
    # Phase 3: Final polish pass with remaining time
    if time.time() < deadline - 3:
        final_mask, final_count = greedy_add_pass(best_overall_mask, deadline - 3)
        if final_count > best_overall_count:
            best_overall_count = final_count
            best_overall_mask = final_mask

    # Build final edge list from the best state
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (1 << v) & best_overall_mask[u]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
