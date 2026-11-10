import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (limit is 120s)
    
    # Target n=21. If we fail, we try n=20 (which is known to exist).
    # The task says "Graphs with 20 vertices are known". 
    # If we can't find 21, we should try to construct 20 to get a better score than 1.0 (which is likely n=5 or similar trivial).
    # Actually, the current program returns n=5 as fallback. 
    # We should aim for 21, then 20.
    
    target_ns = [21, 20]
    
    def get_degrees(mask, adj):
        """Return list of degrees for vertices in mask induced by adj"""
        degs = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (adj[idx] & mask).bit_count()
            degs.append(deg)
            m ^= lsb
        return degs

    def is_regular_mask(mask, adj):
        degs = get_degrees(mask, adj)
        if not degs:
            return False
        d0 = degs[0]
        for d in degs[1:]:
            if d != d0:
                return False
        return True

    def count_regular_subsets(adj, n, min_size=6):
        """Count regular induced subgraphs of size >= min_size. 
        For search, we might only count 6 and 7.
        """
        count = 0
        # For search speed, we usually only track 6 and 7 in the cost function
        # but for final validation we check all.
        # Here we implement a general counter for validation.
        for k in range(min_size, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj):
                    count += 1
        return count

    def validate_graph(adj, n):
        """Check if graph has NO regular induced subgraph on 6+ vertices."""
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj):
                    return False
        return True

    def construct_for_n(n):
        # Precompute masks for size 6 and 7 containing each pair?
        # To speed up delta calculation, we precompute lists of masks for each pair (u,v)
        # for sizes 6 and 7.
        
        # masks_6[pair] = list of masks of size 6 containing pair
        # masks_7[pair] = list of masks of size 7 containing pair
        # This is memory heavy but fast.
        # Number of pairs: n*(n-1)/2 ~ 200.
        # Number of masks per pair for size 6: C(n-2, 4). For n=21, C(19,4)=3876.
        # Total masks stored: 200 * 3876 ~ 775k integers. OK.
        
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        masks_6_by_pair = {p: [] for p in pairs}
        masks_7_by_pair = {p: [] for p in pairs}
        
        # Generate all 6-subsets
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            # Add to all pairs in comb
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = comb[i], comb[j]
                    if u > v: u, v = v, u
                    masks_6_by_pair[(u, v)].append(mask)
                    
        # Generate all 7-subsets
        for comb in itertools.combinations(range(n), 7):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            for i in range(7):
                for j in range(i+1, 7):
                    u, v = comb[i], comb[j]
                    if u > v: u, v = v, u
                    masks_7_by_pair[(u, v)].append(mask)
        
        # Simulated Annealing
        best_adj = None
        best_cost = float('inf')
        
        # Initial random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        def calc_cost(adj):
            # Cost = count_6 + 10 * count_7
            # We only track 6 and 7 during search.
            c6 = 0
            c7 = 0
            # We can sum over pairs to avoid double counting?
            # No, a mask is counted once per pair in it? 
            # If we sum counts per pair, a mask of size 6 has 15 pairs.
            # So total count = (sum over pairs of #regular masks containing pair) / 15.
            # This is a valid way to compute total count.
            # Let's just iterate all 6-subsets and 7-subsets for initial cost.
            for comb in itertools.combinations(range(n), 6):
                mask = 0
                for v in comb: mask |= (1 << v)
                if is_regular_mask(mask, adj): c6 += 1
            for comb in itertools.combinations(range(n), 7):
                mask = 0
                for v in comb: mask |= (1 << v)
                if is_regular_mask(mask, adj): c7 += 1
            return c6 + 10 * c7

        current_cost = calc_cost(adj)
        if current_cost == 0:
             if validate_graph(adj, n):
                 return adj
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        max_steps = 20000
        
        steps = 0
        while T > T_min and steps < max_steps:
            if time.time() > deadline:
                break
            
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Calculate delta
            delta_c6 = 0
            delta_c7 = 0
            
            # Check 6-subsets
            for mask in masks_6_by_pair[(u, v)]:
                was_reg = is_regular_mask(mask, adj)
                # Flip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                now_reg = is_regular_mask(mask, adj)
                # Unflip
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                if was_reg and not now_reg:
                    delta_c6 -= 1
                elif not was_reg and now_reg:
                    delta_c6 += 1
            
            # Check 7-subsets
            for mask in masks_7_by_pair[(u, v)]:
                was_reg = is_regular_mask(mask, adj)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                now_reg = is_regular_mask(mask, adj)
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                if was_reg and not now_reg:
                    delta_c7 -= 1
                elif not was_reg and now_reg:
                    delta_c7 += 1
            
            delta_cost = delta_c6 + 10 * delta_c7
            
            if delta_cost < 0:
                # Accept
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_cost += delta_cost
            else:
                # Accept with prob exp(-delta/T)
                if random.random() < pow(2.71828, -delta_cost / T):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj = adj[:]
                
            if current_cost == 0:
                # Check if fully valid (sizes 8+)
                if validate_graph(adj, n):
                    return adj
                
            T *= cooling
            steps += 1
            
        # If we didn't find 0 cost, return the best found so far?
        # Or return None to try next n.
        if best_cost == 0 and best_adj:
             if validate_graph(best_adj, n):
                 return best_adj
                 
        return None

    # Try to find a graph for n=21
    for n in target_ns:
        if time.time() > deadline:
            break
            
        result_adj = construct_for_n(n)
        if result_adj is not None:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if result_adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # Fallback: Return a known valid graph for n=20 if we didn't find it in the loop?
    # The loop should have found it. 
    # If we failed completely, return n=5 empty graph.
    return {'n': 5, 'edges': []}
