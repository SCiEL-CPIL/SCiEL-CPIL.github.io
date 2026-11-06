import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def greedy_alpha_upper_bound(adj_mask, n):
        """
        Fast upper bound on independence number using greedy coloring of complement.
        If we can color the complement with k colors, then alpha(G) <= k.
        Actually, chromatic number of complement >= alpha(G).
        We use greedy coloring to get an upper bound on chi(complement), which is an upper bound on... 
        Wait, let me think again.
        
        Actually: alpha(G) = chi(complement of G) is NOT true. 
        alpha(G) is the size of the largest independent set in G, which equals the size of the largest clique in complement of G.
        
        Better approach: Use a greedy maximal independent set as a lower bound, and use branch-and-bound for exact.
        
        For a fast check: if we can find an independent set of size 11, alpha >= 11 (fail).
        Otherwise, we need to prove alpha < 11.
        
        Fast approach: Try to find independent set of size 11 using greedy/randomized search.
        If found, fail. If not found after many tries, do exact B&B.
        """
        # First, try to find an independent set of size 11 quickly
        # If found, return 11 (or more)
        
        # Greedy maximal independent set
        def greedy_maximal_independent_set(adj_mask, n, seed=0):
            rng = random.Random(seed)
            candidates = list(range(n))
            rng.shuffle(candidates)
            chosen = []
            chosen_set = set()
            for v in candidates:
                if v in chosen_set:
                    continue
                # Check if v is non-adjacent to all chosen
                ok = True
                for u in chosen:
                    if (adj_mask[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    chosen.append(v)
                    chosen_set.add(v)
            return chosen
        
        # Try multiple greedy MIS to see if any has size >= 11
        for seed in range(20):
            mis = greedy_maximal_independent_set(adj_mask, n, seed)
            if len(mis) >= 11:
                return 11  # alpha >= 11
        
        # Now do exact B&B to prove alpha < 11
        # Use branch and bound with early termination at 11
        best = [0]
        
        def dfs(candidates, size):
            if size >= 11:
                return True  # Found independent set of size 11
            if candidates == 0:
                return False
            # Upper bound
            if size + candidates.bit_count() < 11:
                return False
            
            # Pick vertex with max degree in candidate subgraph
            c = candidates
            max_deg = -1
            chosen_v = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen_v = v
            
            if chosen_v == -1:
                return False
                
            # Branch 1: Include chosen_v
            new_candidates = candidates & ~(adj_mask[chosen_v] | (1 << chosen_v))
            if dfs(new_candidates, size + 1):
                return True
            
            # Branch 2: Exclude chosen_v
            new_candidates = candidates & ~(1 << chosen_v)
            if dfs(new_candidates, size):
                return True
            
            return False

        all_candidates = (1 << n) - 1
        if dfs(all_candidates, 0):
            return 11
        else:
            return 10  # alpha < 11, exact value not needed

    def try_find_graph(n, time_limit):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses greedy random construction with alpha-aware edge selection.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        for i in range(n):
            # Try to connect i to previous vertices
            # Prioritize connections that reduce independence
            
            candidates = list(range(i))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                # Check if we can add (i, j)
                can_add = True
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        if time.time() > deadline:
            return None
            
        # Check alpha
        alpha = greedy_alpha_upper_bound(adj_mask, n)
        if alpha < 11:
            return adj_mask
        else:
            return None

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges
    
    # Try to find a larger graph
    # Focus more attempts on higher n values
    for n in range(43, 30, -1):
        if time.time() > deadline - 3:
            break
            
        # More attempts for higher n
        num_attempts = max(5, 30 - (n - 30) * 2)
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
                
            random.seed(attempt * 7919 + n * 104729)
            adj_mask = try_find_graph(n, 10)
            if adj_mask:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
