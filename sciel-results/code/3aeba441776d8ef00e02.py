import time
import random
import heapq

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append((u, v))
        return adj, edges

    def popcount(x):
        return bin(x).count('1')

    def greedy_partition(n, edges, adj, strategy, rng, deadline):
        """
        Greedy clique partition.
        strategy: 'max_deg', 'min_common', 'random_max'
        """
        if not edges:
            return []
        
        # Copy adj for remaining edges
        rem_adj = adj[:]
        remaining_edges = list(edges)
        
        # Precompute initial degrees
        rem_deg = [popcount(rem_adj[i]) for i in range(n)]
        
        cliques = []
        
        # To speed up, we can use a set for remaining edges for O(1) removal check?
        # But we need to iterate. A list is faster for iteration, set for removal.
        # Since N is small (<=64), we can just filter the list each time?
        # No, that's O(E) per clique. E ~ 1000. Cliques ~ 100-200. 1000*200 = 200k ops. Fine.
        
        while remaining_edges:
            if time.time() > deadline:
                return None
            
            # Select next clique base
            best_score = -1
            best_clique = None
            
            # Strategy selection
            if strategy == 'max_deg':
                # Pick vertex with max degree
                # To avoid bias, we can pick from top K
                # But max degree is a good heuristic
                max_deg = -1
                candidates = []
                for v in range(n):
                    if rem_deg[v] > max_deg:
                        max_deg = rem_deg[v]
                        candidates = [v]
                    elif rem_deg[v] == max_deg:
                        candidates.append(v)
                
                if not candidates:
                    break
                    
                # If too many, sample
                if len(candidates) > 10:
                    start_v = rng.choice(candidates)
                else:
                    start_v = candidates[rng.randint(0, len(candidates)-1)]
                
                clique = [start_v]
                common = rem_adj[start_v]
                
            elif strategy == 'min_common':
                # Pick edge with min common neighbors
                # Sample if too many edges
                sample_size = min(20, len(remaining_edges))
                if len(remaining_edges) > sample_size:
                    sample = rng.sample(remaining_edges, sample_size)
                else:
                    sample = remaining_edges
                
                min_common = float('inf')
                best_u, best_v = -1, -1
                for u, v in sample:
                    # Check if edge still exists (it should, but good for safety if we don't remove properly)
                    # We are removing edges, so remaining_edges is valid
                    c = rem_adj[u] & rem_adj[v]
                    cnt = popcount(c)
                    if cnt < min_common:
                        min_common = cnt
                        best_u, best_v = u, v
                        
                if best_u == -1:
                    break
                    
                clique = [best_u, best_v]
                common = rem_adj[best_u] & rem_adj[best_v]
                
            else: # 'random'
                u, v = rng.choice(remaining_edges)
                clique = [u, v]
                common = rem_adj[u] & rem_adj[v]

            # Expand clique
            # Greedy: add vertex in common with max degree in common
            added = True
            while added:
                added = False
                if time.time() > deadline:
                    return None
                
                # Find best vertex in common
                best_w = -1
                best_w_deg = -1
                cm = common
                while cm:
                    w = (cm & -cm).bit_length() - 1
                    cm &= cm - 1
                    if w in clique:
                        continue
                    # Degree of w within the common set
                    # Actually, we want to maximize the number of edges covered.
                    # Adding w covers edges (w, x) for x in clique.
                    # The "value" of adding w is the number of remaining edges incident to w that are inside the potential clique.
                    # Since all x in clique are in common, (w,x) is an edge.
                    # So we just want to add as many vertices as possible.
                    # Heuristic: Add w with max degree in the *remaining graph*? Or max degree in *common*?
                    # Max degree in common means it connects to many other potential members.
                    deg_in_common = popcount(common & rem_adj[w])
                    if deg_in_common > best_w_deg:
                        best_w_deg = deg_in_common
                        best_w = w
                
                if best_w < 0:
                    break
                
                clique.append(best_w)
                common &= rem_adj[best_w]
                added = True
            
            # Remove edges covered by this clique
            clique_set = set(clique)
            new_remaining = []
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    # Remove edge from rem_adj
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
                    rem_deg[u] -= 1
                    rem_deg[v] -= 1
                else:
                    new_remaining.append((u, v))
            
            cliques.append(clique)
            remaining_edges = new_remaining
        
        return cliques

    def solve_instance(n, p, seed, deadline):
        adj, edges = gen_graph(n, p, seed)
        
        if not edges:
            return []
        
        best_cliques = [[u, v] for u, v in edges]
        best_count = len(best_cliques)
        
        rng_master = random.Random(42)
        
        # Time budget per instance
        elapsed = time.time() - start_time
        remaining_time = deadline - start_time - elapsed
        if n <= 32:
            num_restarts = min(200, max(10, int(remaining_time / 0.05)))
        else:
            num_restarts = min(100, max(5, int(remaining_time / 0.1)))
        
        strategies = ['max_deg', 'min_common', 'random']
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            rng = random.Random(rng_master.randint(0, 2**31))
            strat = strategies[restart % len(strategies)]
            
            result = greedy_partition(n, edges, adj, strat, rng, deadline)
            
            if result is not None:
                count = len(result)
                if count < best_count:
                    best_count = count
                    best_cliques = result
        
        return best_cliques

    # Solve both instances
    # Allocate time roughly proportional to complexity or just split
    # 32 nodes is much faster than 64
    result1 = solve_instance(32, 0.6, 601, DEADLINE)
    
    if time.time() > DEADLINE:
        adj2, edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, DEADLINE)
    
    return [result1, result2]
