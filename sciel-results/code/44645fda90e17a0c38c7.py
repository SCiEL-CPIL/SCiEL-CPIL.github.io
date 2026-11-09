import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s for safety/overhead
    
    # The maximum possible n is 40 because R(3,10) <= 41.
    # We try to find a graph with n=40 first.
    
    def check_alpha_10(adj, n):
        """
        Exact branch and bound to check if alpha(G) >= 10.
        Returns True if an independent set of size 10 exists.
        Returns False if alpha(G) < 10.
        """
        # Precompute non-adjacency masks for independent set search
        # In the independent set problem, we want to pick vertices that are NOT connected.
        # Let comp_adj[i] be the set of vertices NOT adjacent to i (excluding i).
        all_mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            # Vertices not adjacent to i, excluding i itself
            comp_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
        
        # Greedy heuristic for initial upper bound on alpha
        def greedy_alpha(mask):
            count = 0
            m = mask
            while m:
                # Pick vertex with min degree in the induced subgraph? 
                # For independent set, we want to pick a vertex that removes fewest options?
                # Actually, standard greedy for independent set: pick vertex, remove it and its neighbors.
                # Here "neighbors" in the independent set context are the adjacent vertices in the original graph.
                # So if we pick v, we must remove N(v) from consideration.
                # The remaining candidates are m & ~adj[v].
                # To maximize the count, we should probably pick the vertex with the smallest degree in the remaining graph?
                # Or just a simple greedy: pick the first one.
                lsb = m & -m
                v = lsb.bit_length() - 1
                count += 1
                m &= ~adj[v]
                m &= ~lsb # Remove v itself
            return count

        # Exact B&B
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            
            # Pruning: if remaining vertices + current size < 10
            if mask.bit_count() + size < 10:
                return False
            
            # Heuristic: Pick vertex with minimum degree in the induced subgraph of 'mask'
            # In the context of finding an independent set, the "degree" of a vertex v in the remaining graph
            # is the number of other vertices in 'mask' that are adjacent to v in the original graph.
            # If we pick v, we remove all its neighbors from 'mask'.
            # To prune effectively, we want to pick a vertex that removes many candidates?
            # Or a vertex that is "easy" to include?
            # Standard MaxClique/IndependentSet heuristic: branch on vertex with max degree in the induced subgraph
            # for Clique (maximize conflicts). For Independent Set, it's the complement, so min degree in original graph?
            # Let's stick to a robust heuristic: pick vertex v in mask that minimizes the number of remaining candidates
            # after picking v. i.e., minimize |mask & ~adj[v]|.
            
            min_remaining = n + 1
            best_v = -1
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                # If we pick v, the new mask is (mask & ~adj[v]) & ~lsb
                # We want to choose v such that this new mask is small?
                # Actually, for B&B, branching on the vertex that leads to the most pruning is good.
                # Let's just pick the vertex with the smallest index for determinism if heuristic is slow,
                # or use the min-degree in the induced subgraph of the complement?
                # Let's use: pick v such that (mask & ~adj[v]).bit_count() is minimized.
                rem_count = (mask & ~adj[v] & ~lsb).bit_count()
                if rem_count < min_remaining:
                    min_remaining = rem_count
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v
            # New mask: vertices in mask that are not adjacent to best_v
            new_mask = mask & ~adj[best_v] & ~v_bit
            if dfs(new_mask, size + 1):
                return True
            
            # Branch 2: Exclude best_v
            if dfs(mask & ~v_bit, size):
                return True
            
            return False
        
        return dfs(all_mask, 0)

    def try_find_graph(n, time_budget):
        # Initialize random maximal triangle-free graph
        adj = [0] * n
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Local search: try to reduce alpha
        # We will randomly flip edges (add/remove) if it maintains triangle-free property
        # and check if alpha decreases.
        
        # First, check if current graph is valid
        if not check_alpha_10(adj, n):
            return adj
        
        # Greedy alpha estimate for speed
        def greedy_alpha_mask(mask):
            count = 0
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                count += 1
                m &= ~adj[v]
                m &= ~lsb
            return count

        current_alpha_est = greedy_alpha_mask((1 << n) - 1)
        
        # Simple hill climbing
        # We want to ADD edges to reduce alpha, or REMOVE edges if stuck?
        # Adding edges makes triangle-free constraint harder but reduces alpha.
        # Let's try to add random non-edges that don't create triangles.
        
        # Precompute list of non-edges
        # For efficiency, we'll just pick random pairs and check.
        
        for _ in range(5000): # Max iterations per restart
            if time.time() > start_time + time_budget:
                break
            
            # Pick a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if (adj[u] >> v) & 1:
                continue # Already an edge
            
            # Check if adding (u,v) creates a triangle
            if (adj[u] & adj[v]) != 0:
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check new alpha estimate
            new_alpha_est = greedy_alpha_mask((1 << n) - 1)
            
            # If alpha estimate improved (decreased), keep it.
            # If not, revert? Or accept with probability (simulated annealing)?
            # For now, simple hill climbing: keep if better, revert if worse.
            # But we might get stuck. Let's try a few random walks.
            
            if new_alpha_est < current_alpha_est:
                current_alpha_est = new_alpha_est
                # Exact check if we are close
                if current_alpha_est <= 10:
                    if not check_alpha_10(adj, n):
                        return adj
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return None

    # Try n=40
    # We need to be careful with time.
    # Let's do a few restarts for n=40.
    
    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        # Number of restarts based on remaining time
        # Give n=40 more time
        budget = 100 if n == 40 else 50
        
        # Try multiple random initializations
        while time.time() < start_time + budget:
            res = try_find_graph(n, budget)
            if res:
                edges = []
                for i in range(n):
                    mask = res[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
            
            # If we run out of time inside try_find_graph, it returns None.
            # Check if we should stop
            if time.time() > start_time + budget:
                break

    # Fallback: Known graph with n=39?
    # Or just return a valid smaller graph if we found nothing.
    # The problem states "Maximize n".
    # If we fail to find n=40 or 39, we should return the best we have.
    # The current program returns a fallback of n=26.
    # We can try to construct a valid graph for n=38, 37, etc.
    
    # Let's try a deterministic construction for n=39 if random fails?
    # Or just loop down from 38 to 1 and find the first valid one.
    
    for n in range(38, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Quick random search for smaller n
        for _ in range(100):
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not check_alpha_10(adj, n):
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
        
    # Absolute fallback
    return {'n': 1, 'edges': []}
