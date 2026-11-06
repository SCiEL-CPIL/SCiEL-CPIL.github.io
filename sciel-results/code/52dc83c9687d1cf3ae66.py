import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit (150s kill limit mentioned, but 120s task limit is stricter)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def solve_mis_bitset(adj, n, time_limit, start_time, initial_bound=0):
        """
        Solve Maximum Independent Set using Branch and Bound with bitsets.
        adj: list of lists of neighbors (indices)
        n: number of vertices
        time_limit: max seconds to run
        start_time: timestamp of start
        initial_bound: lower bound to prune
        """
        # Map vertices to bit positions
        # We use Python integers as bitsets
        # adj_mask[i] is a bitmask of neighbors of i
        
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_mask[i] = mask
        
        best_set = []
        best_size = initial_bound
        
        # Order vertices by degree (descending) to improve pruning
        # However, reordering complicates the bitset indices. 
        # Standard B&B works on the given order. 
        # We can sort the vertices initially and map indices.
        
        # Let's stick to the original indices for simplicity of bitsets, 
        # but we can use a simple heuristic for branching: pick vertex with max degree in remaining set.
        
        # To optimize, we pre-compute a list of vertices sorted by degree?
        # No, the degree changes as we remove vertices.
        
        # Global variables for recursion state
        current_set = []
        current_size = 0
        
        # We need to pass the remaining available vertices as a bitmask
        # avail: bitmask of vertices that can still be added
        # chosen: bitmask of vertices already chosen? No, we track current_set list.
        
        # Actually, standard MIS B&B:
        # State: (available_mask, current_size, current_list)
        # Branch: Pick a vertex v from available.
        # 1. Include v: new_avail = available & ~adj_mask[v] & ~(1<<v)
        # 2. Exclude v: new_avail = available & ~(1<<v)
        
        # To improve speed, we use a greedy upper bound.
        # Greedy coloring or simple greedy independent set size on the subgraph induced by available.
        
        def greedy_bound(avail_mask):
            # Estimate max independent set size in subgraph induced by avail_mask
            # Simple greedy: repeatedly pick vertex with min degree in subgraph?
            # Or just count bits / (avg degree + 1)?
            # Let's use a fast greedy: 
            # Pick a vertex, remove it and its neighbors, repeat.
            # This gives a lower bound, not upper.
            # For upper bound, we can use: sum of 1/(deg(v)+1) ? No.
            # A common upper bound is the size of a maximal independent set found greedily? No, that's lower.
            # Upper bound: n_avail.
            # Better upper bound: Coloring bound.
            # For now, let's use a simple heuristic: 
            # If the graph is sparse, the bound is loose.
            # Let's try a simple recursive bound:
            # Actually, let's just use the number of bits set as the bound initially, 
            # and rely on the branching to prune.
            # To make it faster, we can use a simple "first fit" coloring approximation?
            # Given time constraints, let's use a simpler bound: 
            # The size of a maximal independent set in the induced subgraph is a lower bound.
            # We need an UPPER bound to prune.
            # Upper bound: The number of vertices in the available set.
            # We can improve this by checking if the available set is an independent set itself.
            
            # Let's implement a simple upper bound:
            # If the available vertices form a clique, bound is 1.
            # If they form an independent set, bound is count.
            # Generally, bound = count.
            
            # To be smarter, we can use a greedy coloring of the available subgraph.
            # The number of colors is an upper bound on the clique number, 
            # but we need independent set.
            # Wait, MIS in G is Clique in Complement(G).
            # Coloring G gives an upper bound on Clique(G)? No.
            # Coloring G gives an upper bound on the size of a Clique in G? 
            # No, if G is k-colorable, the max clique size is <= k.
            # So if we color the *Complement* graph, we get an upper bound on MIS.
            # But we don't have the complement graph easily.
            
            # Let's stick to a simple bound: count bits.
            # And use the branching order to prune quickly.
            return avail_mask.bit_count()

        def dfs(avail_mask, current_list, current_size):
            nonlocal best_set, best_size
            
            if time.time() - start_time > time_limit:
                return
            
            if avail_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_list)
                return
            
            # Upper bound
            bound = greedy_bound(avail_mask)
            if current_size + bound <= best_size:
                return
            
            # Heuristic: pick a vertex with highest degree in the available subgraph
            # Or just pick the first available vertex?
            # Picking high degree vertices first often prunes better.
            
            # Find available vertices
            # To avoid iterating all n, we can extract set bits.
            # For n=240, n is small enough to iterate.
            
            # Let's pick the vertex with the maximum number of neighbors in avail_mask
            # deg[v] = (adj_mask[v] & avail_mask).bit_count()
            
            # Optimization: Pre-compute degrees? No, they change.
            # Iterate through available bits.
            
            # To speed up, we can just pick the lowest index available vertex?
            # Or random?
            # Let's pick the vertex with max degree in the induced subgraph.
            
            # Extract indices of available vertices
            # This is O(n) per call.
            
            # If n is small (< 50), this is fine. For 240, it might be slow if depth is large.
            # Let's try picking the first available vertex to start, and see.
            # Actually, let's use a simple heuristic: 
            # The vertex with the highest degree in the *original* graph, 
            # restricted to available.
            
            # Let's just pick the first available vertex for now to keep it simple and fast.
            # We can optimize later if needed.
            
            # Get lowest set bit
            # v = (avail_mask & -avail_mask).bit_length() - 1
            
            # Better: Pick vertex with max degree in avail subgraph.
            # This requires O(k) where k is number of available vertices.
            
            # Let's try a compromise: 
            # If available count is small, check all.
            # If large, pick first.
            
            avail_count = avail_mask.bit_count()
            
            if avail_count <= 20:
                # Find max degree vertex
                max_deg = -1
                v = -1
                # Iterate bits
                m = avail_mask
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    deg = (adj_mask[idx] & avail_mask).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        v = idx
                    m ^= lb
            else:
                # Pick first available
                lb = avail_mask & -avail_mask
                v = lb.bit_length() - 1
            
            # Branch 1: Include v
            # New avail: remove v and its neighbors
            new_avail_include = avail_mask & ~adj_mask[v] & ~(1 << v)
            current_list.append(v)
            dfs(new_avail_include, current_list, current_size + 1)
            current_list.pop()
            
            # Branch 2: Exclude v
            new_avail_exclude = avail_mask & ~(1 << v)
            dfs(new_avail_exclude, current_list, current_size)

        # Initial call
        # We need an initial lower bound. 
        # We can use a greedy independent set.
        
        def greedy_independent_set():
            avail = (1 << n) - 1
            chosen = []
            while avail:
                # Pick vertex with min degree in avail? 
                # Or just first?
                # Min degree heuristic for MIS is good.
                min_deg = n + 1
                v = -1
                m = avail
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    deg = (adj_mask[idx] & avail).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        v = idx
                    m ^= lb
                
                if v == -1: break
                
                chosen.append(v)
                # Remove v and neighbors
                avail &= ~adj_mask[v]
                avail &= ~(1 << v)
            return chosen

        init_set = greedy_independent_set()
        best_size = len(init_set)
        best_set = init_set
        
        # Start DFS
        full_mask = (1 << n) - 1
        dfs(full_mask, [], 0)
        
        return best_set

    def build_conflict_graph(V):
        """
        Build conflict graph where edge (i,j) exists if V[i] + V[j] is in V.
        """
        n = len(V)
        V_set = set(V)
        adj = [[] for _ in range(n)]
        
        for i in range(n):
            vi = V[i]
            for j in range(i+1, n):
                vj = V[j]
                s = vi + vj
                if s in V_set:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    def heuristic_solve(V, time_limit, start_time):
        """
        Fast heuristic: Greedy removal.
        """
        S = set(V)
        # Greedy removal of conflicts
        # To make it deterministic and fast, we just remove elements involved in sums.
        
        # A simple heuristic: Keep odds?
        # Odds are always sum-free (odd+odd=even, even not in odds if only odds kept? 
        # Wait, if we only keep odds, odd+odd=even. If even is not in the set, it's safe.
        # But if we keep odds, we don't have evens. So odd+odd=even, and even is NOT in the set. 
        # So a set of only odd numbers is sum-free?
        # Let a,b be odd. a+b is even. If the set contains only odds, a+b is not in the set.
        # Yes, any subset of odd numbers is sum-free.
        # Similarly, any subset of numbers with the same sign? No.
        
        # So, one strong heuristic is to take all odds.
        odds = [x for x in V if x % 2 != 0]
        
        # Another heuristic: Take numbers with large absolute value?
        # If |a|, |b| are large, a+b might be out of range or not in set.
        
        # Let's return the best of:
        # 1. All odds
        # 2. All evens? (even+even=even, so NOT sum-free unless no sums exist)
        # 3. Positive numbers? (pos+pos=pos, NOT sum-free)
        
        # So "All Odds" is a very safe and often large subset.
        # For random integers, ~half are odd. So size ~ n/2.
        # Bourgain bound is (n+2)/3. n/2 > n/3.
        
        return odds

    results = []

    # Time allocation
    # Instance 1 (60): 5s
    # Instance 2 (120): 15s
    # Instance 3 (240): 90s
    # Total 110s.
    
    time_alloc = {60: 5.0, 120: 15.0, 240: 90.0}

    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        if remaining_time < 2:
            # Emergency
            results.append([x for x in V if x % 2 != 0])
            continue

        t_limit = min(time_alloc.get(size, 10.0), remaining_time - 2)
        
        # Get heuristic solution first
        heuristic_sol = heuristic_solve(V, t_limit, start_time)
        best_size = len(heuristic_sol)
        best_set = heuristic_sol
        
        # Try exact solver if time permits
        # The exact solver is only viable for smaller n or if we have lots of time.
        # For n=240, B&B might be too slow unless the graph is very sparse.
        # Let's try it for all, but with strict time limit.
        
        if size <= 100 or t_limit > 20:
            adj = build_conflict_graph(V)
            # Map solution indices back to values
            # The solver returns indices
            try:
                idx_sol = solve_mis_bitset(adj, size, t_limit, start_time, initial_bound=best_size)
                if len(idx_sol) > best_size:
                    best_set = [V[i] for i in idx_sol]
            except Exception:
                pass # Fallback to heuristic
        
        # Verify
        if not is_sum_free(best_set):
            # Fallback to odds if something went wrong
            best_set = [x for x in V if x % 2 != 0]
            
        results.append(sorted(best_set))

    return results
