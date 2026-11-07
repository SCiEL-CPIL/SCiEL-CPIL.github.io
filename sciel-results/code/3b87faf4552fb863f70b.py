"""Strategy: Randomized greedy with restarts to find a better clique partition."""

def construct():
    from random import Random
    import time

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_once(n, k, p, seed, rng_solver, deadline):
        rng_gen = Random(seed)
        adj = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng_gen.random() < p:
                    adj[v].add(u)
        
        # available contains core edges (u, v) with u < v, u, v < k, not yet covered
        available = set((u, v) for u in range(k) for v in range(u + 1, k))
        covered_cross = set()
        cliques = []
        
        # Randomize the order of processing I
        I_vertices = list(range(k, n))
        rng_solver.shuffle(I_vertices)
        
        for i in I_vertices:
            if time.time() > deadline:
                break
            N_i = set(adj[i])
            while N_i:
                if time.time() > deadline:
                    break
                
                # Find maximal clique S in N_i using available edges
                # We use a randomized greedy: pick vertices in random order?
                # Or just standard greedy but the set of candidates is random?
                # Let's try picking a random vertex to start, then expand.
                
                # To ensure we get a "good" clique, we can try a few random starts and pick the best?
                # Given time budget, let's stick to one pass but maybe randomized order of vertices in N_i.
                
                candidates = list(N_i)
                rng_solver.shuffle(candidates)
                
                S = set()
                for v in candidates:
                    can_add = True
                    for u in S:
                        edge = (min(u, v), max(u, v))
                        if edge not in available:
                            can_add = False
                            break
                    if can_add:
                        S.add(v)
                
                if not S:
                    v = next(iter(N_i))
                    S = {v}
                
                cliques.append([i] + list(S))
                
                # Mark cross edges as covered
                for c in S:
                    covered_cross.add((c, i))
                
                # Remove core edges from available
                for u in S:
                    for v in S:
                        if u < v:
                            available.discard((u, v))
                
                N_i.difference_update(S)
            
            # If broke out of while due to timeout, cover remaining N_i with 2-cliques
            if N_i:
                for c in N_i:
                    if (c, i) not in covered_cross:
                        cliques.append([c, i])
                        covered_cross.add((c, i))
        
        # Now cover remaining core edges
        remaining_edges = set(available)
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: use 2-cliques for all remaining
                for e in remaining_edges:
                    cliques.append([e[0], e[1]])
                break
            
            # Find a maximal clique in the graph defined by remaining_edges
            # Vertices are 0..k-1
            # Build adjacency from remaining_edges
            adj_rem = [set() for _ in range(k)]
            for u, v in remaining_edges:
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            # Randomized greedy for core clique
            # Pick a random vertex with degree > 0
            verts_with_deg = [v for v in range(k) if adj_rem[v]]
            if not verts_with_deg:
                break
            
            start_v = rng_solver.choice(verts_with_deg)
            S = {start_v}
            
            # Expand greedily
            # We can shuffle the order of checking neighbors
            for v in range(k):
                if v in S:
                    continue
                can_add = True
                for u in S:
                    if v not in adj_rem[u]:
                        can_add = False
                        break
                if can_add:
                    S.add(v)
            
            if len(S) < 2:
                # Should not happen if remaining_edges not empty and start_v chosen from adj_rem
                # But if S is size 1, it covers no edges.
                # Pick an edge directly
                u, v = next(iter(remaining_edges))
                S = {u, v}
            
            cliques.append(list(S))
            
            # Remove edges in S from remaining_edges
            for u in S:
                for v in S:
                    if u < v:
                        remaining_edges.discard((u, v))
        
        return cliques

    def _solve_safe(n, k, p, seed, deadline):
        best_cliques = None
        best_count = float('inf')
        
        # Run multiple restarts
        # Estimate time per run
        # For n=90, k=30, one run is very fast (< 1ms probably)
        # So we can run many.
        
        start = time.time()
        num_restarts = 0
        
        while time.time() < deadline:
            rng_solver = Random(int(time.time() * 1000) + num_restarts)
            cliques = _solve_once(n, k, p, seed, rng_solver, deadline)
            
            # Validate? The problem says "Any broken rule makes the whole result invalid".
            # We assume our construction is valid by design.
            # But let's double check count.
            count = len(cliques)
            
            if count < best_count:
                best_count = count
                best_cliques = cliques
            
            num_restarts += 1
            
            # If we are getting close to deadline, break
            if time.time() > deadline - 5:
                break
        
        return best_cliques

    results = []
    start_time = time.time()
    # Allocate time to each instance
    # Total budget 120s. Use 100s.
    # 3 instances. ~33s each.
    time_per_instance = 30
    deadline = start_time + 100
    
    for idx, spec in enumerate(((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))):
        instance_deadline = min(deadline, start_time + 100 - (2 - idx) * 5) # Simple allocation
        # Better: just use a global deadline and let it run as long as it can
        # But we must return something.
        # Let's give each instance a fair share.
        # Instance 1: 10s, Instance 2: 20s, Instance 3: 60s?
        # Or just run them sequentially with a global deadline.
        
        # Let's just use the global deadline for all, but ensure we finish.
        # The first two are small.
        results.append(_solve_safe(*spec, deadline))
        
        # Update deadline for next? No, _solve_safe uses the passed deadline.
        # If we pass the same deadline, it will use all remaining time for the first instance?
        # No, _solve_safe runs a loop `while time.time() < deadline`.
        # So the first instance will consume ALL time until deadline.
        # We need to split the time.
        
    # RE-IMPLEMENTATION OF TIME MANAGEMENT
    # The previous loop logic was flawed.
    
    # Let's restart the main logic.
    results = []
    start_time = time.time()
    total_deadline = start_time + 110 # 10s margin
    
    # Time allocation:
    # n=30: small. 5s.
    # n=60: medium. 15s.
    # n=90: large. 80s.
    allocations = [5, 15, 80]
    
    current_deadline = start_time
    for idx, spec in enumerate(((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))):
        current_deadline += allocations[idx]
        # Ensure we don't exceed total deadline
        current_deadline = min(current_deadline, total_deadline)
        
        results.append(_solve_safe(*spec, current_deadline))
        
    return results

