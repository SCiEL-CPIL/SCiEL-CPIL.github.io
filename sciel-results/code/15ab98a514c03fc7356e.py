<<<<<<< SEARCH
    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    best_score = float('inf')
    best_vals = None
    best_edges = None

    def consider(sc, vals, edges):
        nonlocal best_score, best_vals, best_edges
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(edges)

    # Phase 1: structured value pools -> greedy edges -> polish edges.
    pools = value_pools()
    random.shuffle(pools)
    
    # Sort pools by potential: prefer pools with more internal structure
    # (many pairs with same sum/product)
    def pool_score(pool):
        if len(pool) < n:
            return 0
        # Count collisions in sums and products for random samples
        samples = random.sample(pool, min(len(pool), 30))
        sum_counts = {}
        prod_counts = {}
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                s = samples[i] + samples[j]
                p = samples[i] * samples[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        sum_collisions = sum(c for c in sum_counts.values() if c > 1)
        prod_collisions = sum(c for c in prod_counts.values() if c > 1)
        return sum_collisions + prod_collisions
    
    # Try best pools first, but also keep some random exploration
    scored_pools = [(pool_score(p), p) for p in pools]
    scored_pools.sort(key=lambda x: -x[0])
    
    # Take top pools + some random
    top_pools = [p for _, p in scored_pools[:20]]
    remaining_pools = [p for _, p in scored_pools[20:]]
    random.shuffle(remaining_pools)
    ordered_pools = top_pools + remaining_pools[:10]
    random.shuffle(ordered_pools)
    
    for pool in ordered_pools:
        if time.time() > deadline - 6.0:
            break
        for _ in range(3):  # More restarts per pool
            if time.time() > deadline - 5.0:
                break
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            # polish edges for this fixed value set
            steps = 5000
            polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                               steps, deadline - 4.0)
            consider(psc, vals, polished)

    # Phase 2: joint search from several starts.
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))
    
    # Add diverse starts from top pools
    for pool in top_pools[:8]:
        vals = random_values(pool)
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))
    
    # Add a few random starts
    for _ in range(4):
        vals = random_values(pools[random.randrange(len(pools))])
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))

    remaining = deadline - 5.0 - time.time()
    per = max(3000, int(remaining / max(1, len(starts)) / 0.00025))
    for vals0, edges0 in starts:
        if time.time() > deadline - 5.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 4.0)
        consider(sc, vals, edges)

    # Phase 3: final edge polish for best values.
    if time.time() < deadline - 3.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           30000, deadline - 2.0)
        consider(psc, best_vals, polished)

    # Phase 4: final value polish for best edges with smarter proposals.
    if time.time() < deadline - 1.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        small_vals = list(range(1, 31))
        while it < 15000 and time.time() < deadline - 1.0:
            it += 1
            idx = random.randrange(n)
            old = vals[idx]
            
            # Smarter value proposal
            r = random.random()
            if r < 0.4:
                nv = random.choice(small_vals)
            elif r < 0.6:
                # Try to create sum collision with an existing edge
                other_idx = random.randrange(n)
                if other_idx != idx:
                    target = random.randint(2, 60)
                    nv = target - vals[other_idx]
                    if nv < 1:
                        nv = random.randint(1, 30)
                else:
                    nv = random.randint(1, 30)
            elif r < 0.8:
                nv = random.randint(1, 50)
            else:
                nv = random.randint(1, 100)
            
            if nv < 1 or nv > 10000:
                continue
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
            else:
                vals[idx] = old

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)
    
    # Final refinement: try small perturbations of the best solution
    if time.time() < deadline - 0.5:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        small_vals = list(range(1, 25))
        for _ in range(5000):
            if time.time() > deadline - 0.5:
                break
            idx = random.randrange(n)
            old = vals[idx]
            nv = random.choice(small_vals)
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                best_vals = list(vals)
                best_score = cur
            else:
                vals[idx] = old
=======
    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    best_score = float('inf')
    best_vals = None
    best_edges = None

    def consider(sc, vals, edges):
        nonlocal best_score, best_vals, best_edges
        if sc < best_score:
            best_score = sc
            best_vals = list(vals)
            best_edges = list(edges)

    # Phase 0: Try constraint-driven backtracking for small K values
    # Start with optimistic K and increase until we find a solution
    for K in [15, 16, 17, 18, 19, 20, 22, 25]:
        if time.time() > deadline - 30.0:
            break
        # Try a few value sets with this K
        pools = value_pools()
        random.shuffle(pools)
        for pool in pools[:10]:
            if time.time() > deadline - 25.0:
                break
            for _ in range(2):
                if time.time() > deadline - 20.0:
                    break
                vals = random_values(pool)
                edges, sc = constrained_edges(vals, K, max_time=2.0)
                if edges is not None:
                    consider(sc, vals, edges)
                    # Polish further
                    polished, pset, psc = polish_edges(vals, edges, set(edges),
                                                       3000, deadline - 15.0)
                    consider(psc, vals, polished)
                    if best_score <= K:
                        break
            if best_score <= K:
                break
        if best_score <= K:
            break

    # Phase 1: structured value pools -> greedy edges -> polish edges.
    pools = value_pools()
    random.shuffle(pools)
    
    # Sort pools by potential: prefer pools with more internal structure
    # (many pairs with same sum/product)
    def pool_score(pool):
        if len(pool) < n:
            return 0
        # Count collisions in sums and products for random samples
        samples = random.sample(pool, min(len(pool), 30))
        sum_counts = {}
        prod_counts = {}
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                s = samples[i] + samples[j]
                p = samples[i] * samples[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        sum_collisions = sum(c for c in sum_counts.values() if c > 1)
        prod_collisions = sum(c for c in prod_counts.values() if c > 1)
        return sum_collisions + prod_collisions
    
    # Try best pools first, but also keep some random exploration
    scored_pools = [(pool_score(p), p) for p in pools]
    scored_pools.sort(key=lambda x: -x[0])
    
    # Take top pools + some random
    top_pools = [p for _, p in scored_pools[:20]]
    remaining_pools = [p for _, p in scored_pools[20:]]
    random.shuffle(remaining_pools)
    ordered_pools = top_pools + remaining_pools[:10]
    random.shuffle(ordered_pools)
    
    for pool in ordered_pools:
        if time.time() > deadline - 6.0:
            break
        for _ in range(3):  # More restarts per pool
            if time.time() > deadline - 5.0:
                break
            vals = random_values(pool)
            chosen, chosen_set = greedy_edges(vals)
            sc = full_score(vals, chosen)
            consider(sc, vals, chosen)
            
            # Try constraint-driven refinement
            if sc > 18:
                edges2, sc2 = constrained_edges(vals, min(sc, 20), max_time=1.5)
                if edges2 is not None and sc2 < sc:
                    consider(sc2, vals, edges2)
                    chosen, chosen_set = edges2, set(edges2)
            
            # polish edges for this fixed value set
            steps = 5000
            polished, pset, psc = polish_edges(vals, chosen[:], chosen_set.copy(),
                                               steps, deadline - 4.0)
            consider(psc, vals, polished)

    # Phase 2: joint search from several starts.
    starts = []
    if best_vals is not None:
        starts.append((list(best_vals), list(best_edges)))
    
    # Add diverse starts from top pools
    for pool in top_pools[:8]:
        vals = random_values(pool)
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))
    
    # Add a few random starts
    for _ in range(4):
        vals = random_values(pools[random.randrange(len(pools))])
        edges = random.sample(all_edges, m)
        starts.append((vals, edges))

    remaining = deadline - 5.0 - time.time()
    per = max(3000, int(remaining / max(1, len(starts)) / 0.00025))
    for vals0, edges0 in starts:
        if time.time() > deadline - 5.0:
            break
        sc, vals, edges = joint_search(vals0, edges0, per, deadline - 4.0)
        consider(sc, vals, edges)

    # Phase 3: final edge polish for best values.
    if time.time() < deadline - 3.0 and best_vals is not None:
        chosen, chosen_set = list(best_edges), set(best_edges)
        polished, pset, psc = polish_edges(best_vals, chosen, chosen_set,
                                           30000, deadline - 2.0)
        consider(psc, best_vals, polished)

    # Phase 4: final value polish for best edges with smarter proposals.
    if time.time() < deadline - 1.0 and best_edges is not None:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        best = cur
        it = 0
        small_vals = list(range(1, 31))
        while it < 15000 and time.time() < deadline - 1.0:
            it += 1
            idx = random.randrange(n)
            old = vals[idx]
            
            # Smarter value proposal
            r = random.random()
            if r < 0.4:
                nv = random.choice(small_vals)
            elif r < 0.6:
                # Try to create sum collision with an existing edge
                other_idx = random.randrange(n)
                if other_idx != idx:
                    target = random.randint(2, 60)
                    nv = target - vals[other_idx]
                    if nv < 1:
                        nv = random.randint(1, 30)
                else:
                    nv = random.randint(1, 30)
            elif r < 0.8:
                nv = random.randint(1, 50)
            else:
                nv = random.randint(1, 100)
            
            if nv < 1 or nv > 10000:
                continue
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                if r < best:
                    best = r
                    best_vals = list(vals)
                    best_score = best
            else:
                vals[idx] = old

    if best_vals is None:
        best_vals = list(range(1, 15))
        best_edges = list(all_edges[:m])
        best_score = full_score(best_vals, best_edges)
    
    # Final refinement: try small perturbations of the best solution
    if time.time() < deadline - 0.5:
        vals = list(best_vals)
        edges = list(best_edges)
        cur = full_score(vals, edges)
        small_vals = list(range(1, 25))
        for _ in range(5000):
            if time.time() > deadline - 0.5:
                break
            idx = random.randrange(n)
            old = vals[idx]
            nv = random.choice(small_vals)
            if nv == old or nv in vals:
                continue
            vals[idx] = nv
            r = full_score(vals, edges)
            if r < cur:
                cur = r
                best_vals = list(vals)
                best_score = cur
            else:
                vals[idx] = old
>>>>>>> REPLACE

