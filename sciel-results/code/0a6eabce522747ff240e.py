import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def popcount(x):
        return x.bit_count()

    def is_valid_add(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        mask_u = adj_mask[u]
        mask_v = adj_mask[v]
        
        # u and v common neighbors (before adding): must be <= 2
        if popcount(mask_u & mask_v) > 2:
            return False
        
        # For each neighbor w of u (w != v): check common(w, v) <= 1
        temp = mask_u
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == v:
                continue
            if popcount(adj_mask[w] & mask_v) > 1:
                return False
        
        # For each neighbor w of v (w != u): check common(u, w) <= 1
        temp = mask_v
        while temp:
            lsb = temp & (-temp)
            w = lsb.bit_length() - 1
            temp ^= lsb
            if w == u:
                continue
            if popcount(mask_u & adj_mask[w]) > 1:
                return False
        
        return True

    def add_edge_fast(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge_fast(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def edges_from_mask(adj_mask):
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def edge_count(adj_mask):
        cnt = 0
        for u in range(n):
            cnt += popcount(adj_mask[u])
        return cnt // 2

    def validate_adj_mask(adj_mask):
        """Verify K2,3-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj_mask[u] & adj_mask[v]) > 2:
                    return False
        return True

    def greedy_construct(seed_offset=0, base_type='cliques'):
        """Greedy construction with different base structures."""
        rng = random.Random(seed_offset)
        adj_mask = [0] * n
        
        if base_type == 'cliques':
            # Start with 5 cliques of size 4
            for g in range(5):
                base = 4 * g
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj_mask[base + i] |= (1 << (base + j))
                        adj_mask[base + j] |= (1 << (base + i))
        elif base_type == 'circulant':
            # 5-circulant: connect to ±1, ±2, +5 mod 20
            for i in range(n):
                for d in [1, 2, 5]:
                    j = (i + d) % n
                    if j != i and not ((adj_mask[i] >> j) & 1):
                        if is_valid_add(adj_mask, i, j):
                            adj_mask[i] |= (1 << j)
                            adj_mask[j] |= (1 << i)
        elif base_type == 'random':
            # Random 4-regular base
            for i in range(n):
                candidates = [j for j in range(n) if j != i and not ((adj_mask[i] >> j) & 1)]
                rng.shuffle(candidates)
                for j in candidates[:4]:
                    if is_valid_add(adj_mask, i, j):
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
        
        # Fill remaining edges greedily in random order
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not (adj_mask[u] & (1 << v)):
                if is_valid_add(adj_mask, u, v):
                    add_edge_fast(adj_mask, u, v)
        
        return adj_mask

    def local_search(adj_mask, max_time=30.0, rng=None):
        """Aggressive local search with 1-edge and 2-edge swaps."""
        if rng is None:
            rng = random.Random()
        current = list(adj_mask)
        best = list(current)
        best_cnt = edge_count(current)
        start_ls = time.time()
        ls_deadline = start_ls + max_time
        
        all_edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Phase 1: Greedy add
        for _ in range(20):
            if time.time() > ls_deadline:
                break
            rng.shuffle(all_edges_list)
            added = False
            for u, v in all_edges_list:
                if not (current[u] & (1 << v)):
                    if is_valid_add(current, u, v):
                        add_edge_fast(current, u, v)
                        added = True
            if not added:
                break
        
        cnt = edge_count(current)
        if cnt > best_cnt:
            best_cnt = cnt
            best = list(current)
        
        # Phase 2: 1-edge swap
        for _ in range(3000):
            if time.time() > ls_deadline:
                break
            
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        existing.append((u, v))
            if not existing:
                break
            
            u, v = rng.choice(existing)
            
            saved = list(current)
            remove_edge_fast(current, u, v)
            
            rng.shuffle(all_edges_list)
            for a, b in all_edges_list:
                if not (current[a] & (1 << b)):
                    if is_valid_add(current, a, b):
                        add_edge_fast(current, a, b)
            
            cnt = edge_count(current)
            if cnt > best_cnt:
                best_cnt = cnt
                best = list(current)
            else:
                current = saved
        
        # Phase 3: 2-edge swap
        for _ in range(800):
            if time.time() > ls_deadline:
                break
            
            existing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if current[u] & (1 << v):
                        existing.append((u, v))
            if len(existing) < 2:
                break
            
            (u1, v1), (u2, v2) = rng.sample(existing, 2)
            if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                continue
            
            saved = list(current)
            remove_edge_fast(current, u1, v1)
            remove_edge_fast(current, u2, v2)
            
            rng.shuffle(all_edges_list)
            for a, b in all_edges_list:
                if not (current[a] & (1 << b)):
                    if is_valid_add(current, a, b):
                        add_edge_fast(current, a, b)
            
            cnt = edge_count(current)
            if cnt > best_cnt:
                best_cnt = cnt
                best = list(current)
            else:
                current = saved
        
        return best

    def genetic_algorithm(max_time=45.0):
        """Genetic algorithm with edge-bitset chromosomes and validity-preserving crossover."""
        rng = random.Random(int(time.time() * 1000) % 100000)
        pop_size = 12
        all_edges_list = [(u, v) for u in range(n) for v in range(u + 1, n)]
        num_edges = len(all_edges_list)
        
        # Initialize population with diverse greedy constructions
        population = []
        base_types = ['cliques', 'circulant', 'random']
        for i in range(pop_size):
            bt = base_types[i % len(base_types)]
            adj = greedy_construct(i * 7919, bt)
            population.append(adj)
        
        # Sort by edge count
        population.sort(key=edge_count, reverse=True)
        best_adj = list(population[0])
        best_cnt = edge_count(best_adj)
        
        ga_deadline = time.time() + max_time
        gen = 0
        
        while time.time() < ga_deadline:
            gen += 1
            
            # Selection: tournament selection
            def tournament_select():
                candidates = rng.sample(population, min(4, len(population)))
                return max(candidates, key=edge_count)
            
            # Crossover: combine disjoint edge subsets from two parents
            def crossover(parent1, parent2):
                """
                Crossover that preserves K2,3-free validity by only combining
                disjoint edge subsets. We take edges from parent1 that don't
                conflict with parent2's structure, and vice versa.
                """
                child = [0] * n
                
                # Get edge sets as bitmasks for fast comparison
                p1_edges = set()
                p2_edges = set()
                for u in range(n):
                    for v in range(u + 1, n):
                        if parent1[u] & (1 << v):
                            p1_edges.add((u, v))
                        if parent2[u] & (1 << v):
                            p2_edges.add((u, v))
                
                # Edges unique to parent1
                p1_unique = p1_edges - p2_edges
                # Edges unique to parent2
                p2_unique = p2_edges - p1_edges
                # Shared edges
                shared = p1_edges & p2_edges
                
                # Start with shared edges (safe)
                for u, v in shared:
                    add_edge_fast(child, u, v)
                
                # Try adding unique edges from parent1 in random order
                p1_unique_list = list(p1_unique)
                rng.shuffle(p1_unique_list)
                for u, v in p1_unique_list:
                    if is_valid_add(child, u, v):
                        add_edge_fast(child, u, v)
                
                # Try adding unique edges from parent2 in random order
                p2_unique_list = list(p2_unique)
                rng.shuffle(p2_unique_list)
                for u, v in p2_unique_list:
                    if is_valid_add(child, u, v):
                        add_edge_fast(child, u, v)
                
                # Greedy fill remaining
                for u, v in all_edges_list:
                    if not (child[u] & (1 << v)):
                        if is_valid_add(child, u, v):
                            add_edge_fast(child, u, v)
                
                return child
            
            # Mutation: add/remove random edges
            def mutate(adj):
                child = list(adj)
                num_mutations = rng.randint(1, 5)
                for _ in range(num_mutations):
                    if rng.random() < 0.5:
                        # Remove a random edge
                        existing = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if child[u] & (1 << v):
                                    existing.append((u, v))
                        if existing:
                            u, v = rng.choice(existing)
                            remove_edge_fast(child, u, v)
                    else:
                        # Add a random edge
                        u, v = rng.choice(all_edges_list)
                        if not (child[u] & (1 << v)):
                            if is_valid_add(child, u, v):
                                add_edge_fast(child, u, v)
                return child
            
            # Create new population
            new_population = []
            
            # Elitism: keep top 2
            for adj in population[:2]:
                new_population.append(list(adj))
            
            # Fill rest with offspring
            while len(new_population) < pop_size:
                if time.time() > ga_deadline:
                    break
                
                p1 = tournament_select()
                p2 = tournament_select()
                
                if rng.random() < 0.7:
                    child = crossover(p1, p2)
                else:
                    child = list(p1)
                
                if rng.random() < 0.4:
                    child = mutate(child)
                
                new_population.append(child)
            
            population = new_population
            population.sort(key=edge_count, reverse=True)
            
            # Track best
            if edge_count(population[0]) > best_cnt:
                best_cnt = edge_count(population[0])
                best_adj = list(population[0])
            
            # Periodically apply local search to best individual
            if gen % 3 == 0 and time.time() < ga_deadline - 5:
                improved = local_search(population[0], max_time=min(10.0, ga_deadline - time.time() - 2))
                cnt = edge_count(improved)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_adj = list(improved)
                    population[0] = list(improved)
        
        return best_adj

    # Main loop
    best_adj = None
    best_count = 0
    
    # Try multiple initial constructions with different base types
    base_types = ['cliques', 'circulant', 'random']
    
    for base_type in base_types:
        for seed in range(8):
            if time.time() > deadline - 10:
                break
            adj = greedy_construct(seed * 100 + hash(base_type) % 1000, base_type)
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj)
    
    # Run genetic algorithm
    if time.time() < deadline - 20:
        ga_result = genetic_algorithm(max_time=min(45.0, deadline - time.time() - 15))
        ga_cnt = edge_count(ga_result)
        if ga_cnt > best_count:
            best_count = ga_cnt
            best_adj = ga_result
    
    # Local search on best
    if best_adj and time.time() < deadline - 5:
        improved = local_search(best_adj, max_time=min(30.0, deadline - time.time() - 3))
        cnt = edge_count(improved)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved
    
    # More local search iterations if time permits
    while time.time() < deadline - 3:
        improved = local_search(best_adj, max_time=min(15.0, deadline - time.time() - 2))
        cnt = edge_count(improved)
        if cnt > best_count:
            best_count = cnt
            best_adj = improved
        else:
            # Try a fresh greedy + local search
            base_type = base_types[random.randint(0, 2)]
            adj = greedy_construct(random.randint(0, 10000), base_type)
            cnt2 = edge_count(adj)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = list(adj)
            if time.time() > deadline - 5:
                break
            improved2 = local_search(best_adj, max_time=min(10.0, deadline - time.time() - 2))
            cnt3 = edge_count(improved2)
            if cnt3 > best_count:
                best_count = cnt3
                best_adj = improved2
    
    # Validate and return
    if best_adj is None:
        # Fallback: just return a valid small graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if is_valid_add(adj, u, v):
                    add_edge_fast(adj, u, v)
        best_adj = adj
    
    # Final validation
    if not validate_adj_mask(best_adj):
        # Something went wrong, rebuild conservatively
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if is_valid_add(adj, u, v):
                    add_edge_fast(adj, u, v)
        best_adj = adj
    
    return edges_from_mask(best_adj)
