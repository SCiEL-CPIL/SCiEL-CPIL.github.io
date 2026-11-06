import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin for 120s limit

    def make_paley(q):
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    edges.append([i, j])
        return {'n': q, 'edges': edges}

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def build_adj_bit(n, edges):
        """Build adjacency as list of integers (bitmasks)."""
        adj_bits = [0] * n
        for u, v in edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj_bits

    def count_k4(n, adj_bits):
        """Count K4 subgraphs using bitmask operations."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_bits[i] & (1 << j):
                    common_ij = adj_bits[i] & adj_bits[j]
                    # Only consider k > j
                    for k in range(j + 1, n):
                        if common_ij & (1 << k):
                            common_ijk = common_ij & adj_bits[k]
                            # Count vertices l > k in common_ijk
                            count += bin(common_ijk >> (k + 1)).count('1')
        return count

    def count_indep_6(n, adj_bits):
        """Count independent sets of size 6 using bitmask backtracking."""
        count = [0]
        
        def backtrack(start, current_mask, size):
            if size == 6:
                count[0] += 1
                return
            remaining = n - start
            if remaining < 6 - size:
                return
            for v in range(start, n):
                if adj_bits[v] & current_mask == 0:
                    backtrack(v + 1, current_mask | (1 << v), size + 1)
        
        backtrack(0, 0, 0)
        return count[0]

    def has_indep_6_bitmask(n, adj_bits):
        """Fast check for independent set of size 6 using bitmask backtracking."""
        best = [0]  # mutable for closure
        
        def backtrack(start, current_mask, size):
            if size == 6:
                best[0] = 1
                return True
            # Pruning: not enough vertices left
            remaining = n - start
            if remaining < 6 - size:
                return False
            # Try vertices from start to n-1
            for v in range(start, n):
                if best[0]:
                    return True
                # Check if v is independent from all in current_mask
                if adj_bits[v] & current_mask == 0:
                    if backtrack(v + 1, current_mask | (1 << v), size + 1):
                        return True
            return False
        
        backtrack(0, 0, 0)
        return best[0] == 1

    def find_indep_sets_5_bitmask(n, adj_bits, max_sets=2000):
        """Find independent sets of size 5 using bitmask backtracking."""
        indep_sets = []
        
        def backtrack(start, current_mask, current_list, size):
            if len(indep_sets) >= max_sets:
                return
            if size == 5:
                indep_sets.append(tuple(current_list))
                return
            for v in range(start, n):
                if len(indep_sets) >= max_sets:
                    return
                if adj_bits[v] & current_mask == 0:
                    current_list.append(v)
                    backtrack(v + 1, current_mask | (1 << v), current_list, size + 1)
                    current_list.pop()
        
        backtrack(0, 0, [], 0)
        return indep_sets

    def find_triangles(n, adj):
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            triangles.append((i, j, k))
        return triangles

    def is_valid(n, edges):
        """Check if graph is K4-free and has no independent set of size 6."""
        adj = build_adj(n, edges)
        adj_bits = build_adj_bit(n, edges)
        
        # K4 check
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    for k in range(j + 1, n):
                        if k in adj[i] and k in adj[j]:
                            common = adj[i] & adj[j] & adj[k]
                            if common:
                                return False
        
        # Independent set of size 6 check
        if has_indep_6_bitmask(n, adj_bits):
            return False
            
        return True

    def walk_sat_improved(n, clauses, max_iters=5000, weights=None):
        """Improved WalkSAT with clause weighting."""
        x = [random.choice([True, False]) for _ in range(n)]
        
        if weights is None:
            weights = [1.0] * len(clauses)
        
        def is_sat(clause):
            for var, sign in clause:
                val = x[var]
                if sign == 1:
                    if val:
                        return True
                else:
                    if not val:
                        return True
            return False

        for it in range(max_iters):
            # Check if all clauses are satisfied
            unsat_indices = []
            for ci, clause in enumerate(clauses):
                if not is_sat(clause):
                    unsat_indices.append(ci)
            
            if not unsat_indices:
                return x
            
            # Pick an unsatisfied clause (weighted random)
            total_w = sum(weights[i] for i in unsat_indices)
            r = random.random() * total_w
            acc = 0
            chosen_clause = clauses[unsat_indices[0]]
            chosen_idx = unsat_indices[0]
            for i in unsat_indices:
                acc += weights[i]
                if r <= acc:
                    chosen_clause = clauses[i]
                    chosen_idx = i
                    break
            
            # Pick a variable from the clause
            var, sign = random.choice(chosen_clause)
            
            # Flip with some probability, otherwise pick random var
            if random.random() < 0.6:
                x[var] = not x[var]
            else:
                var = random.randint(0, n - 1)
                x[var] = not x[var]
            
            # Update weights: increase weight for frequently unsat clauses
            for i in unsat_indices:
                weights[i] *= 1.01
            # Decay all weights slightly
            for i in range(len(weights)):
                weights[i] *= 0.995
        
        return None

    def try_extend(n, adj, adj_bits, edges, max_restarts=5):
        """Try to add one vertex. Returns new (n, adj, adj_bits, edges) or None."""
        triangles = find_triangles(n, adj)
        indep_sets_5 = find_indep_sets_5_bitmask(n, adj_bits, max_sets=3000)
        
        clauses = []
        for tri in triangles:
            u, v, w = tri
            clauses.append([(u, -1), (v, -1), (w, -1)])
        for ind in indep_sets_5:
            u, v, w, x, y = ind
            clauses.append([(u, 1), (v, 1), (w, 1), (x, 1), (y, 1)])
        
        if not clauses:
            assignment = [False] * n
        else:
            assignment = None
            for _ in range(max_restarts):
                assignment = walk_sat_improved(n, clauses, max_iters=4000)
                if assignment is not None:
                    break
            
            if assignment is None:
                # Greedy fallback: try to find assignment that satisfies most clauses
                best_x = None
                best_score = -1
                for _ in range(10):
                    x = [random.choice([True, False]) for _ in range(n)]
                    score = 0
                    for clause in clauses:
                        sat = False
                        for var, sign in clause:
                            val = x[var]
                            if sign == 1:
                                if val:
                                    sat = True
                                    break
                            else:
                                if not val:
                                    sat = True
                                    break
                        if sat:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_x = x[:]
                
                if best_x is not None and best_score == len(clauses):
                    assignment = best_x
                else:
                    # Last resort: greedy repair
                    x = [True] * n
                    changed = True
                    iterations = 0
                    while changed and iterations < 1000:
                        changed = False
                        iterations += 1
                        for tri in triangles:
                            u, v, w = tri
                            if x[u] and x[v] and x[w]:
                                idx = random.choice([u, v, w])
                                x[idx] = False
                                changed = True
                        for ind in indep_sets_5:
                            u, v, w, a, b = ind
                            if not x[u] and not x[v] and not x[w] and not x[a] and not x[b]:
                                idx = random.choice([u, v, w, a, b])
                                x[idx] = True
                                changed = True
                    # Verify
                    valid = True
                    for clause in clauses:
                        sat = False
                        for var, sign in clause:
                            val = x[var]
                            if sign == 1:
                                if val:
                                    sat = True
                                    break
                            else:
                                if not val:
                                    sat = True
                                    break
                        if not sat:
                            valid = False
                            break
                    if valid:
                        assignment = x
        
        if assignment is None:
            return None
        
        # Verify
        valid = True
        for c in clauses:
            sat = False
            for var, sign in c:
                val = assignment[var]
                if sign == 1:
                    if val:
                        sat = True
                        break
                else:
                    if not val:
                        sat = True
                        break
            if not sat:
                valid = False
                break
        
        if not valid:
            return None
        
        new_n = n + 1
        new_edges = edges[:]
        for i in range(n):
            if assignment[i]:
                new_edges.append([i, new_n - 1])
        
        new_adj = build_adj(new_n, new_edges)
        new_adj_bits = build_adj_bit(new_n, new_edges)
        return (new_n, new_adj, new_adj_bits, new_edges)

    def local_search_polish(n, adj, adj_bits, edges, time_limit=5):
        """Try to add edges or remove edges to enable further extension.
        Returns improved (n, adj, adj_bits, edges) or original if no improvement."""
        start = time.time()
        best_n = n
        best_edges = edges[:]
        best_adj = [s.copy() for s in adj]
        best_adj_bits = adj_bits[:]
        
        for attempt in range(30):
            if time.time() - start > time_limit:
                break
            if time.time() > deadline - 10:
                break
            
            # Strategy 1: Remove a random edge
            if edges and random.random() < 0.5:
                idx = random.randint(0, len(edges) - 1)
                new_edges = edges[:]
                new_edges.pop(idx)
                
                new_adj = build_adj(n, new_edges)
                new_adj_bits = build_adj_bit(n, new_edges)
                
                if is_valid(n, new_edges):
                    result = try_extend(n, new_adj, new_adj_bits, new_edges, max_restarts=3)
                    if result is not None:
                        new_n, new_adj, new_adj_bits, new_edges = result
                        if new_n > best_n:
                            best_n = new_n
                            best_edges = new_edges
                            best_adj = new_adj
                            best_adj_bits = new_adj_bits
            
            # Strategy 2: Add a random non-edge
            else:
                non_adj_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if j not in adj[i]:
                            non_adj_pairs.append((i, j))
                if non_adj_pairs:
                    u, v = random.choice(non_adj_pairs)
                    test_edges = edges + [[u, v]]
                    test_adj = build_adj(n, test_edges)
                    test_adj_bits = build_adj_bit(n, test_edges)
                    
                    if is_valid(n, test_edges):
                        result = try_extend(n, test_adj, test_adj_bits, test_edges, max_restarts=3)
                        if result is not None:
                            new_n, new_adj, new_adj_bits, new_edges = result
                            if new_n > best_n:
                                best_n = new_n
                                best_edges = new_edges
                                best_adj = new_adj
                                best_adj_bits = new_adj_bits
        
        if best_n > n:
            return (best_n, best_adj, best_adj_bits, best_edges)
        return (n, adj, adj_bits, edges)

    def ga_fitness(n, adj_bits):
        """
        Genetic algorithm fitness: n - large penalty for violations.
        Uses smooth surrogate via log-sum-exp style penalty.
        """
        k4_count = count_k4(n, adj_bits)
        indep6_count = count_indep_6(n, adj_bits)
        # Penalty: exponential scaling makes violations very costly
        penalty = 100 * (k4_count + indep6_count)
        return n - penalty

    def ga_crossover(parent1_bits, parent2_bits, n):
        """Uniform crossover on adjacency bitmasks."""
        child = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                bit = 1 << j
                # Randomly pick edge presence from parent1 or parent2
                if random.random() < 0.5:
                    if parent1_bits[i] & bit:
                        child[i] |= bit
                        child[j] |= 1 << i
                else:
                    if parent2_bits[i] & bit:
                        child[i] |= bit
                        child[j] |= 1 << i
        return child

    def ga_mutation(bits, n, rate=0.02):
        """Random edge flip mutation."""
        child = bits[:]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < rate:
                    bit = 1 << j
                    if child[i] & bit:
                        child[i] &= ~bit
                        child[j] &= ~(1 << i)
                    else:
                        child[i] |= bit
                        child[j] |= 1 << i
        return child

    def ga_tournament_select(population, fitnesses, k=3):
        """Tournament selection."""
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best_idx = max(candidates, key=lambda i: fitnesses[i])
        return population[best_idx][:]

    def genetic_algorithm(initial_n, initial_edges, pop_size=40, generations=200, time_limit=30):
        """
        Genetic algorithm operating on adjacency matrices.
        Fitness = n - penalty for K4 and independent 6-set violations.
        """
        start = time.time()
        
        # Initialize population
        initial_adj = build_adj(initial_n, initial_edges)
        initial_bits = build_adj_bit(initial_n, initial_edges)
        
        # Create diverse initial population
        population = []
        for _ in range(pop_size):
            if _ == 0:
                pop = initial_bits[:]
            else:
                pop = initial_bits[:]
                # Mutate to create diversity
                pop = ga_mutation(pop, initial_n, rate=0.05)
            population.append(pop)
        
        # Evaluate initial fitness
        fitnesses = [ga_fitness(initial_n, pop) for pop in population]
        
        best_fitness = max(fitnesses)
        best_pop = population[fitnesses.index(best_fitness)][:]
        
        for gen in range(generations):
            if time.time() - start > time_limit:
                break
            if time.time() > deadline - 15:
                break
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individual
            best_idx = fitnesses.index(max(fitnesses))
            new_population.append(population[best_idx][:])
            
            # Fill rest with crossover + mutation
            while len(new_population) < pop_size:
                p1 = ga_tournament_select(population, fitnesses, k=3)
                p2 = ga_tournament_select(population, fitnesses, k=3)
                child = ga_crossover(p1, p2, initial_n)
                child = ga_mutation(child, initial_n, rate=0.03)
                new_population.append(child)
            
            population = new_population
            fitnesses = [ga_fitness(initial_n, pop) for pop in population]
            
            # Track best
            gen_best = max(fitnesses)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_pop = population[fitnesses.index(best_fitness)][:]
        
        # Convert best to edges
        best_adj = build_adj(initial_n, [])
        best_edges = []
        for i in range(initial_n):
            for j in range(i + 1, initial_n):
                if best_pop[i] & (1 << j):
                    best_edges.append([i, j])
        
        return {'n': initial_n, 'edges': best_edges}

    def ga_with_extension(initial_n, initial_edges, time_limit=40):
        """
        Run GA to optimize a graph of fixed size, then try to extend it.
        """
        start = time.time()
        
        # Run GA to find good structure
        ga_result = genetic_algorithm(initial_n, initial_edges, pop_size=30, 
                                       generations=150, time_limit=min(20, time_limit - 10))
        
        if time.time() - start > time_limit - 5:
            return ga_result
        
        # Verify and try to extend
        n = ga_result['n']
        edges = ga_result['edges']
        
        if not is_valid(n, edges):
            # Fall back to original
            return {'n': initial_n, 'edges': initial_edges}
        
        adj = build_adj(n, edges)
        adj_bits = build_adj_bit(n, edges)
        
        # Try to extend
        result = try_extend(n, adj, adj_bits, edges, max_restarts=5)
        if result is not None:
            new_n, new_adj, new_adj_bits, new_edges = result
            return {'n': new_n, 'edges': new_edges}
        
        return ga_result

    # Main search
    best_g = None
    best_n = 0
    
    # Strategy 1: Paley + extension (original approach)
    for q in [29, 17, 13, 5]:
        if time.time() > deadline - 40:
            break
        
        G = make_paley(q)
        if G is None:
            continue
        
        n = G['n']
        edges = G['edges']
        
        if not is_valid(n, edges):
            continue
        
        adj = build_adj(n, edges)
        adj_bits = build_adj_bit(n, edges)
        
        if n > best_n:
            best_g = {'n': n, 'edges': edges[:]}
            best_n = n
        
        # Extension loop
        extended = True
        while extended and time.time() < deadline - 35:
            result = try_extend(n, adj, adj_bits, edges, max_restarts=5)
            if result is None:
                extended = False
            else:
                n, adj, adj_bits, edges = result
                if n > best_n:
                    best_g = {'n': n, 'edges': edges[:]}
                    best_n = n
        
        # Local search polish to escape local optima
        if time.time() < deadline - 30:
            polished_n, polished_adj, polished_adj_bits, polished_edges = local_search_polish(
                n, adj, adj_bits, edges, time_limit=5
            )
            if polished_n > best_n:
                best_g = {'n': polished_n, 'edges': polished_edges}
                best_n = polished_n
            else:
                # Continue extending from polished state if possible
                n2, adj2, adj_bits2, edges2 = polished_n, polished_adj, polished_adj_bits, polished_edges
                extended2 = True
                while extended2 and time.time() < deadline - 25:
                    result = try_extend(n2, adj2, adj_bits2, edges2, max_restarts=5)
                    if result is None:
                        extended2 = False
                    else:
                        n2, adj2, adj_bits2, edges2 = result
                        if n2 > best_n:
                            best_g = {'n': n2, 'edges': edges2[:]}
                            best_n = n2
        
        if best_n >= 35:
            break  # Can't do much better than 35
    
    # Strategy 2: Genetic algorithm optimization
    if time.time() < deadline - 20:
        # Start from best Paley-derived graph
        ga_start_n = min(29, best_n if best_n > 0 else 29)
        if best_g is not None and best_g['n'] <= 35:
            ga_start_n = best_g['n']
            ga_start_edges = best_g['edges']
        else:
            G = make_paley(29)
            ga_start_n = G['n'] if G else 29
            ga_start_edges = G['edges'] if G else []
        
        ga_result = ga_with_extension(ga_start_n, ga_start_edges, time_limit=20)
        
        if ga_result and ga_result['n'] > best_n:
            if is_valid(ga_result['n'], ga_result['edges']):
                best_g = ga_result
                best_n = ga_result['n']
    
    # Strategy 3: GA from scratch with random seed
    if time.time() < deadline - 10 and best_n < 35:
        random.seed(42)
        random_n = 30
        random_edges = []
        for i in range(random_n):
            for j in range(i + 1, random_n):
                if random.random() < 0.3:
                    random_edges.append([i, j])
        
        ga_result2 = ga_with_extension(random_n, random_edges, time_limit=10)
        
        if ga_result2 and ga_result2['n'] > best_n:
            if is_valid(ga_result2['n'], ga_result2['edges']):
                best_g = ga_result2
                best_n = ga_result2['n']
    
    if best_g is None:
        return {'n': 1, 'edges': []}
    
    # Final verification
    if not is_valid(best_g['n'], best_g['edges']):
        return {'n': 1, 'edges': []}
    
    return best_g
