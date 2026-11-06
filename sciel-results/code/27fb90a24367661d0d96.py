import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 115s.
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def repair_union(adj1, adj2):
        """
        Take the union of two valid graphs and greedily remove edges to fix K2,3 violations.
        Returns (adj, count) of a valid graph.
        """
        # Start with union
        adj = [0] * n
        for i in range(n):
            adj[i] = adj1[i] | adj2[i]
        
        # Count edges
        count = sum(adj[i].bit_count() for i in range(n)) // 2
        
        # Find all pairs with >2 common neighbors
        # For each violating pair (u,v), we need to remove edges from their common neighbors
        # Strategy: iteratively find violations and remove random edges to fix them
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Find all violating pairs
            violations = []
            for u in range(n):
                for v in range(u+1, n):
                    common = adj[u] & adj[v]
                    if common.bit_count() > 2:
                        violations.append((u, v, common))
            
            if not violations:
                break
            
            # Pick a random violation
            u, v, common = random.choice(violations)
            
            # We need to reduce common neighbors of u and v to <= 2
            # Remove edges from common neighbors to either u or v
            common_list = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                common_list.append(w)
                mask ^= lsb
            
            # Remove a random edge from a common neighbor
            w = random.choice(common_list)
            # Decide whether to remove (u,w) or (v,w)
            if random.random() < 0.5:
                # Remove (u, w)
                adj[u] &= ~(1 << w)
                adj[w] &= ~(1 << u)
            else:
                # Remove (v, w)
                adj[v] &= ~(1 << w)
                adj[w] &= ~(1 << v)
            count -= 1
            improved = True
        
        return adj, count
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        adj is a list of integers (bitmasks) where adj[i] has bit j set if (i,j) is an edge.
        
        Constraints for adding (u, v):
        1. u and v must not already be connected.
        2. The number of common neighbors of u and v must be <= 2.
        3. For any neighbor w of u (w != v), |N(v) \cap N(w)| <= 1.
        4. For any neighbor w of v (w != u), |N(u) \cap N(w)| <= 1.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 3. Check constraints on neighbors of u
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 4. Check constraints on neighbors of v
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        """
        Try to add as many edges as possible from the missing list.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count
    
    def greedy_fill_from_union(adj1, adj2):
        """
        Start with union of two graphs, repair violations, then greedily fill.
        """
        adj, count = repair_union(adj1, adj2)
        count = greedy_fill(adj, count)
        return adj, count

    def simulated_annealing(adj, count):
        """
        Simulated annealing to escape local optima.
        """
        if time.time() > deadline - 5:
            return count
        
        T = 2.0
        cooling_rate = 0.9995
        min_T = 0.05
        
        # Pre-compute masks for speed? No, adj changes.
        
        while T > min_T and time.time() < deadline - 2:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        # Accept improvement
                        # T *= cooling_rate # Don't cool on success? Or cool slowly.
                    else:
                        # Invalid addition, skip
                        pass
            elif move_type < 0.8:
                # Try to remove a present edge and then refill
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge(adj, u, v)
                    count -= 1
                    
                    # Try to greedily add back edges
                    # We allow adding multiple edges to potentially recover or exceed count
                    # This is a "kick" move
                    count = greedy_fill(adj, count)
            else:
                # Swap: Remove one, add one specific missing
                present = get_present_edges(adj)
                missing = get_missing_edges(adj)
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    
                    # Temporarily remove
                    remove_edge(adj, u1, v1)
                    count -= 1
                    
                    # Try to add
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        count += 1
                        # Count stays same, but structure changed.
                        # Accept with some probability? 
                        # Since we want to maximize edges, keeping same count is neutral.
                        # But it might open up future improvements.
                        # Let's always accept neutral moves to explore.
                    else:
                        # Revert
                        add_edge(adj, u1, v1)
                        count += 1
            
            T *= cooling_rate
            
        return count

    # Phase 1: Build a population of diverse solutions
    population = []
    
    # Generate initial population using random greedy fill
    pop_size = 10
    for _ in range(pop_size):
        if time.time() > deadline - 30:
            break
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        count = greedy_fill(adj, count)
        population.append((adj[:], count))
    
    # Sort by count descending
    population.sort(key=lambda x: -x[1])
    
    if population:
        best_count = max(best_count, population[0][1])
        best_adj = population[0][0][:]
    
    # Phase 2: Genetic algorithm with crossover
    ga_generations = 0
    while time.time() < deadline - 20 and population:
        ga_generations += 1
        
        # Select parents (tournament selection)
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, min(k, len(pop)))
            return max(candidates, key=lambda x: x[1])
        
        new_population = []
        
        # Elitism: keep top solutions
        for adj, count in population[:2]:
            new_population.append((adj[:], count))
        
        # Generate offspring via crossover
        while len(new_population) < pop_size and time.time() < deadline - 20:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover: union + repair
            child_adj, child_count = greedy_fill_from_union(parent1[0], parent2[0])
            new_population.append((child_adj, child_count))
            
            # Occasionally do pure mutation instead of crossover
            if random.random() < 0.3 and len(new_population) < pop_size:
                parent = random.choice(population)
                adj = parent[0][:]
                count = parent[1]
                
                # Random mutation: remove a few edges and refill
                present = get_present_edges(adj)
                if present:
                    for _ in range(random.randint(1, 3)):
                        if not present:
                            break
                        u, v = random.choice(present)
                        remove_edge(adj, u, v)
                        count -= 1
                        present.remove((u, v))
                
                count = greedy_fill(adj, count)
                new_population.append((adj, count))
        
        # Evaluate new population
        for adj, count in new_population:
            if count > best_count:
                best_count = count
                best_adj = adj[:]
        
        # Update population (keep best)
        population = sorted(new_population, key=lambda x: -x[1])[:pop_size]
    
    # Phase 3: Simulated annealing on best solution
    if best_count > 0 and time.time() < deadline - 10:
        adj = best_adj[:]
        count = best_count
        count = simulated_annealing(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]

    # Construct the output list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
    
    # Final verification
    if edges:
        verify_adj = [0] * n
        for u, v in edges:
            verify_adj[u] |= (1 << v)
            verify_adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u+1, n):
                if (verify_adj[u] & verify_adj[v]).bit_count() > 2:
                    # This shouldn't happen, but if it does, we need to fix
                    # For safety, fall back to a simple valid construction
                    adj = [0] * n
                    count = 0
                    for u2, v2 in all_edges:
                        if is_valid_add(adj, u2, v2):
                            add_edge(adj, u2, v2)
                            count += 1
                    best_adj = adj
                    break
    
    # Reconstruct edges from best_adj
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    res = construct()
    print(f"Found {len(res)} edges in {time.time() - start:.2f}s")
    # Verify validity
    n = 20
    adj = [0] * n
    for u, v in res:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = adj[u] & adj[v]
            if common.bit_count() > 2:
                valid = False
                print(f"Invalid: {u}, {v} have {common.bit_count()} common neighbors")
                break
        if not valid:
            break
    
    print(f"Valid: {valid}")
