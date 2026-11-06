import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 140  # Leave margin for 150s limit
    
    # Precompute edge index for faster lookup
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = (1 << u) | (1 << v)
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges_total = len(all_edges)  # 190
    
    # Helper to check if adding edge (u,v) to adj is valid (no K2,3)
    # K2,3-free means: no two vertices share 3+ common neighbors
    # When adding edge (u,v): 
    #   - |N(u) ∩ N(v)| must be <= 2 (they become adjacent, so they can't share 3+ neighbors)
    #   - For any w in N(u)\{v}, |N(w) ∩ N(v)| must be <= 1 (w and v can't share 2+ neighbors since w is already adjacent to u)
    #   - For any w in N(v)\{u}, |N(w) ∩ N(u)| must be <= 1
    def can_add(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        # Check: |N(u) ∩ N(v)| <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # For each w in N(u) \ {v}, check |N(w) ∩ N(v)| <= 1
        # Note: w is adjacent to u, so if w and v share 2+ neighbors, those neighbors + u,v form issues
        # Actually the constraint is: no pair of vertices has 3+ common neighbors
        # After adding (u,v): 
        #   - u and v have common_uv common neighbors, need <= 2 ✓
        #   - For w in N(u): w and v share N(w) ∩ N(v). Since u is in both N(w) and N(v) (after adding), 
        #     we need |N(w) ∩ N(v)| <= 2. But N(w) ∩ N(v) includes u, so we need |N(w) ∩ N(v) \ {u}| <= 1
        #   - Similarly for w in N(v): need |N(w) ∩ N(u) \ {v}| <= 1
        
        mask = adj[u] & ~(1 << v)
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            # N(w) ∩ N(v) includes u (since w~u and u~v now)
            # So |N(w) ∩ N(v)| = 1 + |N(w) ∩ N(v) \ {u}|
            # We need |N(w) ∩ N(v)| <= 2, i.e., |N(w) ∩ N(v) \ {u}| <= 1
            common_wv = (adj[w] & adj[v]) & ~(1 << u)
            if common_wv.bit_count() > 1:
                return False
            mask ^= lb
        
        mask = adj[v] & ~(1 << u)
        while mask:
            lb = mask & (-mask)
            w = lb.bit_length() - 1
            common_wu = (adj[w] & adj[u]) & ~(1 << v)
            if common_wu.bit_count() > 1:
                return False
            mask ^= lb
        
        return True
    
    # Check entire graph for K2,3
    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() >= 3:
                    return False
        return True
    
    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lb
        return edges
    
    # Greedy construction with random ordering
    def greedy():
        adj = [0] * n
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        count = 0
        for u, v in edge_list:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        return adj, count
    
    # Local search: try adding missing edges
    def local_add(adj, count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
                    improved = True
        return adj, count
    
    # Edge swap: remove one, try to add up to two
    def local_swap(adj, count, max_iters=200):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if count == 0:
                break
            
            # Pick random edge to remove
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add two random edges
            added = 0
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
            
            if added == 0:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # 2-opt style: remove two edges, try to add three
    def local_swap2(adj, count, max_iters=100):
        for _ in range(max_iters):
            if time.time() > deadline - 2:
                break
            if count < 2:
                break
            
            # Pick two random edges to remove
            e1 = random.choice(all_edges)
            e2 = random.choice(all_edges)
            if e1 == e2:
                continue
            if not (adj[e1[0]] & (1 << e1[1])):
                continue
            if not (adj[e2[0]] & (1 << e2[1])):
                continue
            
            # Remove both
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            count -= 2
            
            # Try to add three edges
            added = 0
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            for a, b in edge_list:
                if added >= 3:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
            
            if added < 1:
                # Restore both
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                count += 2
        
        return adj, count
    
    # Degree-based greedy: prioritize adding edges between low-degree vertices
    def degree_greedy():
        adj = [0] * n
        degrees = [0] * n
        edge_list = all_edges[:]
        count = 0
        remaining = set(all_edges)
        while remaining:
            best_edge = None
            best_score = float('inf')
            sample_size = min(len(remaining), 50)
            sample = random.sample(list(remaining), sample_size)
            for u, v in sample:
                score = degrees[u] + degrees[v]
                if score < best_score:
                    best_score = score
                    best_edge = (u, v)
            if best_edge is None:
                break
            u, v = best_edge
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                count += 1
            remaining.discard(best_edge)
        return adj, count
    
    # Simulated annealing-like: accept worse moves with probability
    def simulated_annealing(adj, count, steps=5000):
        T = 1.0
        T_min = 0.01
        for i in range(steps):
            if time.time() > deadline - 2:
                break
            T = T_min + (T - T_min) * (1 - i / steps)
            
            # Pick random edge swap
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
            
            # Try to add two
            added = 0
            edge_list = random.sample(all_edges, min(50, num_edges_total))
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add(adj, a, b):
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    count += 1
                    added += 1
            
            if added == 0:
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    # Genetic Algorithm with edge-bitset chromosomes
    def ga_crossover(adj1, adj2):
        """Smarter crossover: prioritize edges present in both parents, then try others."""
        adj = [0] * n
        count = 0
        
        # Collect edges from both parents
        edges1 = set()
        edges2 = set()
        for u in range(n):
            mask = adj1[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    edges1.add((u, v))
                mask ^= lb
            mask = adj2[u]
            while mask:
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if u < v:
                    edges2.add((u, v))
                mask ^= lb
        
        # Edges in both parents (high confidence)
        common = edges1 & edges2
        # Edges in only one parent
        only1 = edges1 - edges2
        only2 = edges2 - edges1
        
        # Try common edges first
        for u, v in common:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        # Then try edges from only one parent (randomized)
        remaining = list(only1) + list(only2)
        random.shuffle(remaining)
        for u, v in remaining:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        # Finally, try to add any other missing edges
        for u, v in all_edges:
            if not (adj[u] & (1 << v)):
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    count += 1
        
        return adj, count
    
    def ga_mutation(adj, count, mutation_rate=0.05):
        """Mutation: randomly remove and add edges."""
        # Remove some random edges
        num_remove = max(1, int(count * mutation_rate))
        for _ in range(num_remove):
            if count <= 0:
                break
            u = random.randint(0, n - 1)
            mask = adj[u]
            if not mask:
                continue
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            if v < u:
                continue
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            count -= 1
        
        # Add some random edges
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        for u, v in edge_list:
            if adj[u] & (1 << v):
                continue
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                count += 1
        
        return adj, count
    
    def genetic_algorithm(pop_size=20, generations=50):
        """Full GA with selection, crossover, mutation, and local search polish."""
        # Initialize population with diverse starts
        population = []
        for _ in range(pop_size):
            if random.random() < 0.5:
                adj, count = greedy()
            else:
                adj, count = degree_greedy()
            adj, count = local_add(adj, count)
            population.append((adj, count))
        
        # Sort by count descending
        population.sort(key=lambda x: x[1], reverse=True)
        
        best_adj = population[0][0][:]
        best_count = population[0][1]
        
        for gen in range(generations):
            if time.time() > deadline - 10:
                break
            
            new_population = []
            
            # Elitism: keep top 3
            for i in range(min(3, len(population))):
                new_population.append((population[i][0][:], population[i][1]))
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                candidates = random.sample(population, min(4, len(population)))
                parent1 = max(candidates, key=lambda x: x[1])[0]
                candidates = random.sample(population, min(4, len(population)))
                parent2 = max(candidates, key=lambda x: x[1])[0]
                
                # Crossover
                child_adj, child_count = ga_crossover(parent1, parent2)
                
                # Mutation
                child_adj, child_count = ga_mutation(child_adj, child_count, mutation_rate=0.04)
                
                # Local search polish
                child_adj, child_count = local_add(child_adj, child_count)
                
                new_population.append((child_adj, child_count))
            
            # Sort new population
            new_population.sort(key=lambda x: x[1], reverse=True)
            population = new_population
            
            # Update best
            if population[0][1] > best_count:
                best_count = population[0][1]
                best_adj = population[0][0][:]
        
        return best_adj, best_count
    
    # Main search loop
    best_count = 0
    best_adj = None
    
    # Phase 1: Many diverse greedy starts
    start_time = time.time()
    num_greedy = 30
    for i in range(num_greedy):
        if time.time() > deadline - 10:
            break
        adj, count = greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 2: Degree-based greedy
    for i in range(10):
        if time.time() > deadline - 10:
            break
        adj, count = degree_greedy()
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Phase 3: Intensive local search on best
    if best_adj is not None:
        adj = best_adj[:]
        count = best_count
        
        # Multiple rounds of improvement
        for round_num in range(10):
            if time.time() > deadline - 5:
                break
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap(adj, count, max_iters=300)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            
            adj, count = local_swap2(adj, count, max_iters=100)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 4: Genetic Algorithm
    if time.time() < deadline - 30:
        ga_adj, ga_count = genetic_algorithm(pop_size=18, generations=40)
        if ga_count > best_count:
            best_count = ga_count
            best_adj = ga_adj[:]
        
        # Polish the GA result
        if best_adj is not None:
            adj = best_adj[:]
            count = best_count
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap(adj, count, max_iters=300)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_add(adj, count)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
            adj, count = local_swap2(adj, count, max_iters=150)
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 5: Restart from best with perturbations
    if best_adj is not None and time.time() < deadline - 15:
        for i in range(20):
            if time.time() > deadline - 5:
                break
            adj = best_adj[:]
            count = best_count
            
            # Perturb: remove 2-5 random edges
            remove_count = random.randint(2, 5)
            for _ in range(remove_count):
                if count <= 0:
                    break
                u = random.randint(0, n - 1)
                mask = adj[u]
                if not mask:
                    continue
                lb = mask & (-mask)
                v = lb.bit_length() - 1
                if v < u:
                    continue
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                count -= 1
            
            # Improve
            adj, count = local_add(adj, count)
            adj, count = local_swap(adj, count, max_iters=250)
            adj, count = local_add(adj, count)
            
            if count > best_count:
                best_count = count
                best_adj = adj[:]
    
    # Phase 6: Final intensive polish
    if best_adj is not None and time.time() < deadline - 3:
        adj = best_adj[:]
        count = best_count
        
        # Aggressive final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swaps
        adj, count = local_swap(adj, count, max_iters=600)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # Final swap2
        adj, count = local_swap2(adj, count, max_iters=200)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # One more final add
        adj, count = local_add(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to edge list
    if best_adj is None:
        return []
    
    return adj_to_edges(best_adj)
