import time
import numpy as np
import random

def construct():
    n = 20
    deadline = time.time() + 145.0
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m_total = len(all_edges)
    
    # Fast K3,3 check using bitmasks
    # adj[i] is an integer bitmask of neighbors of i
    # Condition: For all u < v < w, popcount(adj[u] & adj[v] & adj[w]) <= 2
    
    def is_valid(adj):
        for u in range(n):
            for v in range(u + 1, n):
                common_uv = adj[u] & adj[v]
                if common_uv == 0:
                    continue
                for w in range(v + 1, n):
                    if (common_uv & adj[w]).bit_count() > 2:
                        return False
        return True
        
    def can_add(adj, u, v):
        # Check if adding edge (u,v) violates K3,3-free property
        # Only need to check triples containing u or v
        # Case 1: Triple {u, x, y} where x,y != v.
        # Violation if |N(u) ∩ N(x) ∩ N(y)| == 2 AND v ∈ N(x) ∩ N(y)
        # Case 2: Triple {v, x, y} where x,y != u.
        # Violation if |N(v) ∩ N(x) ∩ N(y)| == 2 AND u ∈ N(x) ∩ N(y)
        
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                
                # Check if v is neighbor of both x and y
                if (adj[x] & adj[y]) & (1 << v):
                    if (adj[u] & adj[x] & adj[y]).bit_count() >= 2:
                        return False
                
                # Check if u is neighbor of both x and y
                if (adj[x] & adj[y]) & (1 << u):
                    if (adj[v] & adj[x] & adj[y]).bit_count() >= 2:
                        return False
                        
        return True

    def repair_and_fill(adj, edges_set):
        """
        Given a graph (possibly invalid), remove edges to make it valid,
        then greedily add edges to maximize count.
        Returns (adj_list, edges_set, count)
        """
        # Phase 1: Repair (Remove violating edges)
        # We iterate until no violations
        max_iter = 1000
        for _ in range(max_iter):
            found_violation = False
            # Find a violation
            # We scan triples
            for u in range(n):
                if found_violation: break
                for v in range(u + 1, n):
                    if found_violation: break
                    common_uv = adj[u] & adj[v]
                    if common_uv == 0: continue
                    for w in range(v + 1, n):
                        common_uvw = common_uv & adj[w]
                        if common_uvw.bit_count() >= 3:
                            # Found K3,3
                            # Remove a random edge from the K3,3 structure
                            # The K3,3 is between {u,v,w} and the 3+ neighbors in common_uvw
                            # We pick a random neighbor z from common_uvw
                            c_list = [i for i in range(n) if (common_uvw >> i) & 1]
                            if not c_list: continue
                            z = c_list[random.randint(0, len(c_list)-1)]
                            # Pick a random vertex from {u,v,w}
                            a = [u, v, w][random.randint(0, 2)]
                            # Remove edge (a, z)
                            ea, ez = min(a, z), max(a, z)
                            if (ea, ez) in edges_set:
                                edges_set.remove((ea, ez))
                                adj[ea] &= ~(1 << ez)
                                adj[ez] &= ~(1 << ea)
                                found_violation = True
                                break
            if not found_violation:
                break
                
        # Phase 2: Greedy Fill
        # Try to add all missing edges in random order
        missing = [(i, j) for i, j in all_edges if (i, j) not in edges_set]
        random.shuffle(missing)
        
        for u, v in missing:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add((u, v))
                
        return adj, edges_set, len(edges_set)

    # Genetic Algorithm Setup
    pop_size = 60
    population = []
    
    # Initialize Population
    for i in range(pop_size):
        adj = [0] * n
        edges_set = set()
        # Random initialization with moderate density
        for e in all_edges:
            if random.random() < 0.5:
                u, v = e
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges_set.add(e)
        
        adj, edges_set, count = repair_and_fill(adj, edges_set)
        population.append((count, adj, edges_set))
        
    # Sort by count descending
    population.sort(key=lambda x: x[0], reverse=True)
    
    # Track best
    if population[0][0] > best_count:
        best_count = population[0][0]
        best_edges = [list(e) for e in population[0][2]]
        
    # Evolution Loop
    while time.time() < deadline - 5.0:
        new_population = []
        
        # Elitism: Keep top 10
        for i in range(min(10, len(population))):
            new_population.append(population[i])
            
        # Generate Offspring to fill the rest
        while len(new_population) < pop_size:
            if time.time() > deadline - 5.0:
                break
                
            # Selection: Pick two parents from top 30
            p1_idx = random.randint(0, min(29, len(population)-1))
            p2_idx = random.randint(0, min(29, len(population)-1))
            
            _, adj1, edges1 = population[p1_idx]
            _, adj2, edges2 = population[p2_idx]
            
            # Crossover: Bitwise OR (union of edges)
            # This creates a dense graph, which will likely have violations,
            # so repair_and_fill will do the heavy lifting of pruning.
            adj_new = [a | b for a, b in zip(adj1, adj2)]
            edges_new = edges1.union(edges2)
            
            # Mutation: Flip some random edges
            # Probability of mutation
            if random.random() < 0.5:
                for _ in range(random.randint(5, 15)):
                    e = all_edges[random.randint(0, m_total - 1)]
                    u, v = e
                    if (u, v) in edges_new:
                        edges_new.remove((u, v))
                        adj_new[u] &= ~(1 << v)
                        adj_new[v] &= ~(1 << u)
                    else:
                        edges_new.add((u, v))
                        adj_new[u] |= (1 << v)
                        adj_new[v] |= (1 << u)
            
            # Repair and Fill
            adj_new, edges_new, count_new = repair_and_fill(adj_new, edges_new)
            
            new_population.append((count_new, adj_new, edges_new))
            
        # Replace population
        population = new_population
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if population[0][0] > best_count:
            best_count = population[0][0]
            best_edges = [list(e) for e in population[0][2]]
            
        # Early stop if we reach a high bound
        # ex(20, K3,3) is known to be around 50-60. 
        # Let's aim for 55+.
        if best_count >= 58:
            break
            
    # Final verification
    if best_edges:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        if is_valid(adj_final):
            return best_edges
            
    # Fallback
    return []
