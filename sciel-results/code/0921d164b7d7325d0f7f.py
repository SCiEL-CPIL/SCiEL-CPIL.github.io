import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 20
    num_pairs = n * (n - 1) // 2
    pair_indices = []
    pair_u = []
    pair_v = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append(idx)
            pair_u.append(i)
            pair_v.append(j)
            idx += 1
    
    pair_u = np.array(pair_u)
    pair_v = np.array(pair_v)

    # Precompute common neighbors for each pair of vertices
    # common_neighbors[i, j] is a bitmask of vertices adjacent to both i and j
    # This is O(n^2 * 2^n) if done naively, but n=20 is small.
    # Actually, we can compute it on the fly or precompute a lookup table for pairs of vertices.
    # Let's precompute a 20x20 matrix of bitmasks? No, the mask depends on the graph.
    # We need to check validity of a graph.
    
    # A graph is K2,3-free if for every pair of vertices (u, v), the number of common neighbors <= 2.
    # common_count(u, v) = popcount(adj[u] & adj[v])
    
    # To make GA fast, we need a fast validity checker.
    # Represent graph as a list of 20 bitmasks (integers).
    # Checking validity:
    # for u in range(n):
    #   for v in range(u+1, n):
    #     if bin(adj[u] & adj[v]).count('1') > 2: return False
    
    # This is 190 checks per graph. Each check is a bitwise AND and popcount.
    # Python integers are fast for this.
    # Let's optimize the popcount using a precomputed table for 16-bit chunks or just bin().count('1').
    # For n=20, bin().count('1') is likely fast enough.
    
    def is_valid(adj):
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if bin(au & adj[v]).count('1') > 2:
                    return False
        return True

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_fill(adj):
        """
        Greedily add edges to adj until no more can be added.
        Returns the modified adj and the number of edges added.
        """
        added = 0
        # Sort pairs by some heuristic? For now, random or fixed order.
        # Fixed order is faster.
        for idx in range(num_pairs):
            u = pair_u[idx]
            v = pair_v[idx]
            if not ((adj[u] >> v) & 1):
                # Check if adding (u, v) is valid
                # 1. common neighbors of u, v must be <= 2
                if bin(adj[u] & adj[v]).count('1') > 2:
                    continue
                # 2. For all w in N(u), common neighbors of v, w must be <= 2
                #    For all w in N(v), common neighbors of u, w must be <= 2
                valid = True
                # Check neighbors of u
                nu = adj[u]
                while nu:
                    low = nu & (-nu)
                    w = low.bit_length() - 1
                    if w != v:
                        if bin(adj[v] & adj[w]).count('1') > 2:
                            valid = False
                            break
                    nu ^= low
                if not valid:
                    continue
                # Check neighbors of v
                nv = adj[v]
                while nv:
                    low = nv & (-nv)
                    w = low.bit_length() - 1
                    if w != u:
                        if bin(adj[u] & adj[w]).count('1') > 2:
                            valid = False
                            break
                    nv ^= low
                if valid:
                    add_edge(adj, u, v)
                    added += 1
        return added

    def local_search(adj, max_iters=50):
        """
        Simple local search: try to swap edges to increase count.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        current_count = best_count
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Get edges and non-edges
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
            
            if not edges or not non_edges:
                break
            
            # Try a random swap
            e1 = edges[random.randint(0, len(edges) - 1)]
            e2 = non_edges[random.randint(0, len(non_edges) - 1)]
            
            backup = current_adj[:]
            remove_edge(current_adj, e1[0], e1[1])
            
            # Try to add e2
            u, v = e2
            if not ((current_adj[u] >> v) & 1):
                if bin(current_adj[u] & current_adj[v]).count('1') <= 2:
                    valid = True
                    nu = current_adj[u]
                    while nu:
                        low = nu & (-nu)
                        w = low.bit_length() - 1
                        if w != v and bin(current_adj[v] & current_adj[w]).count('1') > 2:
                            valid = False
                            break
                        nu ^= low
                    if valid:
                        nv = current_adj[v]
                        while nv:
                            low = nv & (-nv)
                            w = low.bit_length() - 1
                            if w != u and bin(current_adj[u] & current_adj[w]).count('1') > 2:
                                valid = False
                                break
                            nv ^= low
                    if valid:
                        add_edge(current_adj, u, v)
            
            new_count = count_edges(current_adj)
            if new_count > best_count:
                best_adj = current_adj[:]
                best_count = new_count
                current_adj = best_adj[:]
            else:
                current_adj = backup[:]
                
        return best_adj, best_count

    # Initialize population with random graphs and greedy fill
    pop_size = 50
    population = []
    
    # Strategy: Start with some structured graphs
    # 1. Empty graph + greedy
    adj = [0] * n
    greedy_fill(adj)
    population.append(adj[:])
    
    # 2. Random graphs + greedy
    for _ in range(pop_size - 1):
        adj = [0] * n
        # Random initial edges
        for idx in range(num_pairs):
            if random.random() < 0.5:
                u = pair_u[idx]
                v = pair_v[idx]
                add_edge(adj, u, v)
        # Ensure validity by removing invalid edges? Or just greedy fill on valid subset?
        # Greedy fill assumes input is valid. We need to clean up invalid input.
        # Let's just start empty and add random edges if valid.
        adj = [0] * n
        for idx in range(num_pairs):
            if random.random() < 0.7:
                u = pair_u[idx]
                v = pair_v[idx]
                if not ((adj[u] >> v) & 1):
                    if bin(adj[u] & adj[v]).count('1') <= 2:
                         # Quick check, full check in greedy_fill or here?
                         # Let's just add and let greedy_fill handle it? No, greedy_fill only adds.
                         # We need to ensure the starting point is valid.
                         # Simpler: Start empty, add random valid edges.
                         pass
        
        # Re-do random init properly
        adj = [0] * n
        for idx in range(num_pairs):
            if random.random() < 0.5:
                u = pair_u[idx]
                v = pair_v[idx]
                # Try to add
                if not ((adj[u] >> v) & 1):
                    if bin(adj[u] & adj[v]).count('1') <= 2:
                        # Check neighbors
                        valid = True
                        nu = adj[u]
                        while nu:
                            low = nu & (-nu)
                            w = low.bit_length() - 1
                            if w != v and bin(adj[v] & adj[w]).count('1') > 2:
                                valid = False
                                break
                            nu ^= low
                        if valid:
                            nv = adj[v]
                            while nv:
                                low = nv & (-nv)
                                w = low.bit_length() - 1
                                if w != u and bin(adj[u] & adj[w]).count('1') > 2:
                                    valid = False
                                    break
                                nv ^= low
                        if valid:
                            add_edge(adj, u, v)
        
        greedy_fill(adj)
        population.append(adj[:])
    
    # Evaluate population
    scores = []
    for adj in population:
        if is_valid(adj):
            scores.append(count_edges(adj))
        else:
            scores.append(-1)
            
    best_score = max(scores)
    best_adj = population[scores.index(best_score)]
    
    # GA Loop
    generations = 0
    while time.time() < deadline - 10:
        generations += 1
        
        # Selection: Tournament selection
        def tournament_select(pop, scores, k=3):
            indices = random.sample(range(len(pop)), k)
            best_idx = max(indices, key=lambda i: scores[i])
            return pop[best_idx][:]
        
        new_population = []
        # Keep best individual (elitism)
        new_population.append(best_adj[:])
        
        while len(new_population) < pop_size:
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            # Crossover: Uniform crossover on bitmasks
            child = []
            for i in range(n):
                mask = 0
                for bit in range(n):
                    if random.random() < 0.5:
                        mask |= (parent1[i] >> bit) & 1
                    else:
                        mask |= (parent2[i] >> bit) & 1
                child.append(mask)
                
            # Ensure symmetry
            for i in range(n):
                for j in range(i + 1, n):
                    if (child[i] >> j) & 1:
                        child[j] |= (1 << i)
                    else:
                        child[i] &= ~(1 << j)
                        child[j] &= ~(1 << i)
                        
            # Mutation: Flip random edges
            for _ in range(5):
                idx = random.randint(0, num_pairs - 1)
                u = pair_u[idx]
                v = pair_v[idx]
                if (child[u] >> v) & 1:
                    remove_edge(child, u, v)
                else:
                    # Try to add if valid
                    if bin(child[u] & child[v]).count('1') <= 2:
                        valid = True
                        nu = child[u]
                        while nu:
                            low = nu & (-nu)
                            w = low.bit_length() - 1
                            if w != v and bin(child[v] & child[w]).count('1') > 2:
                                valid = False
                                break
                            nu ^= low
                        if valid:
                            nv = child[v]
                            while nv:
                                low = nv & (-nv)
                                w = low.bit_length() - 1
                                if w != u and bin(child[u] & child[w]).count('1') > 2:
                                    valid = False
                                    break
                                nv ^= low
                        if valid:
                            add_edge(child, u, v)
                            
            # Greedy fill to maximize edges
            greedy_fill(child)
            
            # Check validity (greedy_fill should maintain validity, but just in case)
            if is_valid(child):
                new_population.append(child)
            else:
                # Fallback to parent if invalid
                new_population.append(parent1)
                
        population = new_population
        scores = []
        for adj in population:
            if is_valid(adj):
                scores.append(count_edges(adj))
            else:
                scores.append(-1)
                
        gen_best_score = max(scores)
        gen_best_adj = population[scores.index(gen_best_score)]
        
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_adj = gen_best_adj[:]
            
        # Local search on best individual periodically
        if generations % 5 == 0 and time.time() < deadline - 20:
            ls_adj, ls_score = local_search(best_adj, max_iters=100)
            if ls_score > best_score:
                best_score = ls_score
                best_adj = ls_adj[:]
                # Inject into population
                population[0] = best_adj[:]
                scores[0] = best_score

    # Final local search on best
    if time.time() < deadline - 5:
        best_adj, best_score = local_search(best_adj, max_iters=200)

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
