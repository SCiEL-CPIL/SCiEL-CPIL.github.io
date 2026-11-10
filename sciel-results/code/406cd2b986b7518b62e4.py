import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    n = 16
    all_vertices = list(range(n))
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute Q3 copies in K_16
    # A Q3 copy is defined by 8 vertices and a specific labeling isomorphism.
    # To check if a graph G contains a Q3, we can check if there exists a set of 8 vertices
    # and a bijection to {0,1,2,3,4,5,6,7} such that all Q3 edges exist in G.
    # For the complement strategy:
    # We want to minimize edges in complement \bar{G}.
    # \bar{G} must intersect every Q3 copy in K_16.
    # So for every Q3 copy (set of 12 edges), \bar{G} must contain at least one edge from that set.
    
    # Generating all distinct Q3 subgraphs in K_16 is expensive.
    # Number of ways to choose 8 vertices from 16 is C(16,8) = 12870.
    # For each set of 8 vertices, how many distinct Q3 subgraphs are there?
    # The automorphism group of Q3 has size 48.
    # Total bijections from 8 vertices to Q3 vertices is 8! = 40320.
    # Distinct subgraphs = 40320 / 48 = 840.
    # Total Q3 copies = 12870 * 840 = 10,810,800.
    # This is too many to precompute and store explicitly as edge sets if we want speed.
    
    # Alternative: Check validity on the fly during GA evaluation.
    # But GA evaluation needs to be fast.
    # Let's use a hybrid:
    # 1. Generate a random subset of Q3 copies to use as "constraints" for the GA fitness.
    # 2. Use the full check only for the final verification and local search.
    
    # Actually, checking if a specific graph is Q3-free is the bottleneck.
    # Let's optimize the `is_q3_free` check using bitmasks.
    
    def is_q3_free(adj):
        """
        Check if graph defined by adj (list of bitmasks) is Q3-free.
        Uses the same logic as the current program's creates_q3_fast but for the whole graph.
        We iterate over all pairs (u, v) that are edges, and check if adding them creates a Q3?
        No, we need to check if the graph *contains* a Q3.
        
        Efficient check:
        For every pair of non-adjacent vertices? No.
        For every edge (u, v) in G:
          Check if there is a Q3 containing (u, v).
          This is similar to creates_q3_fast(u, v, adj) but we assume (u, v) is already in adj.
          Wait, creates_q3_fast checks if adding (u, v) creates a NEW Q3.
          If (u, v) is already in adj, we can still use the logic:
          Does there exist a Q3 containing (u, v)?
          The logic in creates_q3_fast effectively checks if there is a Q3 containing (u, v)
          given the current adjacencies.
          
          So, if ANY edge (u, v) in G triggers creates_q3_fast(u, v, adj) == True,
          then G contains a Q3.
          
          Is this correct?
          If G contains a Q3, it contains 12 edges. Pick any edge (u, v) from that Q3.
          When we run creates_q3_fast(u, v, adj), it looks for the other 6 vertices and edges
          required to form a Q3 with (u, v). Since the Q3 is fully present in G, the check
          will find the required vertices and edges and return True.
          
          So: G is Q3-free iff for all (u, v) in E(G), creates_q3_fast(u, v, adj) is False.
        """
        for u in range(n):
            mask = adj[u]
            # Only check u < v to avoid duplicates
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    if creates_q3_fast(u, v, adj):
                        return False
                mask -= bit
        return True

    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u, v) to graph adj creates a Q3.
        Assumes (u, v) might or might not be in adj, but checks the structure.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def adj_from_complement(comp_mask):
        """
        comp_mask is an integer bitmask of length num_edges.
        Bit i is 1 if edge i is in the complement \bar{G}.
        Returns adj list for G.
        """
        adj = [0] * n
        for i, (u, v) in enumerate(all_edges):
            if not (comp_mask >> i) & 1:
                # Edge is in G
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def count_edges_from_complement(comp_mask):
        """
        Count edges in G.
        """
        # Total edges in K_16 is 120.
        # Edges in G = 120 - (number of bits set in comp_mask)
        return num_edges - bin(comp_mask).count('1')

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    # --- Genetic Algorithm ---
    
    pop_size = 200
    elite_size = 10
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    def evaluate(comp_mask):
        """
        Fitness: Maximize edges in G (minimize edges in \bar{G}) subject to G being Q3-free.
        If G is not Q3-free, penalty.
        """
        adj = adj_from_complement(comp_mask)
        if not is_q3_free(adj):
            # Penalty: large negative value
            # We want to minimize complement size, so maximize G size.
            # If invalid, return -infinity or a very low score.
            return -1000000
        return count_edges_from_complement(comp_mask)

    # Initialize population
    # Start with some random complements and some structured ones.
    population = []
    
    # Structured init: Random sparse complement
    for _ in range(pop_size):
        # Random complement with ~10-20 edges
        num_comp_edges = random.randint(10, 30)
        comp_edges = random.sample(range(num_edges), num_comp_edges)
        mask = 0
        for i in comp_edges:
            mask |= (1 << i)
        population.append(mask)
    
    # Evaluate initial population
    scores = [evaluate(m) for m in population]
    
    # Sort by score (descending)
    indexed_pop = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    best_score, best_mask = indexed_pop[0]
    
    # If best is invalid, we need to fix it.
    # Let's add a repair step: if invalid, remove random edges from complement (add to G) until valid?
    # No, adding to G makes it MORE likely to have Q3.
    # Removing from complement means adding to G.
    # We need to REMOVE edges from G (ADD to complement) to break Q3s.
    # So if invalid, we should ADD edges to complement.
    
    def repair(mask):
        """
        Add edges to complement (remove from G) until G is Q3-free.
        """
        adj = adj_from_complement(mask)
        if is_q3_free(adj):
            return mask, True
        
        # Find a Q3 and remove one of its edges from G (add to complement)
        # We can find a Q3 by checking edges.
        # For speed, just add random edges to complement until valid.
        # This is a crude repair.
        
        # Better: Identify a specific Q3 and block it.
        # Let's just add random missing edges to complement.
        # Max 120 edges in complement.
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Pick a random edge not in complement
            missing_edges = [i for i in range(num_edges) if not (mask >> i) & 1]
            if not missing_edges:
                break
            i = random.choice(missing_edges)
            mask |= (1 << i)
            adj = adj_from_complement(mask)
            if is_q3_free(adj):
                return mask, True
        return mask, False

    # Repair invalid individuals in population
    new_population = []
    new_scores = []
    for mask, score in zip(population, scores):
        if score < 0:
            new_mask, valid = repair(mask)
            new_score = evaluate(new_mask)
            new_population.append(new_mask)
            new_scores.append(new_score)
        else:
            new_population.append(mask)
            new_scores.append(score)
            
    population = new_population
    scores = new_scores
    
    # Update best
    indexed_pop = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
    best_score, best_mask = indexed_pop[0]
    
    # GA Loop
    generation = 0
    while time.time() < deadline - 10:
        generation += 1
        
        # Sort population
        indexed_pop = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        
        # Elitism
        new_population = [m for _, m in indexed_pop[:elite_size]]
        new_scores = [s for s, _ in indexed_pop[:elite_size]]
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Selection: Tournament
            def select():
                idx1 = random.randint(0, len(scores) - 1)
                idx2 = random.randint(0, len(scores) - 1)
                if scores[idx1] > scores[idx2]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            parent1 = select()
            parent2 = select()
            
            if random.random() < crossover_rate:
                # Uniform Crossover
                child = 0
                for i in range(num_edges):
                    if random.random() < 0.5:
                        if (parent1 >> i) & 1:
                            child |= (1 << i)
                    else:
                        if (parent2 >> i) & 1:
                            child |= (1 << i)
            else:
                child = parent1
            
            # Mutation
            # Flip bits in complement mask
            # Probability of flipping each bit
            for i in range(num_edges):
                if random.random() < mutation_rate:
                    child ^= (1 << i)
            
            # Repair if invalid
            score = evaluate(child)
            if score < 0:
                child, valid = repair(child)
                score = evaluate(child)
            
            new_population.append(child)
            new_scores.append(score)
        
        population = new_population
        scores = new_scores
        
        # Update best
        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            best_idx = scores.index(max_score)
            best_mask = population[best_idx]
            
    # Final Polish: Local Search on Best
    # Try to remove edges from complement (add to G) if possible.
    # This is the same as greedy_add on G.
    
    best_adj = adj_from_complement(best_mask)
    
    # Try to add edges to G (remove from complement)
    # Iterate over all missing edges in G (present in complement)
    # Try to remove them from complement (add to G) if it doesn't create a Q3.
    
    # Get list of edges in complement
    comp_edges = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            comp_edges.append(i)
    
    random.shuffle(comp_edges)
    
    for i in comp_edges:
        if time.time() > deadline - 2:
            break
        u, v = all_edges[i]
        # Try to add (u, v) to G
        # Check if adding (u, v) to best_adj creates a Q3
        if not creates_q3_fast(u, v, best_adj):
            # Add to G
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            # Remove from complement
            best_mask &= ~(1 << i)
    
    # Final Verification
    if not is_q3_free(best_adj):
        # Fallback: Return a known valid graph or empty graph
        # Let's return a simple bipartite graph K_{8,8} which is Q3-free?
        # Wait, K_{8,8} contains Q3.
        # Q3 is bipartite. K_{8,8} contains many bipartite subgraphs.
        # Does K_{8,8} contain Q3?
        # Q3 has 8 vertices, 4 in each part.
        # K_{8,8} has all edges between parts.
        # So yes, K_{8,8} contains Q3.
        
        # We need a Q3-free graph.
        # Random graph with p=0.1 is likely Q3-free and has ~6 edges.
        # Or just return the best found if it was valid before polish?
        # The polish only adds edges, so if it was valid before, it might become invalid.
        # Wait, the polish checks creates_q3_fast before adding.
        # So it should remain valid.
        
        # If it's invalid, something went wrong.
        # Let's return a safe graph: Empty graph.
        # Or a matching.
        # Let's return a random sparse graph.
        adj = [0] * n
        for _ in range(5):
            u, v = random.sample(range(n), 2)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        best_adj = adj

    return edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
