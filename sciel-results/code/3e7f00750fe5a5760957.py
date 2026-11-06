import random
import itertools

def construct():
    n = 16
    points = list(range(n))
    full_mask = (1 << n) - 1
    
    # Precompute popcount for all masks? Or use bin(x).count('1')
    # bin(x).count('1') is fast enough for 65536 items.
    
    def fast_calc_alpha(edges):
        if not edges:
            return n
            
        # Convert edges to masks
        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
            
        # independent[i] indicates if subset i is independent
        # We want to find the max popcount(i) such that independent[i] is True.
        # independent[i] is False if any edge mask is a subset of i.
        
        # Initialize all as True
        is_indep = [True] * (1 << n)
        
        # For each edge, mark all its supersets as NOT independent
        for em in edge_masks:
            # Supersets of em are formed by em | sub, where sub is a subset of (~em) & full_mask
            complement = (~em) & full_mask
            # Iterate all subsets of complement
            # Standard subset iteration:
            sub = complement
            while True:
                s = em | sub
                is_indep[s] = False
                if sub == 0:
                    break
                sub = (sub - 1) & complement
                
        # Find max popcount
        max_alpha = 0
        # We can iterate from full_mask down to 0?
        # Or just iterate all and track max.
        # To be faster, we can check sizes descending.
        # But 65536 checks is small.
        for i in range(1 << n):
            if is_indep[i]:
                pc = bin(i).count('1')
                if pc > max_alpha:
                    max_alpha = pc
                    if max_alpha == n: # Early exit if full set is independent? 
                        # Actually if max_alpha is n, we can stop? No, we want max.
                        # If we found n, it's the max possible.
                        break
                        
        return max_alpha

    def is_linear(edge, existing_edges):
        edge_set = set(edge)
        for e in existing_edges:
            if len(edge_set & set(e)) > 1:
                return False
        return True

    best_E = []
    best_alpha = n
    
    # Greedy construction with many restarts
    all_triples = list(itertools.combinations(points, 3))
    
    # We can also try to bias the random shuffle?
    
    for restart in range(500): # 500 restarts is feasible with fast alpha
        E = []
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for edge in shuffled:
            if is_linear(edge, E):
                E.append(edge)
                
        alpha = fast_calc_alpha(E)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_E = E[:]
            if best_alpha <= 4: # Theoretical min for 16 points linear 3-uniform?
                # f(16) is likely 4 or 5. If we hit 4, it's very good.
                # Let's try to go lower if possible, but 4 is a strong candidate.
                # We can break if we are confident.
                pass # Don't break immediately, maybe we can get 3? Unlikely.
                
    # Local Search to try to improve best_E
    # If best_alpha is 5, try to add triples to break the independent sets of size 5.
    # This requires knowing the independent sets.
    # We can just do more random restarts.
    
    # To be safe, run a few local search steps on the best solution
    # Move: Add a random linear triple. If alpha decreases, keep. Else revert.
    # But adding a triple might increase alpha? No, adding constraints (edges) can only decrease or keep alpha.
    # Wait, adding an edge *can* make a previously independent set non-independent, so alpha decreases or stays same.
    # So we should try to ADD as many linear triples as possible to the best_E found.
    # The greedy algorithm already does this (maximal linear hypergraph).
    # So the greedy result is already "saturated" in the sense that no more triples can be added.
    # Thus, local search by addition is useless.
    # We need SWAPS.
    # Remove a triple, add two others?
    # This is complex.
    
    # Given the fast alpha, 1000 restarts is better.
    for restart in range(500, 1000):
        E = []
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            if is_linear(edge, E):
                E.append(edge)
        alpha = fast_calc_alpha(E)
        if alpha < best_alpha:
            best_alpha = alpha
            best_E = E[:]

    return [list(e) for e in best_E]
