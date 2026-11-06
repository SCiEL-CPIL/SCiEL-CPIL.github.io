import random
import itertools

def construct():
    # Points 0..15
    n = 16
    points = list(range(n))
    
    # Initial empty hypergraph
    E = []
    
    # Helper to check linearity
    def is_linear(edge, existing_edges):
        for e in existing_edges:
            inter = set(edge) & set(e)
            if len(inter) > 1:
                return False
        return True

    # Helper to find independence number
    # This is expensive. We need a fast approximation or exact for small n.
    # Max independent set in 3-uniform hypergraph on 16 nodes.
    # Brute force over subsets is 2^16 = 65536.
    # For each subset, check if it contains a triple.
    # This is feasible.
    
    def calc_alpha(edges):
        # edges is a list of tuples
        # To speed up, we can precompute a bitmask for each triple
        # Or just check subsets.
        
        # Optimization: 
        # A set S is independent if no triple in edges is a subset of S.
        
        # Let's use a bitset representation for vertices.
        # 16 bits.
        
        # Precompute triple masks
        triple_masks = [0] * len(edges)
        for i, e in enumerate(edges):
            mask = 0
            for v in e:
                mask |= (1 << v)
            triple_masks[i] = mask
            
        # Iterate over all subsets? 2^16 = 65536.
        # For each subset, check if any triple mask is a subset.
        # (mask & subset_mask) == mask
        
        max_alpha = 0
        # We can iterate subsets in decreasing size order to find max quickly?
        # Or just check all.
        
        # To speed up, we can stop checking if size <= max_alpha.
        # But we need to check all to be sure.
        
        # Actually, we can check all subsets of size k, starting from 16 down to 1.
        # If we find an independent set of size k, return k.
        
        for k in range(n, 0, -1):
            for subset in itertools.combinations(points, k):
                subset_mask = 0
                for v in subset:
                    subset_mask |= (1 << v)
                
                is_independent = True
                for tm in triple_masks:
                    if (subset_mask & tm) == tm:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
                    
        return 0

    # Randomized construction
    # Start with a random set of edges
    # Try to add edges to minimize alpha
    
    best_E = []
    best_alpha = n
    
    # We can try to build a "dense" linear hypergraph
    # Max edges in linear 3-uniform on 16 is around 40?
    
    # Let's try to generate a random linear hypergraph with many edges
    all_triples = list(itertools.combinations(points, 3))
    random.shuffle(all_triples)
    
    for edge in all_triples:
        if is_linear(edge, E):
            E.append(edge)
            
    alpha = calc_alpha(E)
    best_E = E[:]
    best_alpha = alpha
    
    # Local search
    # Try to swap edges or add/remove
    # If alpha > 5, we need to hit independent sets.
    
    # Given time constraints, if we find alpha=5, return.
    # If not, return the best found.
    
    # To improve, we can try to specifically target independent sets.
    # But calc_alpha is O(2^n * |E|).
    
    # Let's just return the best one found in a few random restarts.
    
    for _ in range(20):
        E = []
        all_triples = list(itertools.combinations(points, 3))
        random.shuffle(all_triples)
        for edge in all_triples:
            if is_linear(edge, E):
                E.append(edge)
        alpha = calc_alpha(E)
        if alpha < best_alpha:
            best_alpha = alpha
            best_E = E[:]
            if best_alpha == 5:
                break
                
    # Convert to list of lists
    return [list(e) for e in best_E]
