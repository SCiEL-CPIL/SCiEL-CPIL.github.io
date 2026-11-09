import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    n = 16
    best_alpha = n
    best_triples = []
    
    # Precompute all possible triples and their pair masks?
    # Actually, just generate on the fly or precompute list.
    all_triples = list(itertools.combinations(range(n), 3))
    
    def get_alpha(triples):
        if not triples:
            return n
            
        # Convert triples to bitmasks
        t_masks = set()
        for a, b, c in triples:
            t_masks.add((1 << a) | (1 << b) | (1 << c))
        t_masks = list(t_masks)
        
        max_alpha = 0
        # Iterate masks from (1<<n)-1 down to 0
        # We want the maximum popcount mask that is independent.
        # A mask is independent if no t in t_masks is a subset of mask.
        
        # Optimization: Check masks in descending order of popcount?
        # It's faster to just iterate all masks and keep track of max popcount.
        # 65536 iterations is very fast.
        
        for mask in range((1 << n) - 1, -1, -1):
            # Check if independent
            independent = True
            for t in t_masks:
                if (t & mask) == t:
                    independent = False
                    break
            if independent:
                # Since we iterate from high to low, the first independent mask isn't necessarily the max popcount?
                # No, iterating from (2^n - 1) down to 0 does NOT guarantee descending popcount order.
                # Example: 1110 (popcount 3) comes before 1000 (popcount 1) but 1100 (popcount 2) comes after 1000? No.
                # Binary: 
                # 15: 1111 (4)
                # 14: 1110 (3)
                # 13: 1101 (3)
                # 12: 1100 (2)
                # 11: 1011 (3) -> Here popcount 3 appears after popcount 2.
                # So we cannot just take the first one.
                
                # We must track max popcount.
                pc = mask.bit_count()
                if pc > max_alpha:
                    max_alpha = pc
                    # If max_alpha is n, we can break? No, we want the max. 
                    # If we found n, that's the max possible.
                    if max_alpha == n:
                        break
                # Optimization: If current mask popcount is <= max_alpha, we can skip checking?
                # No, we already checked independence. 
                # But we can skip masks with popcount <= max_alpha entirely?
                # If we do that, we need to generate masks by popcount.
                # Let's stick to simple iteration for now, it's fast enough.
                # 65k * 50 checks = 3.2M ops. ~0.1s.
                
        return max_alpha

    def try_random_greedy(seed_offset=0):
        triples = []
        used_pairs = set()
        
        # Shuffle all triples
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            p1 = (a, b)
            p2 = (a, c)
            p3 = (b, c)
            
            # Check if any pair is used
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
                
            # Add triple
            triples.append(t)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples

    trials = 0
    while time.time() < deadline:
        trials += 1
        trs = try_random_greedy()
        
        # Calculate alpha
        alpha = get_alpha(trs)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = trs
            if alpha <= 7:
                break

    return best_triples
# EVOLVE-BLOCK-END
