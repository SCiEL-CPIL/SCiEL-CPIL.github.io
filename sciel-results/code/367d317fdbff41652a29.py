import random
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 16
    points = list(range(n))
    
    best_alpha = n
    best_triples = []
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    triple_to_mask = {t: (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples}
    
    # Fast alpha calculation using NumPy
    def get_alpha(triples):
        if not triples:
            return n
            
        # Convert triples to numpy array of masks
        t_masks = np.array([triple_to_mask[t] for t in triples], dtype=np.uint32)
        
        # We want the max k such that there exists a mask of popcount k with no triple subset.
        # Iterate k from n-1 down to 0.
        # For each k, generate masks? 
        # Generating masks of specific popcount is still combinatorial.
        # However, we can just iterate all 2^16 masks and keep the max popcount of independent ones.
        # 65536 masks.
        # For each mask, check against t_masks.
        # Vectorized check: 
        # For a single mask m, independent if not any((t & m) == t for t in t_masks).
        # (t_masks & m) != t_masks -> all must be True.
        
        # Let's iterate masks from (1<<16)-1 down to 0 is not sorted by popcount.
        # But we can just find the max popcount among independent masks.
        
        # Optimization: If we find an independent mask of size k, we can stop checking sizes < k? 
        # No, we want the MAX size. So we check all and keep max.
        # Or iterate sizes from n down to 0, and for each size, check if ANY mask of that size is independent.
        # If yes, return size.
        
        # Generating masks of size k:
        # itertools.combinations(range(16), k)
        
        for k in range(n, 0, -1):
            # Generate all masks of size k
            found = False
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                # Check independence
                # t_masks & mask != t_masks
                # If all are True, it is independent.
                # np.all((t_masks & mask) != t_masks)
                if np.all((t_masks & mask) != t_masks):
                    return k
        return 0

    def try_random_greedy():
        # Random order of triples
        triples_order = all_triples[:]
        random.shuffle(triples_order)
        
        used_pairs = set()
        triples = []
        
        for t in triples_order:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            if not any(p in used_pairs for p in pairs):
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        return triples

    def local_search(triples):
        # Try to remove triples that don't help reduce alpha, or swap them?
        # Simple hill climbing: 
        # 1. Calculate alpha.
        # 2. Try removing a random triple. If alpha stays same or decreases (impossible, alpha increases or stays same when removing constraints? 
        #    Wait, removing a triple means fewer constraints, so alpha can only increase or stay same.
        #    We want to MINIMIZE alpha. So we want to ADD triples? 
        #    But we are constrained by linearity.
        #    So we should try to REPLACE triples or find a different set.
        
        # Actually, the greedy builds a maximal set (can't add more).
        # Different random orders yield different maximal sets with different alphas.
        # So just running many random greedy trials is the main search.
        # To improve, we can do a "swap" local search.
        # Pick a triple T in the system. Remove it. Try to add a different triple T' that was previously blocked by T or others.
        
        # For now, stick to random restarts with fast alpha.
        pass

    trials = 0
    while time.time() < deadline:
        trials += 1
        trs = try_random_greedy()
        
        alpha = get_alpha(trs)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = trs
            if alpha <= 5:
                break

    return best_triples
# EVOLVE-BLOCK-END
