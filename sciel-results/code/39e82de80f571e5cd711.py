import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Construct STS(15) using PG(3,2) lines
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    triples_set = set()
    n = 15
    for i in range(n):
        for j in range(i+1, n):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            triples_set.add(t)
            
    best_packing = [list(t) for t in triples_set]
    
    # NumPy-based fast alpha computation
    def compute_alpha_fast(triples):
        n_points = 16
        n_triples = len(triples)
        
        # Create triple masks as a numpy array
        triple_masks = np.zeros(n_triples, dtype=np.int64)
        for idx, t in enumerate(triples):
            m = 0
            for p in t:
                m |= (1 << p)
            triple_masks[idx] = m
        
        def has_independent_set(k):
            if k == 0:
                return True
            combos = list(itertools.combinations(range(n_points), k))
            if not combos:
                return False
            
            # Create masks for all combinations
            combo_masks = np.zeros(len(combos), dtype=np.int64)
            for i, s in enumerate(combos):
                m = 0
                for p in s:
                    m |= (1 << p)
                combo_masks[i] = m
            
            # Vectorized check: for each combo, is any triple fully contained?
            # (combo_mask & triple_mask) == triple_mask for some triple
            contains = (combo_masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            any_contains = np.any(contains, axis=1)
            # If no combo has any triple contained, all are independent
            return not np.all(any_contains)
        
        for k in range(n_points, 0, -1):
            if has_independent_set(k):
                return k
        return 1

    # Verify alpha
    alpha = compute_alpha_fast(best_packing)
    
    # Local search to reduce alpha
    if time.time() < deadline - 10:
        current_triples = best_packing[:]
        current_alpha = alpha
        best_triples = current_triples[:]
        best_alpha = current_alpha
        
        random.seed(42)
        
        # Precompute set of current triples for fast lookup
        def triple_set(triples):
            return set(tuple(sorted(t)) for t in triples)
        
        def is_linear_with(triple, others):
            """Check if triple is linear with all triples in others."""
            for t in others:
                shared = len(set(triple) & set(t))
                if shared > 1:
                    return False
            return True
        
        for iteration in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Try a random move: remove one triple, add another
            if not current_triples:
                break
            
            idx = random.randint(0, len(current_triples) - 1)
            removed_triple = current_triples[idx]
            remaining = current_triples[:idx] + current_triples[idx+1:]
            
            # Generate candidate triples to add
            candidates = []
            
            # Triples containing point 15 (to break isolation)
            for a in range(15):
                for b in range(a+1, 15):
                    candidates.append((a, b, 15))
            
            # Random triples
            for _ in range(30):
                a, b, c = random.sample(range(16), 3)
                candidates.append(tuple(sorted([a, b, c])))
            
            random.shuffle(candidates)
            
            # Try each candidate
            found_move = False
            for cand in candidates[:50]:  # Limit candidates tried per iteration
                if time.time() > deadline - 5:
                    break
                
                cand = tuple(sorted(cand))
                
                # Check linearity with remaining
                if not is_linear_with(cand, remaining):
                    continue
                
                # Check it's not already present
                if cand in triple_set(remaining):
                    continue
                
                # Try this move
                new_triples = remaining + [list(cand)]
                new_alpha = compute_alpha_fast(new_triples)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_triples = new_triples[:]
                    current_triples = new_triples[:]
                    found_move = True
                    break
                elif new_alpha == best_alpha and random.random() < 0.2:
                    # Accept with some probability to explore
                    current_triples = new_triples[:]
                    found_move = True
                    break
            
            # If no move found, try adding a triple without removing (if possible)
            if not found_move and random.random() < 0.1:
                # Try to add a triple that doesn't conflict
                candidates = []
                for a in range(16):
                    for b in range(a+1, 16):
                        for c in range(b+1, 16):
                            candidates.append((a, b, c))
                
                random.shuffle(candidates)
                for cand in candidates[:20]:
                    if time.time() > deadline - 5:
                        break
                    if is_linear_with(cand, current_triples) and cand not in triple_set(current_triples):
                        new_triples = current_triples + [list(cand)]
                        new_alpha = compute_alpha_fast(new_triples)
                        if new_alpha < best_alpha:
                            best_alpha = new_alpha
                            best_triples = new_triples[:]
                            current_triples = new_triples[:]
                        break
        
        if best_alpha < current_alpha:
            current_triples = best_triples
            current_alpha = best_alpha
    
    # Return the best packing found
    return current_triples
