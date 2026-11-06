import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n_points = 16
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n_points), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute conflicts
    # conflicts[i] is a list of indices j such that all_triples[i] and all_triples[j] share > 1 point
    conflicts = [[] for _ in range(len(all_triples))]
    for i in range(len(all_triples)):
        t1 = set(all_triples[i])
        for j in range(i+1, len(all_triples)):
            t2 = set(all_triples[j])
            if len(t1 & t2) > 1:
                conflicts[i].append(j)
                conflicts[j].append(i)
    
    # Precompute subset masks for independence check
    # We check k=6, 7, 8 ... 
    # If we find an independent set of size k, alpha >= k.
    # We want to minimize alpha.
    # We assume alpha is small (5, 6, 7).
    # We check k=6 first. If no independent set of size 6, alpha <= 5.
    # Then check k=5.
    
    # For speed, precompute masks for k=6 and k=7
    masks_6 = np.array([sum(1 << p for p in c) for c in itertools.combinations(range(n_points), 6)], dtype=np.int64)
    masks_7 = np.array([sum(1 << p for p in c) for c in itertools.combinations(range(n_points), 7)], dtype=np.int64)
    
    def compute_alpha(triples):
        """
        Returns the independence number of the given list of triples.
        """
        if not triples:
            return n_points
            
        triple_masks = np.array([sum(1 << p for p in t) for t in triples], dtype=np.int64)
        
        # Check if there is an independent set of size 7
        # If yes, alpha >= 7
        if len(masks_7) > 0:
            contains_7 = (masks_7[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            any_contains_7 = np.any(contains_7, axis=1)
            if np.any(~any_contains_7):
                # There is an independent set of size 7.
                # Check 8? Unlikely to be >7 in this context, but let's be safe.
                # For k > 7, number of subsets is smaller.
                # But if we have a packing, alpha is usually small.
                # If alpha >= 7, we return 7 (or higher if we check).
                # Let's just check 8 and 9 quickly if 7 exists.
                # Actually, if we have a valid packing, alpha is bounded.
                # If we find one of size 7, we can return 7 immediately if we assume we won't go higher?
                # No, we want the LARGEST.
                # But for minimization, if we find 7, we know it's "bad".
                # We can just return 7 if we are confident it's the max.
                # Given the density, 7 is likely the max.
                # Let's check 8 to be sure.
                masks_8 = np.array([sum(1 << p for p in c) for c in itertools.combinations(range(n_points), 8)], dtype=np.int64)
                contains_8 = (masks_8[:, None] & triple_masks[None, :]) == triple_masks[None, :]
                any_contains_8 = np.any(contains_8, axis=1)
                if np.any(~any_contains_8):
                    return 8
                return 7
        
        # If no independent set of size 7, check 6
        if len(masks_6) > 0:
            contains_6 = (masks_6[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            any_contains_6 = np.any(contains_6, axis=1)
            if np.any(~any_contains_6):
                return 6
        
        # If no independent set of size 6, check 5
        # Binomial(16,5) = 4368
        masks_5 = np.array([sum(1 << p for p in c) for c in itertools.combinations(range(n_points), 5)], dtype=np.int64)
        contains_5 = (masks_5[:, None] & triple_masks[None, :]) == triple_masks[None, :]
        any_contains_5 = np.any(contains_5, axis=1)
        if np.any(~any_contains_5):
            return 5
            
        # If no independent set of size 5, check 4
        # It is highly unlikely to be < 5 for a packing of this size.
        # But let's return 4 or 5.
        # If we reach here, alpha <= 4.
        # We can just return 4 as a lower bound estimate or check 4.
        # Checking 4 is fast.
        masks_4 = np.array([sum(1 << p for p in c) for c in itertools.combinations(range(n_points), 4)], dtype=np.int64)
        contains_4 = (masks_4[:, None] & triple_masks[None, :]) == triple_masks[None, :]
        any_contains_4 = np.any(contains_4, axis=1)
        if np.any(~any_contains_4):
            return 4
            
        return 3

    # Initial construction: STS(15) on 0..14
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    initial_triples = set()
    n_sts = 15
    for i in range(n_sts):
        for j in range(i+1, n_sts):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            initial_triples.add(t)
            
    # Perform 2 swaps to involve point 15
    # Swap pair (0,1) and (2,3)
    # Find triples containing (0,1) and (2,3)
    t_01 = None
    t_23 = None
    for t in initial_triples:
        if 0 in t and 1 in t:
            t_01 = t
        if 2 in t and 3 in t:
            t_23 = t
            
    current_triples = list(initial_triples)
    if t_01:
        current_triples.remove(t_01)
        current_triples.append(tuple(sorted([15, 0, 1])))
    if t_23:
        current_triples.remove(t_23)
        current_triples.append(tuple(sorted([15, 2, 3])))
        
    current_alpha = compute_alpha(current_triples)
    best_triples = current_triples[:]
    best_alpha = current_alpha
    
    random.seed(42)
    
    # Local Search
    current_set = set(current_triples)
    
    def is_linear(idx_new, current_indices):
        """Check if triple idx_new is linear with all triples in current_indices."""
        for idx_old in current_indices:
            if idx_new in conflicts[idx_old]:
                return False
        return True

    # Main loop
    while time.time() < deadline - 5:
        # Try a swap
        if current_set:
            idx_remove = random.choice(list(current_set))
            
            # Pick a candidate to add
            # Bias towards triples containing 15 or random
            if random.random() < 0.3:
                a, b = random.sample(range(15), 2)
                t_add = tuple(sorted([a, b, 15]))
            else:
                t_add = random.choice(all_triples)
            
            idx_add = triple_to_idx[t_add]
            
            if idx_add in current_set:
                continue
                
            # Check linearity with current_set - {idx_remove}
            valid = True
            for idx_old in current_set:
                if idx_old == idx_remove:
                    continue
                if idx_add in conflicts[idx_old]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Apply move
            new_set = set(current_set)
            new_set.remove(idx_remove)
            new_set.add(idx_add)
            
            new_triples = [all_triples[i] for i in new_set]
            new_alpha = compute_alpha(new_triples)
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = new_triples
                current_set = new_set
                current_alpha = new_alpha
            elif new_alpha <= current_alpha and random.random() < 0.5:
                current_set = new_set
                current_alpha = new_alpha
        else:
            break

    return [list(t) for t in best_triples]
