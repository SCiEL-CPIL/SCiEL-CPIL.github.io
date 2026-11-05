import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute pairs for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = frozenset([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
        triple_pairs.append(pairs)
        
    # Precompute masks for each triple
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        
    # Precompute masks for all subsets of points (0 to 2^16 - 1)
    # This allows O(1) mask lookup for any subset represented as an integer
    subset_masks = np.zeros(1 << 16, dtype=np.uint16)
    for i in range(1 << 16):
        m = 0
        for j in range(16):
            if i & (1 << j):
                m |= (1 << j)
        subset_masks[i] = m
        
    # Map from pair to list of triple indices containing that pair
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)

    def get_alpha_and_set(triple_indices):
        """
        Returns (alpha, independent_set) where independent_set is a tuple of points.
        Uses bitmask optimization.
        """
        if not triple_indices:
            return 16, tuple(points)
        
        # Create a boolean array of active triples
        active = np.zeros(n_triples, dtype=bool)
        active[triple_indices] = True
        
        # Get masks of active triples
        active_masks = triple_masks  # We will use the full list and filter by active flag if needed, 
                                     # but numpy slicing is faster if we have a precomputed array of masks
                                     # Let's just use the precomputed triple_masks array and index by triple_indices
        
        masks = np.array([triple_masks[i] for i in triple_indices], dtype=np.uint16)
        
        # Iterate k from 16 down to 1
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                # Compute mask for this subset
                s_mask = 0
                for p in subset:
                    s_mask |= (1 << p)
                
                # Check if any active triple is fully contained in s_mask
                # A triple mask m is contained in s_mask if (m & s_mask) == m
                # We check all active triple masks
                if len(masks) == 0:
                    return k, subset
                    
                contained = (masks & s_mask) == masks
                if not contained.any():
                    return k, subset
        return 0, ()

    best_alpha = 16
    best_triples = []
    
    # Greedy construction helper
    def greedy_construct():
        indices = list(range(n_triples))
        random.shuffle(indices)
        current_indices = []
        used_pairs = set()
        
        for idx in indices:
            t_pairs = triple_pairs[idx]
            if all(p not in used_pairs for p in t_pairs):
                current_indices.append(idx)
                used_pairs.update(t_pairs)
        
        return current_indices, used_pairs

    # Local search step
    def local_search(current_indices, used_pairs, max_steps=500):
        alpha, indep_set = get_alpha_and_set(current_indices)
        
        if alpha <= best_alpha:
            return alpha, current_indices, used_pairs
            
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            alpha, indep_set = get_alpha_and_set(current_indices)
            
            if alpha <= best_alpha:
                return alpha, current_indices, used_pairs
                
            if not indep_set:
                break

            # Find all triples contained in indep_set
            indep_set_set = set(indep_set)
            candidates = []
            for t in combinations(indep_set, 3):
                # Find index of t
                # This is slow if done naively. 
                # Since n_triples is 560, we can precompute a map from tuple to index
                pass
            
            # Optimization: Precompute tuple to index map
            # We'll do this outside the loop if possible, but for now, 
            # iterating over all 560 triples and checking if subset is faster 
            # if the independent set is small.
            
            # Actually, let's just iterate over all_triples and check if it's a subset of indep_set
            # This is 560 checks.
            for idx, t in enumerate(all_triples):
                if all(p in indep_set_set for p in t):
                    # Check if this triple is already in current_indices
                    # It shouldn't be, because indep_set contains no triples from current_indices
                    # But let's verify it's not in used_pairs
                    t_pairs = triple_pairs[idx]
                    if all(p not in used_pairs for p in t_pairs):
                        candidates.append(idx)
            
            if candidates:
                # Add a random candidate
                idx = random.choice(candidates)
                current_indices.append(idx)
                used_pairs.update(triple_pairs[idx])
                continue # Recalculate alpha
                
            # If no direct addition possible, try a swap
            # Pick a random triple in indep_set that conflicts
            # We need to find a triple T in indep_set that conflicts with some T_conf in current_indices
            # And T_conf is "removable"
            
            # Strategy: Pick a random point p in indep_set.
            # Find triples containing p that are in current_indices? No, indep_set has no triples.
            # Find triples T in all_triples such that T subset of indep_set.
            # For each such T, if it conflicts with T_conf, try removing T_conf and adding T.
            
            # Let's collect all potential swaps
            swaps = []
            for idx, t in enumerate(all_triples):
                if all(p in indep_set_set for p in t):
                    t_pairs = triple_pairs[idx]
                    conflicts = []
                    for p in t_pairs:
                        if p in used_pairs:
                            # Find the triple in current_indices that has this pair
                            # We need a map from pair to current triple index
                            # This map is not maintained dynamically in current_indices easily.
                            # We can just scan current_indices.
                            for ci in current_indices:
                                if p in triple_pairs[ci]:
                                    conflicts.append(ci)
                                    break
                    if conflicts:
                        # We have a swap candidate: remove any of conflicts, add idx
                        # We prefer to remove a triple that is "least useful" or random
                        swaps.append((idx, conflicts))
                        
            if swaps:
                # Pick a random swap
                add_idx, conflict_list = random.choice(swaps)
                rem_idx = random.choice(conflict_list)
                
                # Remove rem_idx
                current_indices.remove(rem_idx)
                for p in triple_pairs[rem_idx]:
                    # Check if p is used by any other triple
                    # If yes, don't remove from used_pairs
                    # If no, remove
                    # This check is O(N). N is small (<560).
                    still_used = False
                    for ci in current_indices:
                        if p in triple_pairs[ci]:
                            still_used = True
                            break
                    if not still_used:
                        used_pairs.discard(p)
                        
                # Add add_idx
                current_indices.append(add_idx)
                used_pairs.update(triple_pairs[add_idx])
                
            else:
                # No moves available in this independent set
                # Perturb: remove a random triple
                if current_indices:
                    rem_pos = random.randrange(len(current_indices))
                    rem_idx = current_indices.pop(rem_pos)
                    for p in triple_pairs[rem_idx]:
                        still_used = False
                        for ci in current_indices:
                            if p in triple_pairs[ci]:
                                still_used = True
                                break
                        if not still_used:
                            used_pairs.discard(p)
                else:
                    break

        return alpha, current_indices, used_pairs

    # Main loop
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_indices, used_pairs = greedy_construct()
        
        # Initial evaluation
        alpha, _ = get_alpha_and_set(current_indices)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [all_triples[i] for i in current_indices]
            
        if best_alpha <= 4: # Theoretical lower bound is around 4-5 for 16 points? 
                           # Actually, a Steiner Triple System on 15 points has alpha 5? 
                           # If we can get 4, that's great.
            pass

        # Local search
        alpha, current_indices, used_pairs = local_search(current_indices, used_pairs, max_steps=200)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [all_triples[i] for i in current_indices]
            
    return best_triples
