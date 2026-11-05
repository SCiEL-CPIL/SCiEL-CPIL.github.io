import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks and pairs
    triple_masks = []
    triple_pairs = []
    triple_to_idx = {}
    for i, t in enumerate(all_triples):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        pairs = frozenset([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])
        triple_pairs.append(pairs)
        triple_to_idx[t] = i

    # Precompute all subsets of size k as masks
    all_masks_of_size = {k: [] for k in range(17)}
    for k in range(1, 17):
        for subset in combinations(points, k):
            mask = 0
            for p in subset:
                mask |= (1 << p)
            all_masks_of_size[k].append(mask)
            
    def get_alpha(triple_mask_list, start_k=16):
        # Returns (alpha, mask_of_independent_set)
        for k in range(min(start_k, 16), 0, -1):
            for sm in all_masks_of_size[k]:
                is_ind = True
                for tm in triple_mask_list:
                    if (tm & sm) == tm:
                        is_ind = False
                        break
                if is_ind:
                    return k, sm
        return 0, 0

    best_alpha = 16
    best_triples = []
    
    def can_add(triple_idx, pair_to_triple):
        t_pairs = triple_pairs[triple_idx]
        for p in t_pairs:
            if p in pair_to_triple:
                return False
        return True

    def add_triple(triple_idx, pair_to_triple, active_triples):
        active_triples.append(triple_idx)
        for p in triple_pairs[triple_idx]:
            pair_to_triple[p] = triple_idx

    def remove_triple(triple_idx, pair_to_triple, active_triples):
        # Find position
        pos = -1
        for i, idx in enumerate(active_triples):
            if idx == triple_idx:
                pos = i
                break
        if pos == -1:
            return False
        # Swap with last
        last_idx = active_triples[-1]
        active_triples[pos] = last_idx
        active_triples.pop()
        # Remove pairs
        for p in triple_pairs[triple_idx]:
            if pair_to_triple.get(p) == triple_idx:
                del pair_to_triple[p]
        return True

    def get_active_masks(active_triples):
        return [triple_masks[i] for i in active_triples]

    def local_search(active_triples, pair_to_triple, current_alpha):
        # Mutates active_triples and pair_to_triple
        # Returns final alpha
        no_improve_count = 0
        while no_improve_count < 20 and time.time() < deadline:
            masks = get_active_masks(active_triples)
            alpha, ind_mask = get_alpha(masks, current_alpha)
            
            if alpha < current_alpha:
                current_alpha = alpha
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 10:
                    break

            if alpha <= 6:
                break

            # Find independent set points
            ind_points = []
            for p in points:
                if ind_mask & (1 << p):
                    ind_points.append(p)
            
            # Try to add a triple inside ind_points
            candidates = []
            # To speed up, we can limit the number of candidates checked
            # But combinations of 8 points is 56, which is small.
            for t_comb in combinations(ind_points, 3):
                t_idx = triple_to_idx[t_comb]
                if can_add(t_idx, pair_to_triple):
                    candidates.append((t_idx, []))
                else:
                    conflicts = set()
                    for p in triple_pairs[t_idx]:
                        if p in pair_to_triple:
                            conflicts.add(pair_to_triple[p])
                    if len(conflicts) == 1:
                        conflict_idx = list(conflicts)[0]
                        candidates.append((t_idx, [conflict_idx]))
            
            if candidates:
                t_idx, conflicts = random.choice(candidates)
                for c_idx in conflicts:
                    remove_triple(c_idx, pair_to_triple, active_triples)
                add_triple(t_idx, pair_to_triple, active_triples)
                # Alpha might drop, so we don't increment no_improve_count here
                # It will be checked in next loop iteration
            else:
                # No move found. Try random perturbation
                if active_triples and random.random() < 0.5:
                    rem_idx = random.choice(active_triples)
                    remove_triple(rem_idx, pair_to_triple, active_triples)
                    # Add random valid triple
                    all_indices = list(range(n_triples))
                    random.shuffle(all_indices)
                    for idx in all_indices:
                        if can_add(idx, pair_to_triple):
                            add_triple(idx, pair_to_triple, active_triples)
                            break
                else:
                    break

        return current_alpha

    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction
        active_triples = []
        pair_to_triple = {}
        
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        for idx in indices:
            if can_add(idx, pair_to_triple):
                add_triple(idx, pair_to_triple, active_triples)
                
        # Initial alpha
        masks = get_active_masks(active_triples)
        alpha, _ = get_alpha(masks, 16)
        
        # 2. Local Search
        alpha = local_search(active_triples, pair_to_triple, alpha)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [all_triples[i] for i in active_triples]
            
        if best_alpha <= 6:
            break

    return best_triples

