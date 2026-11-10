import itertools
import random
import time

def construct():
    n = 16
    points = range(n)
    
    # Precompute combinations for alpha calculation
    combs_by_k = {}
    for k in range(1, n + 1):
        combs_by_k[k] = list(itertools.combinations(points, k))
        
    # Generate all triples: (mask, pairs, triple_list)
    all_triples = []
    for a, b, c in itertools.combinations(points, 3):
        mask = (1 << a) | (1 << b) | (1 << c)
        pairs = frozenset([(a, b), (a, c), (b, c)])
        all_triples.append((mask, pairs, [a, b, c]))
        
    num_triples = len(all_triples)

    def calc_alpha(active_indices):
        """
        Calculate the independence number.
        active_indices: list of indices into all_triples
        """
        if not active_indices:
            return n
            
        active_masks = [all_triples[i][0] for i in active_indices]
        
        for k in range(n, 0, -1):
            for comb in combs_by_k[k]:
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                is_independent = True
                for t_mask in active_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 0

    def greedy_fill(seed):
        random.seed(seed)
        indices = list(range(num_triples))
        random.shuffle(indices)
        active_indices = []
        used_pairs = set()
        
        for idx in indices:
            _, pairs, _ = all_triples[idx]
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                active_indices.append(idx)
                used_pairs.update(pairs)
                
        return active_indices

    def local_search(initial_indices, time_limit=10):
        active_indices = initial_indices[:]
        used_pairs = set()
        active_set = set(active_indices)
        
        for idx in active_indices:
            _, pairs, _ = all_triples[idx]
            used_pairs.update(pairs)
            
        current_alpha = calc_alpha(active_indices)
        best_alpha = current_alpha
        best_indices = active_indices[:]
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if not active_indices:
                break
            
            idx_out = random.choice(active_indices)
            _, pairs_out, _ = all_triples[idx_out]
            
            # Remove
            active_indices.remove(idx_out)
            active_set.remove(idx_out)
            used_pairs.difference_update(pairs_out)
            
            # Find candidate
            candidate_idx = -1
            for _ in range(100):
                cand_idx = random.randint(0, num_triples - 1)
                if cand_idx in active_set:
                    continue
                _, cand_pairs, _ = all_triples[cand_idx]
                valid = True
                for p in cand_pairs:
                    if p in used_pairs:
                        valid = False
                        break
                if valid:
                    candidate_idx = cand_idx
                    break
                    
            if candidate_idx == -1:
                # Revert
                active_indices.append(idx_out)
                active_set.add(idx_out)
                used_pairs.update(pairs_out)
                continue
                
            # Add candidate
            active_indices.append(candidate_idx)
            active_set.add(candidate_idx)
            _, pairs_in, _ = all_triples[candidate_idx]
            used_pairs.update(pairs_in)
            
            new_alpha = calc_alpha(active_indices)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_indices = active_indices[:]
                    if best_alpha <= 4:
                        break
            else:
                # Revert
                active_indices.remove(candidate_idx)
                active_set.remove(candidate_idx)
                used_pairs.difference_update(pairs_in)
                active_indices.append(idx_out)
                active_set.add(idx_out)
                used_pairs.update(pairs_out)
                
        return best_indices

    best_global_alpha = n
    best_global_indices = []
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 100:
        seed += 1
        initial_indices = greedy_fill(seed)
        alpha = calc_alpha(initial_indices)
        
        if alpha < best_global_alpha:
            best_global_alpha = alpha
            best_global_indices = initial_indices[:]
            
        if alpha <= 5:
            refined_indices = local_search(initial_indices, time_limit=10)
            refined_alpha = calc_alpha(refined_indices)
            if refined_alpha < best_global_alpha:
                best_global_alpha = refined_alpha
                best_global_indices = refined_indices[:]
                
        if best_global_alpha <= 4:
            break

    result = [all_triples[i][2] for i in best_global_indices]
    
    return result
