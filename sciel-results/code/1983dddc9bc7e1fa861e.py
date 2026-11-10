import itertools
import random
import time

def construct():
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute masks and pairs for each triple
    triple_masks = [0] * len(all_triples)
    triple_pairs = []
    for i, t in enumerate(all_triples):
        a, b, c = t
        triple_masks[i] = (1 << a) | (1 << b) | (1 << c)
        triple_pairs.append([(a, b), (a, c), (b, c)])

    def get_alpha(active_indices):
        """
        Calculate the independence number of the hypergraph defined by active_indices.
        """
        if not active_indices:
            return n
            
        active_masks = [triple_masks[i] for i in active_indices]
        
        # Check k from n down to 1
        for k in range(n, 0, -1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                is_ind = True
                for am in active_masks:
                    if (mask & am) == am:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def greedy_construct(seed):
        random.seed(seed)
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        active = []
        used_pairs = set()
        
        for idx in indices:
            conflict = False
            for p in triple_pairs[idx]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                if key in used_pairs:
                    conflict = True
                    break
            if not conflict:
                active.append(idx)
                for p in triple_pairs[idx]:
                    key = (min(p[0], p[1]), max(p[0], p[1]))
                    used_pairs.add(key)
        return active, used_pairs

    def local_search(active, used_pairs, max_time=5):
        current_alpha = get_alpha(active)
        best_alpha = current_alpha
        best_active = active[:]
        best_used_pairs = used_pairs.copy()
        
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            if not active:
                break
                
            # Pick a random triple to remove
            idx_to_remove = random.choice(active)
            
            # Remove it
            active.remove(idx_to_remove)
            removed_pairs = []
            for p in triple_pairs[idx_to_remove]:
                key = (min(p[0], p[1]), max(p[0], p[1]))
                used_pairs.discard(key)
                removed_pairs.append(key)
            
            # Try to add a new triple
            # We can try all available triples or a random subset
            # To be fast, try a random subset of candidates
            candidates = list(range(len(all_triples)))
            random.shuffle(candidates)
            
            for candidate_idx in candidates:
                if candidate_idx == idx_to_remove:
                    continue
                if candidate_idx in active:
                    continue
                    
                # Check linearity
                valid = True
                for p in triple_pairs[candidate_idx]:
                    key = (min(p[0], p[1]), max(p[0], p[1]))
                    if key in used_pairs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Add it and check alpha
                active.append(candidate_idx)
                added_pairs = []
                for p in triple_pairs[candidate_idx]:
                    key = (min(p[0], p[1]), max(p[0], p[1]))
                    used_pairs.add(key)
                    added_pairs.append(key)
                
                new_alpha = get_alpha(active)
                
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_active = active[:]
                    best_used_pairs = used_pairs.copy()
                    if best_alpha <= 4: # Good enough
                        break
                else:
                    # Revert
                    active.remove(candidate_idx)
                    for key in added_pairs:
                        used_pairs.discard(key)
                
                # Only try one addition per removal step for simplicity/speed
                break
            
            if best_alpha <= 4:
                break

        return best_active, best_used_pairs, best_alpha

    best_alpha = n
    best_active = []
    
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 115: # Leave 5s buffer
        seed += 1
        active, used_pairs = greedy_construct(seed)
        
        # Run local search
        active, used_pairs, alpha = local_search(active, used_pairs, max_time=5)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_active = active[:]
            
        if best_alpha <= 4:
            break

    result = [list(all_triples[i]) for i in best_active]
    return result
