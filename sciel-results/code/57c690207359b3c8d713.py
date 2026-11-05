import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append([i, j, k])
                
    triple_masks = []
    for t in all_triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
            
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    def is_valid_idx(idx, current_indices):
        t_mask = triple_masks[idx]
        for c_idx in current_indices:
            c_mask = triple_masks[c_idx]
            if (t_mask & c_mask).bit_count() > 1:
                return False
        return True
        
    def calc_alpha(indices, min_size=0):
        if not indices:
            return n
        masks = [triple_masks[i] for i in indices]
        max_alpha = min_size
        for m in range(1 << n):
            if popcount[m] <= max_alpha:
                continue
            indep = True
            for tm in masks:
                if (m & tm) == tm:
                    indep = False
                    break
            if indep:
                max_alpha = popcount[m]
        return max_alpha

    def greedy():
        indices = []
        idxs = list(range(len(all_triples)))
        random.shuffle(idxs)
        for idx in idxs:
            if is_valid_idx(idx, indices):
                indices.append(idx)
        return indices

    best_alpha = n
    best_indices = []
    
    for _ in range(50):
        if time.time() > deadline: break
        indices = greedy()
        alpha = calc_alpha(indices)
        if alpha < best_alpha:
            best_alpha = alpha
            best_indices = indices[:]
            
    current_indices = best_indices[:]
    current_alpha = best_alpha
    
    steps = 0
    no_improve = 0
    
    while time.time() < deadline:
        steps += 1
        if current_indices:
            rem_pos = random.randint(0, len(current_indices) - 1)
            new_indices = current_indices[:]
            new_indices.pop(rem_pos)
            
            # Try to add
            added = False
            for _ in range(10):
                add_idx = random.randint(0, len(all_triples) - 1)
                if is_valid_idx(add_idx, new_indices):
                    new_indices.append(add_idx)
                    added = True
                    break
            if not added:
                # Keep the smaller set
                pass
        else:
            new_indices = [random.randint(0, len(all_triples) - 1)]
            
        if steps % 5 == 0:
            new_alpha = calc_alpha(new_indices, current_alpha)
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                current_indices = new_indices
                no_improve = 0
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_indices = current_indices[:]
            else:
                no_improve += 1
                if no_improve > 200:
                    # Restart
                    if random.random() < 0.5:
                        current_indices = best_indices[:]
                        current_alpha = best_alpha
                    else:
                        current_indices = greedy()
                        current_alpha = calc_alpha(current_indices)
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_indices = current_indices[:]
                    no_improve = 0
                    
    return [all_triples[i] for i in best_indices]
