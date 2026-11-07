import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 75  # Safety margin before 90s kill

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if is_comparable(config[i], config[j]):
                    cnt += 1
        return cnt

    # Precompute comparability matrix
    masks = np.arange(total_masks, dtype=np.int32)
    comp_mat = np.zeros((total_masks, total_masks), dtype=bool)
    chunk_size = 512
    for start in range(0, total_masks, chunk_size):
        end = min(start + chunk_size, total_masks)
        a_chunk = masks[start:end, np.newaxis]
        b_chunk = masks[np.newaxis, :]
        a_sub_b = (a_chunk & ~b_chunk) == 0
        b_sub_a = (b_chunk & ~a_chunk) == 0
        comp_mat[start:end, :] = (a_sub_b | b_sub_a)
    np.fill_diagonal(comp_mat, False)
    comp_int = comp_mat.astype(np.int32)

    def build_compat(selected_arr):
        sel_np = np.array(selected_arr, dtype=np.int32)
        compat = comp_int[:, sel_np].sum(axis=1)
        return compat

    def run_local_search(initial_config):
        selected = initial_config[:]
        current_score = count_comparable_pairs(selected)
        best_config = selected[:]
        best_score = current_score
        compat = build_compat(selected)
        selected_set = set(selected)
        
        no_improve_passes = 0
        max_no_improve = 20
        
        while time.time() < deadline:
            improved = False
            for i in range(m):
                old_mask = selected[i]
                loss = int(compat[old_mask])
                
                valid = np.ones(total_masks, dtype=bool)
                valid[old_mask] = False
                for s in selected_set:
                    valid[s] = False
                
                gains = compat[valid]
                if len(gains) == 0:
                    continue
                best_idx = int(np.argmax(gains))
                best_delta = int(gains[best_idx]) - loss
                
                if best_delta <= 0:
                    continue
                
                valid_masks = np.where(valid)[0]
                best_new_mask = int(valid_masks[best_idx])
                
                selected_set.remove(old_mask)
                selected_set.add(best_new_mask)
                selected[i] = best_new_mask
                current_score += best_delta
                
                compat -= comp_int[:, old_mask]
                compat += comp_int[:, best_new_mask]
                
                improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_config = selected[:]
            
            if improved:
                no_improve_passes = 0
            else:
                no_improve_passes += 1
                if no_improve_passes >= max_no_improve:
                    # Perturbation
                    success = False
                    for _attempt in range(50):
                        i1 = random.randint(0, m - 1)
                        i2 = random.randint(0, m - 1)
                        if i1 == i2:
                            continue
                        old1 = selected[i1]
                        old2 = selected[i2]
                        
                        m1 = random.randint(0, total_masks - 1)
                        m2 = random.randint(0, total_masks - 1)
                        if m1 in selected_set or m2 in selected_set:
                            continue
                        if m1 == m2:
                            continue
                        if m1 == old1 or m2 == old2:
                            continue
                        
                        old_score = current_score
                        selected_set.remove(old1)
                        selected_set.remove(old2)
                        selected_set.add(m1)
                        selected_set.add(m2)
                        selected[i1] = m1
                        selected[i2] = m2
                        new_score = count_comparable_pairs(selected)
                        
                        if new_score > old_score:
                            current_score = new_score
                            compat = build_compat(selected)
                            no_improve_passes = 0
                            success = True
                            break
                        else:
                            selected_set.remove(m1)
                            selected_set.remove(m2)
                            selected_set.add(old1)
                            selected_set.add(old2)
                            selected[i1] = old1
                            selected[i2] = old2
                            current_score = old_score
                    if not success:
                        break
        return best_config, best_score

    # Initialize with B5 + 8 strategy
    # B5: all subsets of {0,1,2,3,4}
    base_sets = []
    for mask in range(1 << 5):
        base_sets.append(mask)
    
    # T': 8 largest subsets of {0,1,2,3,4} (B4)
    # Sizes: 1 of size 4, 5 of size 3, 2 of size 2
    # We want to maximize sum of 2^{|B'|} + internal pairs
    # Let's just pick the lexicographically largest or specific ones
    # For now, pick the 8 subsets of B4 with largest sizes
    b4_subsets = list(range(1 << 4))
    b4_subsets.sort(key=lambda x: (bin(x).count('1'), x), reverse=True)
    extra_projections = b4_subsets[:8]
    
    # Construct T: add element 4 (bit 4) to each projection
    extra_sets = [p | (1 << 4) for p in extra_projections]
    
    initial_config = base_sets + extra_sets
    
    best_overall_config, best_overall_score = run_local_search(initial_config)
    
    # Try a few random restarts if time permits
    start_time = time.time()
    while time.time() < deadline - 10:
        random_config = random.sample(range(total_masks), m)
        config, score = run_local_search(random_config)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_config = config

    # Convert to lists of ints
    result = []
    for mask in best_overall_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result
