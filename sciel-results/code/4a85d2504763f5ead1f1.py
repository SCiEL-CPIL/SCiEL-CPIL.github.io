import time
import random
from itertools import combinations

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85

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

    # Precompute comparability for all 4096 masks
    total_masks = 1 << n
    # comp[i][j] = 1 if comparable
    # Use a list of bitsets or just a 2D array? 4096*4096 is ~16M entries.
    # Python lists of lists are slow. Let's use a list of bytearrays or similar.
    # Or, since we only need to check specific pairs during search, maybe precompute
    # neighbor lists for each mask?
    
    # Actually, let's precompute a boolean matrix as a list of lists of ints (0/1)
    # 4096 * 4096 * 1 byte = 16 MB. Acceptable.
    # But Python object overhead is high.
    # Let's try a different approach: for each mask, precompute the list of comparable masks.
    # This might be memory heavy if lists are long, but average degree is moderate.
    
    # Alternative: Precompute a 2D numpy array?
    # import numpy as np
    # comp = np.zeros((total_masks, total_masks), dtype=np.uint8)
    # This is fast for indexing.
    
    import numpy as np
    
    comp = np.zeros((total_masks, total_masks), dtype=np.uint8)
    # Symmetric, so fill upper triangle
    # This loop is 4096^2 / 2 ~ 8M iterations. In Python this might be slow.
    # Let's vectorize or use a faster method.
    
    # Method: For each mask a, its subsets are {b | b & ~a == 0}.
    # Its supersets are {b | a & ~b == 0}.
    # We can iterate over masks and mark comparability.
    
    # Faster precomputation using bit operations and numpy?
    # Or just a simple loop with optimization.
    # 8M iterations in Python might take ~1-2 seconds. Acceptable.
    
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if is_comparable(i, j):
                comp[i, j] = 1
                comp[j, i] = 1

    # Greedy initialization
    selected = []
    available = set(range(total_masks))
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in available:
            score = 0
            for s in selected:
                if comp[mask, s]:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        selected.append(best_mask)
        available.remove(best_mask)
        
    current_score = count_comparable_pairs(selected)
    best_config = selected[:]
    best_score = current_score
    selected_set = set(selected)

    # Local Search: 1-move (replace one element)
    # For speed, use numpy for gain calculation?
    # gain = sum(comp[new_mask, s] for s in selected if s != old_mask)
    # This is a dot product of comp[new_mask, :] and a mask of selected indices.
    
    # Let's precompute a numpy array of selected indices?
    # Or just use the comp matrix directly.
    
    def get_gain(new_mask, selected_list, skip_idx=None):
        # selected_list is a list of masks
        # We want sum of comp[new_mask, s] for s in selected_list, s != selected_list[skip_idx]
        # Using numpy:
        # indices = np.array(selected_list)
        # if skip_idx is not None:
        #    mask_arr = np.ones(len(indices), dtype=np.uint8)
        #    mask_arr[skip_idx] = 0
        #    return np.sum(comp[new_mask, indices] * mask_arr)
        # else:
        #    return np.sum(comp[new_mask, indices])
        
        # To avoid creating numpy arrays every time, we can keep a numpy array of selected masks.
        # But selected changes.
        # Let's just use the Python loop for now, it's only 40 iterations.
        cnt = 0
        for j, s in enumerate(selected_list):
            if j == skip_idx:
                continue
            if comp[new_mask, s]:
                cnt += 1
        return cnt

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            old_mask = selected[i]
            loss = get_gain(old_mask, selected, i)
            
            best_new_mask = -1
            best_delta = 0
            
            # Search all masks for best replacement
            # 4096 iterations * 40 checks = 160k ops per i. 40 i's = 6.4M ops per pass.
            # Might be slow in Python.
            
            # Optimization: Use numpy to compute gains for all candidate masks at once?
            # candidates = np.array(list(available)) # This is expensive
            # gains = comp[candidates, selected_indices].sum(axis=1)
            
            # Let's stick to Python loop but maybe restrict candidates?
            # Or use the numpy comp matrix efficiently.
            
            # Let's try a vectorized approach for the inner loop.
            # We have a fixed list of selected masks (except one).
            # Let's create a numpy array of the current selected masks.
            # sel_np = np.array(selected)
            # We want to find argmax over all masks k of (sum_{j != i} comp[k, sel[j]])
            # This is equivalent to:
            # gains = comp[:, sel_np].sum(axis=1) - comp[:, sel_np[i]]
            # This is a matrix-vector multiplication essentially.
            # comp is 4096x4096. sel_np is 40.
            # comp[:, sel_np] is 4096x40. Sum is 4096.
            # This is fast in numpy.
            
            sel_np = np.array(selected, dtype=np.int32)
            # Compute scores for all masks
            # scores[k] = sum(comp[k, s] for s in selected) - comp[k, old_mask]
            # Note: comp[k, old_mask] is the contribution of the old mask itself, which we remove.
            # Wait, the loss is the contribution of old_mask to the rest.
            # The gain of new_mask is its contribution to the rest.
            # Delta = Gain(new) - Loss(old)
            # Loss(old) = sum(comp[old, s] for s in selected, s != old)
            # Gain(new) = sum(comp[new, s] for s in selected, s != old)
            
            # Let's compute S[k] = sum(comp[k, s] for s in selected)
            # Then Gain(new) = S[new] - comp[new, old]
            # Loss(old) = S[old] - comp[old, old] (which is 0) = S[old]
            # So Delta = S[new] - comp[new, old] - S[old]
            
            S = comp[:, sel_np].sum(axis=1) # Shape (4096,)
            S_old = S[old_mask]
            
            # We want to maximize S[new] - comp[new, old_mask]
            # subject to new not in selected (except we are replacing old, so new != old is implied if we check)
            # and new not in selected_set
            
            # Let's create a mask of valid candidates
            valid = np.ones(total_masks, dtype=bool)
            valid[sel_np] = False # Exclude all currently selected
            # Actually, we can replace old with old? No, delta would be 0.
            # We can allow new == old, delta 0.
            # But we must exclude other selected masks.
            
            # scores = S - comp[:, old_mask]
            scores = S - comp[:, old_mask]
            scores[~valid] = -1 # Invalidate
            
            best_new_mask = np.argmax(scores)
            best_delta = scores[best_new_mask] - S_old
            
            if best_delta > 0:
                # Update
                selected[i] = best_new_mask
                selected_set.remove(old_mask)
                selected_set.add(best_new_mask)
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_config = selected[:]

        # 2-move search if 1-move stagnates?
        # For now, let's stick to 1-move with numpy acceleration.
        # If time permits, add 2-move.

    # Random restarts?
    # The numpy acceleration makes each pass fast.
    # Let's do multiple restarts if time allows.
    
    start_time = time.time()
    best_score_global = best_score
    best_config_global = best_config
    
    while time.time() < deadline - 5:
        # Restart
        random.seed(int(time.time() * 1000) % 10000)
        selected_r = []
        available_r = set(range(total_masks))
        
        # Randomized Greedy
        for _ in range(m):
            candidates = list(available_r)
            random.shuffle(candidates)
            # Pick top few and choose best
            top_k = 10
            best_mask = -1
            best_score_r = -1
            for mask in candidates[:top_k]:
                score = 0
                for s in selected_r:
                    if comp[mask, s]:
                        score += 1
                if score > best_score_r:
                    best_score_r = score
                    best_mask = mask
            selected_r.append(best_mask)
            available_r.remove(best_mask)
            
        current_score_r = count_comparable_pairs(selected_r)
        
        # Local Search on this restart
        improved_r = True
        while improved_r and time.time() < deadline - 5:
            improved_r = False
            sel_np = np.array(selected_r, dtype=np.int32)
            S = comp[:, sel_np].sum(axis=1)
            
            for i in range(m):
                if time.time() > deadline - 5:
                    break
                old_mask = selected_r[i]
                S_old = S[old_mask]
                
                valid = np.ones(total_masks, dtype=bool)
                valid[sel_np] = False
                
                scores = S - comp[:, old_mask]
                scores[~valid] = -1
                
                best_new_mask = np.argmax(scores)
                best_delta = scores[best_new_mask] - S_old
                
                if best_delta > 0:
                    selected_r[i] = best_new_mask
                    # Update S?
                    # S depends on selected_r.
                    # This is tricky. S is a global sum.
                    # If we change selected_r[i], S changes for all masks.
                    # S[k] = sum(comp[k, s_j])
                    # New S'[k] = S[k] - comp[k, old] + comp[k, new]
                    # We can update S incrementally.
                    S = S - comp[:, old_mask] + comp[:, best_new_mask]
                    sel_np[i] = best_new_mask
                    current_score_r += best_delta
                    improved_r = True
                    if current_score_r > best_score_global:
                        best_score_global = current_score_r
                        best_config_global = selected_r[:]

    result = []
    for mask in best_config_global:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result
