import time
import random
import math
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i][j] = 1 if i and j are comparable (strict subset or superset)
    comp_np = np.zeros((total_masks, total_masks), dtype=np.uint8)
    comp_list = [[] for _ in range(total_masks)]
    
    # Optimization: Use symmetry
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            # i is subset of j OR j is subset of i
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp_np[i, j] = 1
                comp_np[j, i] = 1
                comp_list[i].append(j)
                comp_list[j].append(i)

    # Helper to convert mask to list
    def mask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]

    # Helper to convert list to mask
    def list_to_mask(lst):
        mask = 0
        for b in lst:
            mask |= (1 << b)
        return mask

    # 1. Initialization: Chain + Greedy
    def greedy_init():
        family = []
        used = set()
        
        # Add maximal chain: 0, 1, 3, 7, 15, ...
        cur = 0
        family.append(cur)
        used.add(cur)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
        
        # Precompute scores for all masks relative to current family
        # Use numpy for speed
        scores = np.zeros(total_masks, dtype=np.int32)
        for f in family:
            scores[comp_list[f]] += 1 # This is slow if done in loop.
            # Better: scores += comp_np[:, f]
            # But comp_np[:, f] is a view.
            # Let's just use the loop for init, it's only 13 sets.
            
        # Actually, let's just use the numpy matrix for init too.
        scores = np.zeros(total_masks, dtype=np.int32)
        for f in family:
            scores += comp_np[:, f]
            
        # Greedily add remaining sets
        in_fam_mask = np.zeros(total_masks, dtype=bool)
        in_fam_mask[family] = True
        
        while len(family) < m:
            # Mask out used sets
            masked_scores = scores.copy()
            masked_scores[in_fam_mask] = -1
            
            best_mask = np.argmax(masked_scores)
            family.append(best_mask)
            in_fam_mask[best_mask] = True
            
            # Update scores
            scores += comp_np[:, best_mask]
            
        return family

    def evaluate(fam):
        # Returns score and state variables
        in_fam = np.zeros(total_masks, dtype=bool)
        in_fam[fam] = True
            
        sum_comp = np.zeros(total_masks, dtype=np.int32)
        for f in fam:
            sum_comp += comp_np[:, f]
            
        contribs = np.array([sum_comp[f] for f in fam], dtype=np.int32)
        current_score = np.sum(contribs) // 2
        return current_score, in_fam, sum_comp, contribs

    # Multiple random restarts of initialization to find a good start
    best_family = None
    best_score = -1
    best_in_fam = None
    best_sum_comp = None
    best_contribs = None
    
    # Try a few different greedy inits with slight randomness in tie-breaking?
    # Or just run the deterministic one, it's usually strong.
    # Let's run the deterministic chain-greedy once.
    best_family = greedy_init()
    best_score, best_in_fam, best_sum_comp, best_contribs = evaluate(best_family)
    
    # State for current search
    fam = np.array(best_family, dtype=np.int32)
    in_fam = best_in_fam.copy()
    sum_comp = best_sum_comp.copy()
    contribs = best_contribs.copy()
    current_score = best_score
    
    deadline = time.time() + 58 # 58s to leave 2s for overhead and return
    
    T = 5.0
    cooling = 0.9995
    T_min = 0.01
    
    no_improve = 0
    top_k = 10 # Increased from 5
    
    restarts = 0
    max_restarts = 50 # Allow more restarts if we have time
    
    # Precompute indices for 2-swap
    # We need to iterate over pairs (i1, i2)
    
    while time.time() < deadline:
        # 1. Find best 1-swap and collect top candidates for 2-swap
        best_1_swap_delta = -1000
        best_1_swap_i = -1
        best_1_swap_s = -1
        
        top_candidates = [[] for _ in range(m)]
        
        for i in range(m):
            old_val = fam[i]
            
            # Calculate gain for swapping fam[i] with s
            # Delta = (sum_comp[s] - comp[s][old_val]) - contribs[i]
            
            # Vectorized gain calculation
            # gains[s] = sum_comp[s] - comp_np[s, old_val]
            gains = sum_comp - comp_np[:, old_val]
            
            # Mask out used sets
            gains[in_fam] = -1000 # Use a very low value to avoid selection
            
            # Get top K candidates
            # np.argpartition is faster for top K
            # But we need the values too.
            # For small K, sorting is fine.
            # Or just take the max for best_1_swap, and store top K for 2-swap.
            
            # Find best 1-swap
            best_s_idx = np.argmax(gains)
            best_gain_val = gains[best_s_idx]
            
            if best_gain_val > -500: # Check if valid
                 delta = best_gain_val - contribs[i]
                 if delta > best_1_swap_delta:
                     best_1_swap_delta = delta
                     best_1_swap_i = i
                     best_1_swap_s = best_s_idx

            # Get top K for 2-swap
            # We need the top K indices and their gains
            # np.argsort(gains)[::-1][:top_k]
            # This is O(N log N). N=4096. 40 times. ~160k ops.
            # We can use np.argpartition
            if top_k < total_masks:
                # Get top K indices
                top_indices = np.argpartition(gains, -top_k)[-top_k:]
                # Sort them by gain descending
                top_indices = top_indices[np.argsort(gains[top_indices])[::-1]]
                top_candidates[i] = [(gains[idx], idx) for idx in top_indices]
            else:
                top_indices = np.argsort(gains)[::-1]
                top_candidates[i] = [(gains[idx], idx) for idx in top_indices]

        # 2. Find best 2-swap using top candidates
        best_2_swap_delta = -1000
        best_2_swap_i1 = -1
        best_2_swap_i2 = -1
        best_2_swap_s1 = -1
        best_2_swap_s2 = -1
        
        # Only do 2-swap if it might beat 1-swap or if 1-swap is not great
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                if not top_candidates[i1] or not top_candidates[i2]:
                    continue
                
                old1 = fam[i1]
                old2 = fam[i2]
                
                for delta1, s1 in top_candidates[i1]:
                    d1 = delta1 - contribs[i1]
                    
                    for delta2, s2 in top_candidates[i2]:
                        if s1 == s2:
                            continue
                        d2 = delta2 - contribs[i2]
                        old_int = comp_np[old1, old2]
                        new_int = comp_np[s1, s2]
                        cross = comp_np[s1, old2] + comp_np[s2, old1]
                        
                        total_delta = d1 + d2 + old_int - cross + new_int
                        
                        if total_delta > best_2_swap_delta:
                            best_2_swap_delta = total_delta
                            best_2_swap_i1 = i1
                            best_2_swap_i2 = i2
                            best_2_swap_s1 = s1
                            best_2_swap_s2 = s2

        # Decide move
        accept = False
        move_type = 0
        move_i1 = -1
        move_i2 = -1
        move_s1 = -1
        move_s2 = -1
        move_delta = 0
        
        if best_2_swap_delta > best_1_swap_delta:
            if best_2_swap_delta > 0:
                accept = True
                move_type = 2
                move_i1 = best_2_swap_i1
                move_i2 = best_2_swap_i2
                move_s1 = best_2_swap_s1
                move_s2 = best_2_swap_s2
                move_delta = best_2_swap_delta
            else:
                if random.random() < math.exp(best_2_swap_delta / T):
                    accept = True
                    move_type = 2
                    move_i1 = best_2_swap_i1
                    move_i2 = best_2_swap_i2
                    move_s1 = best_2_swap_s1
                    move_s2 = best_2_swap_s2
                    move_delta = best_2_swap_delta
        else:
            if best_1_swap_delta > 0:
                accept = True
                move_type = 1
                move_i1 = best_1_swap_i
                move_s1 = best_1_swap_s
                move_delta = best_1_swap_delta
            else:
                if random.random() < math.exp(best_1_swap_delta / T):
                    accept = True
                    move_type = 1
                    move_i1 = best_1_swap_i
                    move_s1 = best_1_swap_s
                    move_delta = best_1_swap_delta

        if accept:
            if move_type == 1:
                old_val = fam[move_i1]
                new_val = move_s1
                
                fam[move_i1] = new_val
                in_fam[old_val] = False
                in_fam[new_val] = True
                
                # Efficient update of sum_comp without sets
                # sum_comp += comp[:, new] - comp[:, old]
                sum_comp += comp_np[:, new_val] - comp_np[:, old_val]
                
                # Update contribs
                # contribs[k] += comp[fam[k], new] - comp[fam[k], old]
                # Note: fam[move_i1] is already updated to new_val
                # So comp[fam, new] at index move_i1 is comp[new_val, new_val] = 0
                # comp[fam, old] at index move_i1 is comp[new_val, old_val]
                
                delta_contribs = comp_np[fam, new_val] - comp_np[fam, old_val]
                contribs += delta_contribs
                contribs[move_i1] = sum_comp[new_val]
                        
            elif move_type == 2:
                old1 = fam[move_i1]
                old2 = fam[move_i2]
                new1 = move_s1
                new2 = move_s2
                
                fam[move_i1] = new1
                fam[move_i2] = new2
                
                in_fam[old1] = False
                in_fam[old2] = False
                in_fam[new1] = True
                in_fam[new2] = True
                
                # Efficient update of sum_comp
                sum_comp += comp_np[:, new1] + comp_np[:, new2] - comp_np[:, old1] - comp_np[:, old2]
                
                # Update contribs
                # contribs[k] += comp[fam[k], new1] + comp[fam[k], new2] - comp[fam[k], old1] - comp[fam[k], old2]
                delta_contribs = comp_np[fam, new1] + comp_np[fam, new2] - comp_np[fam, old1] - comp_np[fam, old2]
                contribs += delta_contribs
                
                # Correct the two changed indices
                contribs[move_i1] = sum_comp[new1]
                contribs[move_i2] = sum_comp[new2]

            current_score = np.sum(contribs) // 2
            
            if current_score > best_score:
                best_score = current_score
                best_family = fam.copy()
                best_in_fam = in_fam.copy()
                best_sum_comp = sum_comp.copy()
                best_contribs = contribs.copy()
                no_improve = 0
            else:
                no_improve += 1
                
            T *= cooling
            if T < T_min:
                T = T_min
                
            # Restart logic
            if no_improve > 50: # Lower threshold for faster restarts
                # Perturb the best solution
                fam = best_family.copy()
                in_fam = best_in_fam.copy()
                sum_comp = best_sum_comp.copy()
                contribs = best_contribs.copy()
                
                # Perform random swaps to kick out of local optimum
                num_perturb = 8
                for _ in range(num_perturb):
                    i = random.randint(0, m - 1)
                    old_val = fam[i]
                    # Pick a random unused mask
                    while True:
                        s = random.randint(0, total_masks - 1)
                        if not in_fam[s]:
                            break
                    new_val = s
                    fam[i] = new_val
                    in_fam[old_val] = False
                    in_fam[new_val] = True
                    # Update sum_comp
                    sum_comp += comp_np[:, new_val] - comp_np[:, old_val]
                    # Update contribs
                    delta_contribs = comp_np[fam, new_val] - comp_np[fam, old_val]
                    contribs += delta_contribs
                    contribs[i] = sum_comp[new_val]
                    
                # Re-evaluate state after perturbation to ensure consistency
                # The incremental updates should be correct, but let's double check score
                # Actually, just re-evaluating is safer and fast enough
                current_score, in_fam, sum_comp, contribs = evaluate(fam)
                
                T = 2.0 # Reheat
                no_improve = 0
                restarts += 1
                if restarts > max_restarts:
                    # Try a completely new greedy init?
                    # For now, just keep perturbing best.
                    pass
        else:
            no_improve += 1
            T *= cooling
            if T < T_min:
                T = T_min

    # Final Sanitization: Ensure distinct sets
    # fam is a numpy array of ints
    if len(set(fam.tolist())) < m:
        unique_masks = list(set(fam.tolist()))
        used_set = set(unique_masks)
        random.shuffle(unique_masks)
        
        while len(unique_masks) < m:
            s = random.randint(0, total_masks - 1)
            if s not in used_set:
                unique_masks.append(s)
                used_set.add(s)
        
        best_family = np.array(unique_masks[:m], dtype=np.int32)
        
    result = []
    for mask in best_family:
        result.append(mask_to_list(int(mask)))
        
    return result

# Quick self-test to ensure no crashes
if __name__ == "__main__":
    res = construct()
    assert len(res) == 40
    for r in res:
        assert isinstance(r, list)
        for x in r:
            assert isinstance(x, int)
    print("Score:", end=" ")
    comp = [[0]*4096 for _ in range(4096)]
    for i in range(4096):
        for j in range(i+1, 4096):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i][j] = 1
                comp[j][i] = 1
    masks = [0]*40
    for i, l in enumerate(res):
        m = 0
        for b in l:
            m |= (1<<b)
        masks[i] = m
    score = 0
    for i in range(40):
        for j in range(i+1, 40):
            score += comp[masks[i]][masks[j]]
    print(score)
