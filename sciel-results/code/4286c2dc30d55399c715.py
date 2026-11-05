# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes
    
    # Precompute weights as Fractions for exact arithmetic
    weights = {v: Fraction(1, v) for v in nodes}
    
    # Precompute conflict masks
    # conflict_mask[i] is an integer bitmask where bit j is set if node i and node j conflict (lcm <= 256)
    conflict_mask = [0] * num_nodes
    
    for i in range(num_nodes):
        vi = idx_to_val[i]
        mask = 0
        for j in range(num_nodes):
            if i == j:
                continue
            vj = idx_to_val[j]
            if math.lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask

    def get_score(mask):
        score = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += weights[idx_to_val[idx]]
            m ^= lsb
        return score

    def local_search(initial_mask, time_limit=5.0):
        current_mask = initial_mask
        current_score = get_score(current_mask)
        improved = True
        start_ls_time = time.time()
        
        while improved:
            improved = False
            
            # Check time
            if time.time() > deadline or time.time() - start_ls_time > time_limit:
                break
            
            # Add phase: try to add any v not in mask
            # Prioritize adding smaller v (higher weight)
            for v in range(2, 257):
                idx = val_to_idx[v]
                if not (current_mask & (1 << idx)):
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_score += weights[v]
                        improved = True
            
            # Build lists once per iteration
            in_mask = []
            not_in_mask = []
            for i in range(num_nodes):
                if current_mask & (1 << i):
                    in_mask.append(i)
                else:
                    not_in_mask.append(i)
            
            # 1-for-1 swap: remove u, add v where v < u
            in_mask_desc = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            
            found_swap = False
            for u_idx in in_mask_desc:
                if time.time() > deadline:
                    return current_mask, current_score
                u_val = idx_to_val[u_idx]
                mask_without_u = current_mask & ~(1 << u_idx)
                
                for v_idx in not_in_mask_asc:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        break  # Since sorted ascending, no more candidates
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        # Calculate gain exactly
                        gain = weights[v_val] - weights[u_val]
                        if gain > 0:
                            current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                            current_score += gain
                            improved = True
                            found_swap = True
                            break
                
                if found_swap:
                    break
            
            # 2-for-1 swap: remove u1, u2, add v where 1/v > 1/u1 + 1/u2
            if not found_swap:
                in_mask_desc2 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
                not_in_mask_asc2 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
                found_2swap = False
                for i1 in range(len(in_mask_desc2)):
                    if time.time() > deadline:
                        return current_mask, current_score
                    u1_idx = in_mask_desc2[i1]
                    u1_val = idx_to_val[u1_idx]
                    for i2 in range(i1+1, len(in_mask_desc2)):
                        u2_idx = in_mask_desc2[i2]
                        u2_val = idx_to_val[u2_idx]
                        gain_needed = weights[u1_val] + weights[u2_val]
                        mask_without_u1_u2 = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)
                        
                        for v_idx in not_in_mask_asc2:
                            v_val = idx_to_val[v_idx]
                            if weights[v_val] <= gain_needed:
                                break
                            if (conflict_mask[v_idx] & mask_without_u1_u2) == 0:
                                gain = weights[v_val] - gain_needed
                                if gain > 0:
                                    current_mask = (current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)) | (1 << v_idx)
                                    current_score += gain
                                    improved = True
                                    found_2swap = True
                                    break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
            
            # 1-for-2 swap: remove u, add v1, v2 where 1/v1 + 1/v2 > 1/u
            if not found_swap and not found_2swap:
                in_mask_desc3 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
                not_in_mask_asc3 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
                found_12swap = False
                for u_idx in in_mask_desc3:
                    if time.time() > deadline:
                        return current_mask, current_score
                    u_val = idx_to_val[u_idx]
                    gain_needed = weights[u_val]
                    mask_without_u = current_mask & ~(1 << u_idx)
                    
                    for v1_idx in not_in_mask_asc3:
                        v1_val = idx_to_val[v1_idx]
                        if (conflict_mask[v1_idx] & mask_without_u) != 0:
                            continue
                        if weights[v1_val] >= gain_needed:
                            # Single add would have been caught in add phase, so skip
                            continue
                        mask_with_v1 = mask_without_u | (1 << v1_idx)
                        for v2_idx in not_in_mask_asc3:
                            if v2_idx == v1_idx:
                                continue
                            v2_val = idx_to_val[v2_idx]
                            if weights[v1_val] + weights[v2_val] <= gain_needed:
                                break
                            if (conflict_mask[v2_idx] & mask_with_v1) != 0:
                                continue
                            gain = weights[v1_val] + weights[v2_val] - gain_needed
                            if gain > 0:
                                current_mask = (current_mask & ~(1 << u_idx)) | (1 << v1_idx) | (1 << v2_idx)
                                current_score += gain
                                improved = True
                                found_12swap = True
                                break
                        if found_12swap:
                            break
                    if found_12swap:
                        break
            
            # Rebuild in_mask/not_in_mask if we made a change
            if improved:
                in_mask = []
                not_in_mask = []
                for i in range(num_nodes):
                    if current_mask & (1 << i):
                        in_mask.append(i)
                    else:
                        not_in_mask.append(i)
        
        return current_mask, current_score

    def perturb(mask):
        # Remove a random element from the mask
        if mask == 0:
            return mask
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        
        if not bits:
            return mask
            
        # Remove 1-3 random elements, biased towards larger values (lower weight)
        # so we keep the high-value elements
        num_remove = random.choice([1, 1, 2, 2, 3])
        # Sort bits by value descending, pick from the tail (larger values)
        bits_sorted = sorted(bits, key=lambda i: idx_to_val[i], reverse=True)
        # Pick from the last 50% (larger values)
        half = max(1, len(bits_sorted) // 2)
        candidates = bits_sorted[half:]
        if len(candidates) < num_remove:
            candidates = bits_sorted
        indices_to_remove = random.sample(candidates, min(num_remove, len(candidates)))
        
        new_mask = mask
        for idx in indices_to_remove:
            new_mask &= ~(1 << idx)
        return new_mask

    best_mask = 0
    best_score = Fraction(0)

    # Generate diverse initial solutions
    # 1. Greedy by weight (small to large)
    mask1 = 0
    for v in range(2, 257):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask1) == 0:
            mask1 |= (1 << idx)
    
    # 2. Large numbers set {129..256}
    mask2 = 0
    for v in range(129, 257):
        idx = val_to_idx[v]
        mask2 |= (1 << idx)

    # 3. Randomized Greedy
    def randomized_greedy():
        m = 0
        available = list(range(2, 257))
        while available:
            # Filter available to those compatible with current mask
            compatible = [v for v in available if (conflict_mask[val_to_idx[v]] & m) == 0]
            if not compatible:
                break
            # Pick one from compatible. 
            # Bias towards smaller values (higher weight).
            compatible.sort()
            # Pick randomly from the first few smallest (highest weight)
            k = min(5, len(compatible))
            v = random.choice(compatible[:k])
            idx = val_to_idx[v]
            m |= (1 << idx)
            available.remove(v)
        return m

    mask3 = randomized_greedy()
    
    initial_masks = [mask1, mask2, mask3]
    
    # Add more randomized starts if time permits
    num_restarts = 0
    max_restarts = 30
    
    for i in range(max_restarts):
        if time.time() > deadline - 15.0: # Reserve time for polish
            break
            
        # Generate a new initial mask via randomized greedy
        m_init = randomized_greedy()
        
        # Local search
        m_ls, s_ls = local_search(m_init, time_limit=2.0)
        
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
            
        num_restarts += 1
        
        # If we found a good solution, we can try to perturb it
        if num_restarts % 4 == 0 and best_mask != 0:
             m_pert = perturb(best_mask)
             m_ls2, s_ls2 = local_search(m_pert, time_limit=2.0)
             if s_ls2 > best_score:
                 best_score = s_ls2
                 best_mask = m_ls2

    # Ensure we run LS on the initial specific masks if not done
    if mask1 != best_mask:
        m1, s1 = local_search(mask1, time_limit=5.0)
        if s1 > best_score:
            best_score = s1
            best_mask = m1
            
    if mask2 != best_mask:
        m2, s2 = local_search(mask2, time_limit=5.0)
        if s2 > best_score:
            best_score = s2
            best_mask = m2

    if mask3 != best_mask:
        m3, s3 = local_search(mask3, time_limit=5.0)
        if s3 > best_score:
            best_score = s3
            best_mask = m3

    # Multiple polish passes with perturbation to escape local optima
    if time.time() < deadline - 5.0:
        for polish_iter in range(5):
            if time.time() > deadline - 3.0:
                break
            # Perturb the best solution
            m_pert = perturb(best_mask)
            remaining = deadline - time.time() - 2.0
            if remaining < 1.0:
                break
            m_pol, s_pol = local_search(m_pert, time_limit=min(remaining, 4.0))
            if s_pol > best_score:
                best_score = s_pol
                best_mask = m_pol
            else:
                # If perturbation didn't help, try a direct polish
                remaining = deadline - time.time() - 2.0
                if remaining < 1.0:
                    break
                m_pol2, s_pol2 = local_search(best_mask, time_limit=min(remaining, 3.0))
                if s_pol2 > best_score:
                    best_score = s_pol2
                    best_mask = m_pol2
                else:
                    break  # No improvement, stop

    # Final polish on the best found
    if time.time() < deadline - 1.0:
        m_final, s_final = local_search(best_mask, time_limit=deadline - time.time() - 0.5)
        if s_final > best_score:
            best_score = s_final
            best_mask = m_final

    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx_to_val[idx])
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
