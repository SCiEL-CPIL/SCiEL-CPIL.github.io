# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes
    
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
        score = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += 1.0 / idx_to_val[idx]
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
                        current_score += 1.0 / v
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
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                        current_score += 1.0 / v_val - 1.0 / u_val
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
                        gain_needed = 1.0/u1_val + 1.0/u2_val
                        mask_without_u1_u2 = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)
                        
                        for v_idx in not_in_mask_asc2:
                            v_val = idx_to_val[v_idx]
                            if 1.0/v_val <= gain_needed:
                                break
                            if (conflict_mask[v_idx] & mask_without_u1_u2) == 0:
                                current_mask = (current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)) | (1 << v_idx)
                                current_score += 1.0/v_val - 1.0/u1_val - 1.0/u2_val
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
                    gain_needed = 1.0/u_val
                    mask_without_u = current_mask & ~(1 << u_idx)
                    
                    for v1_idx in not_in_mask_asc3:
                        v1_val = idx_to_val[v1_idx]
                        if (conflict_mask[v1_idx] & mask_without_u) != 0:
                            continue
                        if 1.0/v1_val >= gain_needed:
                            # Single add would have been caught in add phase, so skip
                            continue
                        mask_with_v1 = mask_without_u | (1 << v1_idx)
                        for v2_idx in not_in_mask_asc3:
                            if v2_idx == v1_idx:
                                continue
                            v2_val = idx_to_val[v2_idx]
                            if 1.0/v1_val + 1.0/v2_val <= gain_needed:
                                break
                            if (conflict_mask[v2_idx] & mask_with_v1) != 0:
                                continue
                            current_mask = (current_mask & ~(1 << u_idx)) | (1 << v1_idx) | (1 << v2_idx)
                            current_score += 1.0/v1_val + 1.0/v2_val - 1.0/u_val
                            improved = True
                            found_12swap = True
                            break
                        if found_12swap:
                            break
                    if found_12swap:
                        break
            
            # 2-for-2 swap: remove u1, u2, add v1, v2 where 1/v1 + 1/v2 > 1/u1 + 1/u2
            if not found_swap and not found_2swap and not found_12swap:
                in_mask_desc4 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
                not_in_mask_asc4 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
                found_22swap = False
                
                # Limit search space for 2-for-2 to avoid O(n^4) explosion
                # Only try swapping the largest 20 elements
                limit = min(20, len(in_mask_desc4))
                for i1 in range(limit):
                    if time.time() > deadline:
                        return current_mask, current_score
                    u1_idx = in_mask_desc4[i1]
                    u1_val = idx_to_val[u1_idx]
                    for i2 in range(i1+1, limit):
                        u2_idx = in_mask_desc4[i2]
                        u2_val = idx_to_val[u2_idx]
                        gain_needed = 1.0/u1_val + 1.0/u2_val
                        mask_without_u1_u2 = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)
                        
                        for v1_idx in not_in_mask_asc4:
                            v1_val = idx_to_val[v1_idx]
                            if (conflict_mask[v1_idx] & mask_without_u1_u2) != 0:
                                continue
                            if 1.0/v1_val >= gain_needed:
                                # This would be a 1-for-2 or 1-for-1, handled earlier
                                continue
                            mask_with_v1 = mask_without_u1_u2 | (1 << v1_idx)
                            for v2_idx in not_in_mask_asc4:
                                if v2_idx == v1_idx:
                                    continue
                                v2_val = idx_to_val[v2_idx]
                                if 1.0/v1_val + 1.0/v2_val <= gain_needed:
                                    break
                                if (conflict_mask[v2_idx] & mask_with_v1) != 0:
                                    continue
                                current_mask = (current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)) | (1 << v1_idx) | (1 << v2_idx)
                                current_score += 1.0/v1_val + 1.0/v2_val - 1.0/u1_val - 1.0/u2_val
                                improved = True
                                found_22swap = True
                                break
                            if found_22swap:
                                break
                        if found_22swap:
                            break
                    if found_22swap:
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
            
        num_remove = random.choice([1, 1, 2, 2, 3])
        bits_sorted = sorted(bits, key=lambda i: idx_to_val[i], reverse=True)
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
    best_score = 0.0

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
        
    def randomized_greedy():
        m = 0
        available = list(range(2, 257))
        while available:
            compatible = [v for v in available if (conflict_mask[val_to_idx[v]] & m) == 0]
            if not compatible:
                break
            compatible.sort()
            k = min(5, len(compatible))
            v = random.choice(compatible[:k])
            idx = val_to_idx[v]
            m |= (1 << idx)
            available.remove(v)
        return m

    # 3. Another randomized greedy
    mask3 = randomized_greedy()
    
    initial_masks = [mask1, mask2, mask3]
    
    # Run LS on initial masks
    for m_init in initial_masks:
        if time.time() > deadline - 10.0:
            break
        m_ls, s_ls = local_search(m_init, time_limit=3.0)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls

    # Multiple restarts with perturbation
    while time.time() < deadline - 5.0:
        # Generate a new initial mask via randomized greedy
        m_init = randomized_greedy()
        
        # Local search
        remaining = deadline - time.time() - 2.0
        if remaining < 1.0:
            break
        m_ls, s_ls = local_search(m_init, time_limit=min(remaining, 2.0))
        
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
        else:
            # If not improved, try perturbing the best found so far
            if best_mask != 0:
                m_pert = perturb(best_mask)
                remaining = deadline - time.time() - 2.0
                if remaining < 1.0:
                    break
                m_ls2, s_ls2 = local_search(m_pert, time_limit=min(remaining, 2.0))
                if s_ls2 > best_score:
                    best_score = s_ls2
                    best_mask = m_ls2

    # Final polish
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
