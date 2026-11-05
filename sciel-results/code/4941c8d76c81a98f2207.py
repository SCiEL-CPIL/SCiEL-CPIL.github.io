# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes
    
    # Precompute weights as Fractions
    weights = [Fraction(1, v) for v in nodes]
    weight_by_val = {v: Fraction(1, v) for v in nodes}
    
    # Precompute conflict masks using bitsets
    # conflict_mask[i] has bit j set if nodes[i] and nodes[j] conflict (lcm <= 256)
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
    
    def get_score_from_mask(mask):
        score = Fraction(0)
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += weights[idx]
            m ^= lsb
        return score
    
    def local_search(initial_mask, time_limit=3.0):
        current_mask = initial_mask
        current_score = get_score_from_mask(current_mask)
        improved = True
        start_ls_time = time.time()
        
        while improved:
            improved = False
            
            if time.time() > deadline or time.time() - start_ls_time > time_limit:
                break
            
            # Build in_mask and not_in_mask lists
            in_mask = []
            not_in_mask = []
            for i in range(num_nodes):
                if current_mask & (1 << i):
                    in_mask.append(i)
                else:
                    not_in_mask.append(i)
            
            # Phase 1: Add any compatible node (prefer smaller values = higher weight)
            not_in_mask_sorted = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            for v_idx in not_in_mask_sorted:
                if (conflict_mask[v_idx] & current_mask) == 0:
                    current_mask |= (1 << v_idx)
                    current_score += weights[v_idx]
                    improved = True
                    in_mask.append(v_idx)
                    not_in_mask.remove(v_idx)
            
            # Phase 2: 1-for-1 swap (remove larger, add smaller)
            in_mask_desc = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask_asc = sorted(not_in_mask, key=lambda i: idx_to_val[i])
            
            found_swap = False
            for u_idx in in_mask_desc:
                if time.time() > deadline:
                    break
                u_val = idx_to_val[u_idx]
                mask_without_u = current_mask & ~(1 << u_idx)
                
                for v_idx in not_in_mask_asc:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        break
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        gain = weights[v_idx] - weights[u_idx]
                        if gain > 0:
                            current_mask = mask_without_u | (1 << v_idx)
                            current_score += gain
                            improved = True
                            found_swap = True
                            # Update lists
                            in_mask.remove(u_idx)
                            in_mask.append(v_idx)
                            not_in_mask.remove(v_idx)
                            break
                
                if found_swap:
                    break
            
            # Phase 3: 2-for-1 swap (remove two larger, add one smaller)
            if not found_swap:
                in_mask_desc2 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
                not_in_mask_asc2 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
                found_2swap = False
                
                for i1 in range(len(in_mask_desc2)):
                    if time.time() > deadline:
                        break
                    u1_idx = in_mask_desc2[i1]
                    u1_val = idx_to_val[u1_idx]
                    for i2 in range(i1 + 1, len(in_mask_desc2)):
                        u2_idx = in_mask_desc2[i2]
                        u2_val = idx_to_val[u2_idx]
                        combined_weight = weights[u1_idx] + weights[u2_idx]
                        mask_without_both = current_mask & ~(1 << u1_idx) & ~(1 << u2_idx)
                        
                        for v_idx in not_in_mask_asc2:
                            v_val = idx_to_val[v_idx]
                            if weights[v_idx] <= combined_weight:
                                break
                            if (conflict_mask[v_idx] & mask_without_both) == 0:
                                current_mask = mask_without_both | (1 << v_idx)
                                current_score += weights[v_idx] - combined_weight
                                improved = True
                                found_2swap = True
                                in_mask.remove(u1_idx)
                                in_mask.remove(u2_idx)
                                in_mask.append(v_idx)
                                not_in_mask.remove(v_idx)
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
            
            # Phase 4: 1-for-2 swap (remove one, add two smaller)
            if not found_swap and not found_2swap:
                in_mask_desc3 = sorted(in_mask, key=lambda i: idx_to_val[i], reverse=True)
                not_in_mask_asc3 = sorted(not_in_mask, key=lambda i: idx_to_val[i])
                found_12swap = False
                
                for u_idx in in_mask_desc3:
                    if time.time() > deadline:
                        break
                    u_val = idx_to_val[u_idx]
                    mask_without_u = current_mask & ~(1 << u_idx)
                    
                    for v1_idx in not_in_mask_asc3:
                        v1_val = idx_to_val[v1_idx]
                        if (conflict_mask[v1_idx] & mask_without_u) != 0:
                            continue
                        mask_with_v1 = mask_without_u | (1 << v1_idx)
                        
                        for v2_idx in not_in_mask_asc3:
                            if v2_idx == v1_idx:
                                continue
                            v2_val = idx_to_val[v2_idx]
                            if weights[v1_idx] + weights[v2_idx] <= weights[u_idx]:
                                break
                            if (conflict_mask[v2_idx] & mask_with_v1) != 0:
                                continue
                            current_mask = mask_with_v1 | (1 << v2_idx)
                            current_score += weights[v1_idx] + weights[v2_idx] - weights[u_idx]
                            improved = True
                            found_12swap = True
                            in_mask.remove(u_idx)
                            in_mask.append(v1_idx)
                            in_mask.append(v2_idx)
                            not_in_mask.remove(v1_idx)
                            not_in_mask.remove(v2_idx)
                            break
                        if found_12swap:
                            break
                    if found_12swap:
                        break
            
            # Rebuild lists if changed
            if improved:
                in_mask = []
                not_in_mask = []
                for i in range(num_nodes):
                    if current_mask & (1 << i):
                        in_mask.append(i)
                    else:
                        not_in_mask.append(i)
        
        return current_mask, current_score
    
    def perturb(mask, num_remove=2):
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
        
        # Remove random elements, biased toward larger values
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
    best_score = Fraction(0)
    
    # Strategy 1: Start with all numbers > 128 (they're pairwise compatible)
    mask_large = 0
    for v in range(129, 257):
        idx = val_to_idx[v]
        mask_large |= (1 << idx)
    
    # Strategy 2: Greedy by weight
    mask_greedy = 0
    for v in range(2, 257):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask_greedy) == 0:
            mask_greedy |= (1 << idx)
    
    # Strategy 3: Randomized greedy with different biases
    def randomized_greedy(bias='small'):
        m = 0
        available = list(range(2, 257))
        random.shuffle(available)
        
        while available:
            compatible = [v for v in available if (conflict_mask[val_to_idx[v]] & m) == 0]
            if not compatible:
                break
            
            if bias == 'small':
                compatible.sort()
                k = min(3, len(compatible))
                v = random.choice(compatible[:k])
            elif bias == 'medium':
                compatible.sort()
                mid = len(compatible) // 2
                k = min(5, len(compatible))
                v = random.choice(compatible[mid:mid+k] if mid + k <= len(compatible) else compatible[mid:])
            else:
                compatible.sort(reverse=True)
                k = min(5, len(compatible))
                v = random.choice(compatible[:k])
            
            idx = val_to_idx[v]
            m |= (1 << idx)
            available.remove(v)
        return m
    
    initial_masks = [mask_large, mask_greedy]
    
    # Add randomized starts
    for _ in range(5):
        initial_masks.append(randomized_greedy('small'))
        initial_masks.append(randomized_greedy('medium'))
    
    # Local search on all initial masks
    for i, init_mask in enumerate(initial_masks):
        if time.time() > deadline - 10.0:
            break
        m_ls, s_ls = local_search(init_mask, time_limit=3.0)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
    
    # Iterative improvement with perturbation
    if time.time() < deadline - 5.0:
        for polish_iter in range(20):
            if time.time() > deadline - 3.0:
                break
            
            # Try different perturbation strengths
            num_remove = random.choice([1, 2, 2, 3])
            m_pert = perturb(best_mask, num_remove)
            
            remaining = deadline - time.time() - 2.0
            if remaining < 1.0:
                break
            
            m_pol, s_pol = local_search(m_pert, time_limit=min(remaining, 2.5))
            if s_pol > best_score:
                best_score = s_pol
                best_mask = m_pol
            else:
                # Direct polish
                remaining = deadline - time.time() - 2.0
                if remaining < 1.0:
                    break
                m_pol2, s_pol2 = local_search(best_mask, time_limit=min(remaining, 2.0))
                if s_pol2 > best_score:
                    best_score = s_pol2
                    best_mask = m_pol2
                else:
                    # Try a fresh randomized start
                    if time.time() < deadline - 5.0:
                        m_fresh = randomized_greedy('small')
                        m_fresh_ls, s_fresh = local_search(m_fresh, time_limit=2.0)
                        if s_fresh > best_score:
                            best_score = s_fresh
                            best_mask = m_fresh_ls
    
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
