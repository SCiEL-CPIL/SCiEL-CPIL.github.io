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

    # Identify core vs periphery: core = nodes with high conflict degree
    conflict_degree = [bin(conflict_mask[i]).count('1') for i in range(num_nodes)]
    
    # Core: nodes with conflict degree > threshold
    # Periphery: nodes with low conflict degree (can be solved exactly)
    CORE_THRESHOLD = 10
    core_indices = set(i for i in range(num_nodes) if conflict_degree[i] > CORE_THRESHOLD)
    periphery_indices = list(i for i in range(num_nodes) if i not in core_indices)
    
    # For exact periphery solving: given a set of active core nodes,
    # find optimal subset of periphery that's compatible with core and internally compatible
    # Use bitmask DP for small periphery components
    
    def solve_periphery_exact(core_mask, periphery_list):
        """
        Given core_mask (bitmask of active core nodes), find optimal subset of periphery_list
        that is compatible with all active core nodes and internally compatible.
        Uses bitmask DP for the periphery nodes.
        """
        if not periphery_list:
            return 0, 0.0
        
        # Filter periphery nodes that are compatible with the current core
        compatible_periph = []
        for p_idx in periphery_list:
            if (conflict_mask[p_idx] & core_mask) == 0:
                compatible_periph.append(p_idx)
        
        if not compatible_periph:
            return 0, 0.0
        
        n_periph = len(compatible_periph)
        
        if n_periph <= 20:
            # Exact bitmask DP
            # dp[mask] = max reciprocal sum using subset 'mask' of compatible_periph
            # that is internally compatible
            size = 1 << n_periph
            dp = [-1.0] * size
            dp[0] = 0.0
            best_mask_val = 0
            best_score_val = 0.0
            
            for mask in range(size):
                if dp[mask] < 0:
                    continue
                # Try adding each node not in mask
                for i in range(n_periph):
                    if mask & (1 << i):
                        continue
                    new_mask = mask | (1 << i)
                    # Check if compatible_periph[i] is compatible with all nodes in mask
                    p_idx = compatible_periph[i]
                    # Check conflicts with nodes already in mask
                    conflict_with_mask = 0
                    for j in range(n_periph):
                        if mask & (1 << j):
                            q_idx = compatible_periph[j]
                            if conflict_mask[p_idx] & (1 << q_idx):
                                conflict_with_mask = 1
                                break
                    if conflict_with_mask:
                        continue
                    new_score = dp[mask] + 1.0 / idx_to_val[p_idx]
                    if new_score > dp[new_mask]:
                        dp[new_mask] = new_score
                        if new_score > best_score_val:
                            best_score_val = new_score
                            best_mask_val = new_mask
            
            return best_mask_val, best_score_val
        else:
            # For larger periphery, use greedy with local improvement
            best_mask = 0
            best_score = 0.0
            # Greedy: add nodes in order of decreasing weight (smallest value first)
            sorted_periph = sorted(compatible_periph, key=lambda i: idx_to_val[i])
            for p_idx in sorted_periph:
                if (conflict_mask[p_idx] & best_mask) == 0:
                    best_mask |= (1 << p_idx)
                    best_score += 1.0 / idx_to_val[p_idx]
            return best_mask, best_score

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
                        break
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

    def polish_with_exact_periphery(mask):
        """
        Given a solution mask, try to improve by:
        1. Keeping core nodes fixed
        2. Solving periphery exactly given the core
        """
        if time.time() > deadline - 1.0:
            return mask, get_score(mask)
        
        # Separate core and periphery in the current mask
        core_in_mask = mask & 0  # We'll build this
        periphery_in_mask = mask & 0
        
        core_bits = 0
        periph_bits = 0
        for i in range(num_nodes):
            if mask & (1 << i):
                if i in core_indices:
                    core_bits |= (1 << i)
                else:
                    periph_bits |= (1 << i)
        
        # Try different core configurations by removing/adding core nodes
        # First: solve periphery exactly given current core
        periph_list = [i for i in periphery_indices if not (mask & (1 << i))]
        # Actually, we want to consider ALL periphery nodes, not just those not in mask
        all_periph = list(periphery_indices)
        
        # Solve periphery exactly given the core
        new_periph_mask, new_periph_score = solve_periphery_exact(core_bits, all_periph)
        
        # new_periph_mask is a bitmask over the periphery_indices list
        # Convert back to global mask
        global_periph_mask = 0
        for j, p_idx in enumerate(all_periph):
            if new_periph_mask & (1 << j):
                global_periph_mask |= (1 << p_idx)
        
        new_mask = core_bits | global_periph_mask
        new_score = get_score(new_mask)
        
        if new_score > get_score(mask):
            return new_mask, new_score
        
        # Try perturbing core: remove one core node and re-solve
        best_mask = mask
        best_score = new_score
        
        core_nodes_in = [i for i in core_indices if mask & (1 << i)]
        for c_idx in core_nodes_in:
            if time.time() > deadline - 1.0:
                break
            perturbed_core = core_bits & ~(1 << c_idx)
            p_mask, p_score = solve_periphery_exact(perturbed_core, all_periph)
            gp_mask = 0
            for j, pi in enumerate(all_periph):
                if p_mask & (1 << j):
                    gp_mask |= (1 << pi)
            candidate = perturbed_core | gp_mask
            cscore = get_score(candidate)
            if cscore > best_score:
                best_score = cscore
                best_mask = candidate
        
        return best_mask, best_score

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
        if time.time() > deadline - 15.0:
            break
        m_ls, s_ls = local_search(m_init, time_limit=3.0)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls

    # Multiple restarts with perturbation
    while time.time() < deadline - 10.0:
        # Generate a new initial mask via randomized greedy
        m_init = randomized_greedy()
        
        # Local search
        remaining = deadline - time.time() - 5.0
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
                remaining = deadline - time.time() - 5.0
                if remaining < 1.0:
                    break
                m_ls2, s_ls2 = local_search(m_pert, time_limit=min(remaining, 2.0))
                if s_ls2 > best_score:
                    best_score = s_ls2
                    best_mask = m_ls2

    # Final polish with exact periphery solving
    if time.time() < deadline - 2.0:
        m_final, s_final = polish_with_exact_periphery(best_mask)
        if s_final > best_score:
            best_score = s_final
            best_mask = m_final
    
    # One more local search after polish
    if time.time() < deadline - 1.0:
        m_final2, s_final2 = local_search(best_mask, time_limit=deadline - time.time() - 0.5)
        if s_final2 > best_score:
            best_score = s_final2
            best_mask = m_final2

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
