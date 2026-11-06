import random
import math

def construct():
    N = 256
    
    # Precompute all 4-term APs
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    ap_id = 0
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            aps.append((i0, i1, i2, i3))
            index_to_aps[i0].append(ap_id)
            index_to_aps[i1].append(ap_id)
            index_to_aps[i2].append(ap_id)
            index_to_aps[i3].append(ap_id)
            ap_id += 1
            
    total_aps = len(aps)
    
    def count_mono(colors):
        count = 0
        for ap in aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    def delta_flip_single(colors, idx):
        """
        Calculate change in monochromatic AP count if bit at idx is flipped.
        """
        affected_aps = index_to_aps[idx]
        if not affected_aps:
            return 0
            
        delta = 0
        c_idx = colors[idx]
        
        for ap_id in affected_aps:
            ap = aps[ap_id]
            
            # Check if the AP is currently monochromatic
            # and if flipping idx breaks it, or if it was not mono and flipping makes it mono.
            
            # Current state
            is_mono_before = True
            for j in range(4):
                if colors[ap[j]] != colors[ap[0]]:
                    is_mono_before = False
                    break
            
            # New state (conceptually flipping idx)
            # Instead of creating a new array, we check logic:
            # If idx is in ap, its color changes.
            
            # Let's just compute the new mono status efficiently.
            # If idx is not in ap, delta is 0 for this ap (but idx is in ap by construction of index_to_aps).
            
            # Count colors in the AP after flip
            # This is slightly expensive if done naively for every affected AP.
            # Optimization:
            # Before flip:
            #   If mono, count is 1 (if all same) or 0.
            #   We know is_mono_before.
            
            # After flip:
            #   The color at idx changes from c_idx to 1-c_idx.
            #   Let's check if the AP becomes mono.
            #   An AP is mono if all 4 are 0 or all 4 are 1.
            
            # Case 1: Before flip, it was mono.
            if is_mono_before:
                # It was mono with color c_idx (since idx is in ap, and all were same).
                # After flip, idx has color 1-c_idx.
                # The other 3 have color c_idx.
                # So it is NOT mono anymore (1 vs 3).
                delta -= 1
            else:
                # It was not mono before.
                # After flip, can it become mono?
                # It becomes mono if all 4 become c_idx (impossible since idx becomes 1-c_idx) 
                # OR all 4 become 1-c_idx.
                # For all 4 to become 1-c_idx, the other 3 must already be 1-c_idx.
                
                other_indices = [ap[j] for j in range(4) if ap[j] != idx]
                # Check if all other_indices have color 1-c_idx
                is_other_all_1_minus_c = True
                for o_idx in other_indices:
                    if colors[o_idx] != (1 - c_idx):
                        is_other_all_1_minus_c = False
                        break
                
                if is_other_all_1_minus_c:
                    # Then after flip, idx is 1-c_idx, others are 1-c_idx.
                    # So it becomes mono.
                    delta += 1
                
        return delta

    best_colors = None
    best_count = float('inf')
    
    num_restarts = 30
    max_il_iter = 10
    perturb_size = 5
    
    base_seed = 42
    
    for restart in range(num_restarts):
        random.seed(base_seed + restart)
        
        # Initialization: Random
        colors = [random.randint(0, 1) for _ in range(N)]
        current_count = count_mono(colors)
        
        for il_iter in range(max_il_iter):
            # Local Search: Single Flip Hill Climbing
            improved = True
            while improved:
                improved = False
                # Random order to avoid bias
                indices = list(range(N))
                random.shuffle(indices)
                for idx in indices:
                    delta = delta_flip_single(colors, idx)
                    if delta < 0:
                        colors[idx] = 1 - colors[idx]
                        current_count += delta
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_colors = colors[:]
            
            # If we found a new best, we might want to restart IL or just continue?
            # Usually IL continues.
            
            # Perturbation
            for _ in range(perturb_size):
                idx = random.randint(0, N - 1)
                colors[idx] = 1 - colors[idx]
                
            # Update current count after perturbation
            # This is expensive if we recount everything. 
            # Better to apply delta or just recount if perturb_size is small.
            # Recounting is O(NumAPs) ~ 10k. Perturb 5 times.
            # It's safer to just recount for correctness in IL loop logic.
            current_count = count_mono(colors)
            
            # Check if perturbation improved things (unlikely but possible)
            if current_count < best_count:
                best_count = current_count
                best_colors = colors[:]

    return best_colors
