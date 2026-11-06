import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 118.0
    
    # Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(0, max_a):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Build inverse mapping: for each index i, which APs does it participate in?
    # And what is its position (0, 1, 2, 3) in those APs?
    # This allows O(1) delta calculation for a single flip.
    ap_indices_per_pos = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        a0, a1, a2, a3 = ap_array[ap_idx]
        ap_indices_per_pos[a0].append((ap_idx, 0))
        ap_indices_per_pos[a1].append((ap_idx, 1))
        ap_indices_per_pos[a2].append((ap_idx, 2))
        ap_indices_per_pos[a3].append((ap_idx, 3))

    # Precompute for each AP, the indices of the other 3 positions for each position
    # other_indices[ap_idx][pos] = (idx0, idx1, idx2)
    other_indices = np.zeros((num_aps, 4, 3), dtype=np.int32)
    for ap_idx in range(num_aps):
        a0, a1, a2, a3 = ap_array[ap_idx]
        other_indices[ap_idx, 0] = [a1, a2, a3]
        other_indices[ap_idx, 1] = [a0, a2, a3]
        other_indices[ap_idx, 2] = [a0, a1, a3]
        other_indices[ap_idx, 3] = [a0, a1, a2]
        
    # Helper to compute initial mono status
    def compute_is_mono(color_vec):
        c0 = color_vec[ap_array[:, 0]]
        c1 = color_vec[ap_array[:, 1]]
        c2 = color_vec[ap_array[:, 2]]
        c3 = color_vec[ap_array[:, 3]]
        return (c0 == c1) & (c1 == c2) & (c2 == c3)
        
    def local_search_sat(color_vec, is_mono, current_score, deadline):
        """
        Vectorized, weighted-clause local search with dynamic temperature and noise.
        """
        improved = True
        temperature = 1.0
        base_temp = 1.0
        min_temp = 0.1
        cooling_rate = 0.995
        
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Shuffle order of indices to avoid bias
            indices = np.random.permutation(n)
            
            for idx in indices:
                # Calculate delta if we flip color_vec[idx]
                delta = 0
                for ap_idx, pos in ap_indices_per_pos[idx]:
                    if is_mono[ap_idx]:
                        # It is currently mono. Flipping idx breaks it.
                        # In Max-SAT terms, we satisfy this clause (reduce violations).
                        # Wait, the task is to MINIMIZE monochromatic APs.
                        # So "violated clause" = non-mono AP? No.
                        # Let's stick to the objective: Minimize count of Mono APs.
                        # So we want to flip if delta < 0.
                        delta -= 1
                    else:
                        # It is currently non-mono. 
                        # Does flipping idx make it mono?
                        # If so, we increase the count (bad).
                        oidxs = other_indices[ap_idx, pos]
                        c0 = color_vec[oidxs[0]]
                        c1 = color_vec[oidxs[1]]
                        c2 = color_vec[oidxs[2]]
                        
                        if c0 == c1 and c1 == c2:
                            if c0 != color_vec[idx]:
                                delta += 1
                
                # Dynamic temperature and noise
                # Accept if delta < 0 (improvement)
                # Or if delta >= 0 and random() < exp(-delta / temperature) (simulated annealing style)
                # But we want to be aggressive. Let's add noise to delta.
                # Actually, simple SA is good for escaping local minima.
                
                accept = False
                if delta < 0:
                    accept = True
                else:
                    # Metropolis criterion
                    if temperature > min_temp:
                        # delta is positive (worse) or 0
                        # We want to accept with probability exp(-delta/temperature)
                        # To avoid overflow, use log
                        if delta == 0:
                            accept = True # Neutral, accept to move around
                        else:
                            # Use random log
                            log_rand = np.log(np.random.rand() + 1e-9)
                            if log_rand > -delta / temperature:
                                accept = True
                            
                if accept:
                    # Apply flip
                    color_vec[idx] = 1 - color_vec[idx]
                    current_score += delta
                    
                    # Update is_mono for affected APs
                    for ap_idx, pos in ap_indices_per_pos[idx]:
                        a0, a1, a2, a3 = ap_array[ap_idx]
                        c0 = color_vec[a0]
                        c1 = color_vec[a1]
                        c2 = color_vec[a2]
                        c3 = color_vec[a3]
                        is_mono[ap_idx] = (c0 == c1) and (c1 == c2) and (c2 == c3)
                    
                    improved = True
                    if current_score == 0:
                        return color_vec, is_mono, current_score
                        
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp

        return color_vec, is_mono, current_score

    best_score = float('inf')
    best_color = None
    
    # Multi-start strategy
    while time.time() < deadline - 1.0:
        # Choose initialization
        init_type = random.randint(0, 3)
        
        if init_type == 0:
            # Thue-Morse
            color = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
        elif init_type == 1:
            # Random
            color = np.random.randint(0, 2, n, dtype=np.int8)
        elif init_type == 2:
            # Random with bias towards 0 (or just random, but let's try structured random)
            # Actually, random is fine. Let's try "blocky" random
            color = np.zeros(n, dtype=np.int8)
            # Randomly flip some blocks
            for _ in range(10):
                start = random.randint(0, n-1)
                length = random.randint(1, 10)
                color[start:start+length] = 1
        else:
            # Alternating blocks of 2 (seed strategy)
            color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
            
        is_mono = compute_is_mono(color)
        current_score = np.sum(is_mono)
        
        # Local Search
        color, is_mono, current_score = local_search_sat(color, is_mono, current_score, deadline)
        
        if current_score < best_score:
            best_score = current_score
            best_color = color.copy()
            
        # If we found 0, we can stop immediately
        if best_score == 0:
            break
            
        # Perturbation for next iteration if we don't break
        # We just loop back to a new random init. 
        # However, we might get stuck in the same local minima if inits are similar.
        # The random init ensures diversity. 
        # To make it "Iterated", we can perturb the best solution found so far
        # and run local search again.
        if best_color is not None and time.time() < deadline - 2.0:
            # Perturb best_color
            perturbed = best_color.copy()
            # Flip 5-10 random bits
            num_flips = random.randint(5, 10)
            flip_indices = np.random.choice(n, num_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            
            is_mono_p = compute_is_mono(perturbed)
            score_p = np.sum(is_mono_p)
            
            perturbed, is_mono_p, score_p = local_search_sat(perturbed, is_mono_p, score_p, deadline)
            
            if score_p < best_score:
                best_score = score_p
                best_color = perturbed.copy()
                
                if best_score == 0:
                    break

    if best_color is None:
        # Fallback
        best_color = np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8)
        
    return best_color.tolist()
