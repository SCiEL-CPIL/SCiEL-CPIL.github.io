# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    # Try increasing n
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        # Precompute all quads and quints and their triple indices
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)], 
                                     triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]])
        
        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])

        # Precompute: for each triple index, which quads/quints contain it
        # quad_map[t] = list of (quad_idx, pos_in_quad)
        # quint_map[t] = list of (quint_idx, pos_in_quint)
        quad_map = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                quad_map[tidx].append((qi, pos))

        quint_map = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                quint_map[tidx].append((qi, pos))

        # Convert maps to numpy arrays for faster access if possible, or keep lists
        # Lists are fine for iteration in the inner loop if we are careful.
        # However, updating counts requires iterating over neighbors.
        
        found = False
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Initialization
            if restart == 0:
                color_flat = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                color_flat = np.array([0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                color_flat = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            elif restart == 3:
                # Random with bias towards red (since K4 is smaller than K5, maybe balance?)
                # Actually, random is usually fine.
                color_flat = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            else:
                # Perturb modular init
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_rate = 0.1 + 0.05 * (restart % 10)
                flip_mask = np.random.random(num_triples) < flip_rate
                color_flat = base.copy()
                color_flat[flip_mask] = 1 - color_flat[flip_mask]

            # Initial Violation Counting
            # Red K4: count of quads with 4 red triples
            # Blue K5: count of quints with 10 blue triples
            
            # We track:
            # rk4_count: number of red K4s
            # bk5_count: number of blue K5s
            # quad_red_counts: array of length len(all_quads), count of red triples in each quad
            # quint_blue_counts: array of length len(all_quints), count of blue triples in each quint
            
            quad_red_counts = np.zeros(len(all_quads), dtype=np.int8)
            for qt_idx in quad_triples_idx:
                # qt_idx is list of 4 triple indices
                # color_flat is 0 for red
                # count zeros
                quad_red_counts[qt_idx] = np.sum(color_flat[qt_idx] == 0)
            
            # Wait, qt_idx is a list of indices. We need to do this for all quads.
            # This loop is O(num_quads * 4).
            # Let's vectorize:
            # quad_triples_idx is a list of lists. 
            # We can reshape it if we know the size.
            quad_indices_np = np.array(quad_triples_idx, dtype=np.int32)
            # quad_indices_np shape: (num_quads, 4)
            # color_flat[quad_indices_np] shape: (num_quads, 4)
            quad_red_counts = np.sum(color_flat[quad_indices_np] == 0, axis=1)
            rk4_count = np.sum(quad_red_counts == 4)
            
            quint_indices_np = np.array(quint_triples_idx, dtype=np.int32)
            # quint_indices_np shape: (num_quints, 10)
            # color_flat[quint_indices_np] shape: (num_quints, 10)
            quint_blue_counts = np.sum(color_flat[quint_indices_np] == 1, axis=1)
            bk5_count = np.sum(quint_blue_counts == 10)

            if rk4_count == 0 and bk5_count == 0:
                found = True
                break

            # Simulated Annealing
            max_iters = 20000
            temp_start = 1.0
            temp_end = 0.01
            current_temp = temp_start
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                if rk4_count == 0 and bk5_count == 0:
                    found = True
                    break
                
                # Update temperature
                current_temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

                # Select triple to flip
                # Heuristic: Pick from a violating constraint if possible
                tidx = -1
                if rk4_count > 0 and bk5_count > 0:
                    if random.random() < 0.5:
                        # Pick a random red K4
                        # Find indices where quad_red_counts == 4
                        # This is expensive if done every step.
                        # Instead, just pick random triple and check? 
                        # Or maintain a list of violating quads?
                        # For now, random triple is safe but slow convergence.
                        # Let's try picking a random triple that is part of a "critical" set.
                        # Simple approach: Random triple.
                        tidx = random.randint(0, num_triples - 1)
                    else:
                        tidx = random.randint(0, num_triples - 1)
                elif rk4_count > 0:
                    # Try to pick a triple from a red K4
                    # Efficient way: 
                    # quad_red_counts == 4 gives violating quads.
                    # We can pick a random violating quad, then a random triple in it.
                    violating_quads = np.where(quad_red_counts == 4)[0]
                    if len(violating_quads) > 0:
                        qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                        tidx = quad_indices_np[qi][random.randint(0, 3)]
                    else:
                        tidx = random.randint(0, num_triples - 1)
                elif bk5_count > 0:
                    violating_quints = np.where(quint_blue_counts == 10)[0]
                    if len(violating_quints) > 0:
                        qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                        tidx = quint_indices_np[qi][random.randint(0, 9)]
                    else:
                        tidx = random.randint(0, num_triples - 1)
                else:
                    tidx = random.randint(0, num_triples - 1)

                old_color = color_flat[tidx]
                new_color = 1 - old_color

                # Calculate Delta
                # Red K4 change:
                # For each quad containing tidx:
                # If old_color == 0 (Red):
                #   If other 3 are Red (quad_red_counts[qi] == 4), then this quad was a K4. Removing tidx makes it 3. delta_rk4 -= 1.
                #   If other 3 are Red (quad_red_counts[qi] == 4 is false, but sum of others is 3), i.e. quad_red_counts[qi] == 4? No.
                #   Let's use the counts.
                #   If old_color == 0:
                #      count_red_others = quad_red_counts[qi] - 1
                #      if count_red_others == 3: # Was a K4
                #          delta_rk4 -= 1
                #      # Does it become a K4? No, we removed a red.
                #   If old_color == 1 (Blue):
                #      count_red_others = quad_red_counts[qi]
                #      if count_red_others == 3: # Becomes a K4
                #          delta_rk4 += 1

                delta_rk4 = 0
                for qi, pos in quad_map[tidx]:
                    c = quad_red_counts[qi]
                    if old_color == 0:
                        if c == 4:
                            delta_rk4 -= 1
                    else:
                        if c == 3:
                            delta_rk4 += 1

                # Blue K5 change:
                # If old_color == 1 (Blue):
                #   If other 9 are Blue (quint_blue_counts[qi] == 10), then this quint was a K5. Removing tidx makes it 9. delta_bk5 -= 1.
                # If old_color == 0 (Red):
                #   If other 9 are Blue (quint_blue_counts[qi] == 9), then this quint becomes a K5. delta_bk5 += 1.

                delta_bk5 = 0
                for qi, pos in quint_map[tidx]:
                    c = quint_blue_counts[qi]
                    if old_color == 1:
                        if c == 10:
                            delta_bk5 -= 1
                    else:
                        if c == 9:
                            delta_bk5 += 1

                old_total = rk4_count + bk5_count
                new_total = (rk4_count + delta_rk4) + (bk5_count + delta_bk5)
                delta = new_total - old_total

                # Acceptance
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    # Metropolis
                    prob = np.exp(-delta / max(current_temp, 1e-10))
                    if random.random() < prob:
                        accept = True

                if accept:
                    color_flat[tidx] = new_color
                    rk4_count += delta_rk4
                    bk5_count += delta_bk5
                    
                    # Update counts
                    for qi, pos in quad_map[tidx]:
                        if old_color == 0:
                            quad_red_counts[qi] -= 1
                        else:
                            quad_red_counts[qi] += 1
                            
                    for qi, pos in quint_map[tidx]:
                        if old_color == 1:
                            quint_blue_counts[qi] -= 1
                        else:
                            quint_blue_counts[qi] += 1

            if rk4_count == 0 and bk5_count == 0:
                found = True
                break

        if found:
            best_n = target_n
            best_colors = color_flat.tolist()

    # Final verification
    if best_n > 9:
        color_map = {}
        for i, (a, b, c) in enumerate(combinations(range(best_n), 3)):
            color_map[(a, b, c)] = best_colors[i]

        valid = True
        for quad in combinations(range(best_n), 4):
            a, b, c, d = quad
            if all(color_map[t] == 0 for t in [(a,b,c),(a,b,d),(a,c,d),(b,c,d)]):
                valid = False
                break
        if valid:
            for quint in combinations(range(best_n), 5):
                if all(color_map[t] == 1 for t in combinations(quint, 3)):
                    valid = False
                    break

        if not valid:
            best_n = 9
            best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
