import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    # Precompute for target n
    # We'll try to find a valid coloring for n=34 (the known bound)
    # If we can't, we fall back to smaller n

    def try_solve(target_n, time_budget):
        """Try to find a valid coloring for target_n within time_budget seconds."""
        if target_n <= 1:
            return True, [0] * (target_n * (target_n-1) * (target_n-2) // 6)
        
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        # For each quad (4-set), the 4 triples it contains
        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        # For each quint (5-set), the 10 triples it contains
        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, which quads/quints contain it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        # Convert to numpy arrays for faster access
        quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

        # Precompute "neighbors" for each triple: which other triples share a quad or quint
        # This helps in local search to know what changes when we flip a triple

        def compute_violations(color_arr):
            """Compute number of red K4 and blue K5 violations."""
            quad_colors = color_arr[quad_triples_np]  # shape (num_quads, 4)
            quad_red_counts = np.sum(quad_colors == 0, axis=1)
            rk4 = int(np.sum(quad_red_counts == 4))
            
            quint_colors = color_arr[quint_triples_np]  # shape (num_quints, 10)
            quint_blue_counts = np.sum(quint_colors == 1, axis=1)
            bk5 = int(np.sum(quint_blue_counts == 10))
            
            return rk4, bk5, quad_red_counts, quint_blue_counts

        def local_search(color_arr, time_limit):
            """Run local search to minimize violations."""
            rk4, bk5, quad_red_counts, quint_blue_counts = compute_violations(color_arr)
            total_viol = rk4 + bk5
            
            if total_viol == 0:
                return True, color_arr
            
            # Simulated annealing
            temp_start = 5.0
            temp_end = 0.1
            cooling = 0.995
            temp = temp_start
            min_temp = temp_end
            
            max_iters = 100000
            it = 0
            last_improve = 0
            best_viol = total_viol
            best_color = color_arr.copy()
            
            while it < max_iters:
                if time.time() > time_limit:
                    break
                
                # Choose a triple to flip
                # Bias toward violating structures
                tidx = -1
                r = random.random()
                
                if rk4 > 0 and r < 0.4:
                    # Find a violating red K4 and flip one of its triples
                    violating_quads = np.where(quad_red_counts == 4)[0]
                    if len(violating_quads) > 0:
                        qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                        qt = quad_triples_np[qi]
                        tidx = int(qt[random.randint(0, 3)])
                elif bk5 > 0 and r < 0.8:
                    # Find a violating blue K5 and flip one of its triples
                    violating_quints = np.where(quint_blue_counts == 10)[0]
                    if len(violating_quints) > 0:
                        qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                        qt = quint_triples_np[qi]
                        tidx = int(qt[random.randint(0, 9)])
                
                if tidx == -1:
                    tidx = random.randint(0, num_triples - 1)
                
                old_color = int(color_arr[tidx])
                new_color = 1 - old_color
                
                # Compute delta violations
                delta_rk4 = 0
                for (qi, pos) in triple_in_quads[tidx]:
                    qt = quad_triples_np[qi]
                    # Check if other 3 triples are red
                    others = [qt[p] for p in range(4) if p != pos]
                    others_red = all(int(color_arr[t]) == 0 for t in others)
                    if old_color == 0 and others_red:
                        delta_rk4 -= 1
                    elif old_color == 1 and others_red:
                        delta_rk4 += 1
                
                delta_bk5 = 0
                for (qi, pos) in triple_in_quints[tidx]:
                    qt = quint_triples_np[qi]
                    # Check if other 9 triples are blue
                    others = [qt[p] for p in range(10) if p != pos]
                    others_blue = all(int(color_arr[t]) == 1 for t in others)
                    if old_color == 1 and others_blue:
                        delta_bk5 -= 1
                    elif old_color == 0 and others_blue:
                        delta_bk5 += 1
                
                delta_total = delta_rk4 + delta_bk5
                
                # Acceptance criterion
                if delta_total < 0:
                    accept = True
                elif delta_total == 0:
                    accept = random.random() < 0.3
                else:
                    accept = random.random() < np.exp(-delta_total / temp)
                
                if accept:
                    color_arr[tidx] = new_color
                    
                    # Update counts incrementally
                    for (qi, pos) in triple_in_quads[tidx]:
                        if new_color == 0:
                            quad_red_counts[qi] += 1
                        else:
                            quad_red_counts[qi] -= 1
                    
                    for (qi, pos) in triple_in_quints[tidx]:
                        if new_color == 1:
                            quint_blue_counts[qi] += 1
                        else:
                            quint_blue_counts[qi] -= 1
                    
                    rk4 += delta_rk4
                    bk5 += delta_bk5
                    total_viol = rk4 + bk5
                    
                    if total_viol < best_viol:
                        best_viol = total_viol
                        best_color = color_arr.copy()
                        last_improve = it
                    
                    if total_viol == 0:
                        return True, color_arr
                
                temp *= cooling
                if temp < min_temp:
                    temp = min_temp
                
                it += 1
                
                # Restart if stuck
                if it - last_improve > 10000:
                    color_arr = best_color.copy()
                    rk4, bk5, quad_red_counts, quint_blue_counts = compute_violations(color_arr)
                    total_viol = rk4 + bk5
                    best_viol = total_viol
                    last_improve = it
                    temp = max(temp, 2.0)
            
            # Check if best is valid
            if best_viol == 0:
                return True, best_color
            return False, best_color

        # Try multiple initializations
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Different initialization strategies
            if restart == 0:
                color_arr = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 1:
                color_arr = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 2:
                color_arr = np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            elif restart == 3:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            elif restart == 4:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.4] = 0
            elif restart == 5:
                color_arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
                color_arr[np.random.random(num_triples) < 0.4] = 1
            elif restart % 2 == 0:
                base = np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.2
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]
            else:
                base = np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
                flip_mask = np.random.random(num_triples) < 0.2
                color_arr = base.copy()
                color_arr[flip_mask] = 1 - color_arr[flip_mask]
            
            found, result = local_search(color_arr, time_budget)
            if found:
                return True, result.tolist()
        
        return False, None

    # Try to solve for n=34
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        found, colors = try_solve(34, remaining_time - 5)
        if found:
            return {'n': 34, 'colors': colors}
    
    # If we couldn't find n=34, try smaller n
    # Try n=33, 32, ... down to 10
    for n in range(33, 9, -1):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        found, colors = try_solve(n, remaining_time - 3)
        if found:
            return {'n': n, 'colors': colors}
    
    # Fallback
    return {'n': best_n, 'colors': best_colors}
