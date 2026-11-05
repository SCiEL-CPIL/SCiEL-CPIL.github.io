import numpy as np
import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def build_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        k4s_containing = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            idx_tuple = (t1, t2, t3, t4)
            k4s.append(idx_tuple)
            for t in idx_tuple:
                k4s_containing[t].append(len(k4s) - 1)
                
        k5s = []
        k5s_containing = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            idx_tuple = tuple(idxs)
            k5s.append(idx_tuple)
            for t in idx_tuple:
                k5s_containing[t].append(len(k5s) - 1)
        
        # Pre-build numpy arrays for fast cost calculation
        k4_array = np.array(k4s, dtype=np.int32) if k4s else None
        k5_array = np.array(k5s, dtype=np.int32) if k5s else None
        
        # Pre-build containing arrays as numpy for faster access
        k4s_containing_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in k4s_containing]
        k5s_containing_np = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in k5s_containing]
        
        return triples, num_triples, k4s, k5s, k4s_containing, k5s_containing, k4_array, k5_array, k4s_containing_np, k5s_containing_np

    def calculate_cost_vectorized(colors, k4_array, k5_array):
        cost = 0
        if k4_array is not None and len(k4_array) > 0:
            k4_colors = colors[k4_array]
            red_k4s = np.all(k4_colors == 0, axis=1)
            cost += int(np.sum(red_k4s))
        if k5_array is not None and len(k5_array) > 0:
            k5_colors = colors[k5_array]
            blue_k5s = np.all(k5_colors == 1, axis=1)
            cost += int(np.sum(blue_k5s))
        return cost

    def local_search_repair(colors, k4s, k5s, k4s_containing, k5s_containing, k4_array, k5_array, k4s_containing_np, k5s_containing_np, num_triples, time_limit):
        """Targeted local search: find violations and try to fix them by flipping involved triples."""
        current_cost = calculate_cost_vectorized(colors, k4_array, k5_array)
        if current_cost == 0:
            return colors, True
        
        best_colors = colors.copy()
        best_cost = current_cost
        max_iters = 50000
        iter_count = 0
        
        while iter_count < max_iters and time.time() < time_limit:
            # Find a violating K4 or K5
            if k4_array is not None and len(k4_array) > 0:
                k4_colors = colors[k4_array]
                red_k4_mask = np.all(k4_colors == 0, axis=1)
            else:
                red_k4_mask = np.array([], dtype=bool)
            
            if k5_array is not None and len(k5_array) > 0:
                k5_colors = colors[k5_array]
                blue_k5_mask = np.all(k5_colors == 1, axis=1)
            else:
                blue_k5_mask = np.array([], dtype=bool)
            
            if not np.any(red_k4_mask) and not np.any(blue_k5_mask):
                current_cost = 0
                break
            
            # Pick a random violation to fix
            if np.any(red_k4_mask) and (not np.any(blue_k5_mask) or np.random.rand() < 0.5):
                violating_k4 = np.random.choice(np.where(red_k4_mask)[0])
                idxs = k4s[violating_k4]
                # Try flipping one of the 4 triples to blue
                for t in idxs:
                    delta = 0
                    # K4s containing t that are currently all red: flipping t to 1 removes them
                    for k4_idx in k4s_containing_np[t]:
                        idxs_k4 = k4s[k4_idx]
                        if all(colors[i] == 0 for i in idxs_k4):
                            delta -= 1  # removing a red K4
                    # K5s containing t where all other 9 are 1: flipping t to 1 creates a blue K5
                    for k5_idx in k5s_containing_np[t]:
                        idxs_k5 = k5s[k5_idx]
                        others_all_blue = True
                        for i in idxs_k5:
                            if i != t:
                                if colors[i] != 1:
                                    others_all_blue = False
                                    break
                        if others_all_blue:
                            delta += 1  # creating a blue K5
                    
                    new_cost = current_cost + delta
                    if new_cost < current_cost:
                        colors[t] = 1
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()
                        iter_count += 1
                        break
                else:
                    iter_count += 1
                    continue
            elif np.any(blue_k5_mask):
                violating_k5 = np.random.choice(np.where(blue_k5_mask)[0])
                idxs = k5s[violating_k5]
                # Try flipping one of the 10 triples to red
                for t in idxs:
                    delta = 0
                    for k4_idx in k4s_containing_np[t]:
                        idxs_k4 = k4s[k4_idx]
                        others_all_red = True
                        for i in idxs_k4:
                            if i != t:
                                if colors[i] != 0:
                                    others_all_red = False
                                    break
                        if others_all_red:
                            delta += 1
                    for k5_idx in k5s_containing_np[t]:
                        idxs_k5 = k5s[k5_idx]
                        if all(colors[i] == 1 for i in idxs_k5):
                            delta -= 1
                    
                    new_cost = current_cost + delta
                    if new_cost < current_cost:
                        colors[t] = 0
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()
                        iter_count += 1
                        break
                else:
                    iter_count += 1
                    continue
            
            if best_cost == 0:
                break
        
        return best_colors, (best_cost == 0)

    def simulate_annealing(n, time_limit, init_colors=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        triples, num_triples, k4s, k5s, k4s_containing, k5s_containing, k4_array, k5_array, k4s_containing_np, k5s_containing_np = build_indices(n)
        
        if init_colors is None:
            best_init = None
            best_init_cost = float('inf')
            
            patterns = [
                lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a+b+c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a**2+b**2+c**2) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b*c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a+b) % 2 == c % 2 else 1,
                lambda a,b,c: 0 if (a+b+c) % 4 < 2 else 1,
                lambda a,b,c: 0 if (a*b + b*c + c*a) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a+b+c) % 5 < 3 else 1,
                lambda a,b,c: 0 if (a*b + b*c + c*a) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a + 2*b + 3*c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a*b*c + a + b + c) % 2 == 0 else 1,
            ]
            
            for pat in patterns:
                c = np.array([pat(a,b,c) for a,b,c in triples], dtype=np.int8)
                cost = calculate_cost_vectorized(c, k4_array, k5_array)
                if cost < best_init_cost:
                    best_init_cost = cost
                    best_init = c
            if best_init is None:
                best_init = np.random.randint(0, 2, num_triples, dtype=np.int8)
                best_init_cost = calculate_cost_vectorized(best_init, k4_array, k5_array)
            colors = best_init.copy()
        else:
            colors = init_colors.copy()
        
        current_cost = calculate_cost_vectorized(colors, k4_array, k5_array)
        best_cost = current_cost
        best_colors = colors.copy()
        
        temp = 1.0
        cool_rate = 0.9997
        min_temp = 0.01
        last_improve_time = time.time()
        no_improve_steps = 0
        max_no_improve = 8000
        
        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_cost = 0
            
            for k4_idx in k4s_containing_np[t]:
                idxs = k4s[k4_idx]
                is_red_before = all(colors[i] == 0 for i in idxs)
                if is_red_before:
                    if new_color != 0:
                        delta_cost -= 1
                else:
                    if new_color == 0:
                        others_are_red = True
                        for i in idxs:
                            if i != t:
                                if colors[i] != 0:
                                    others_are_red = False
                                    break
                        if others_are_red:
                            delta_cost += 1
            
            for k5_idx in k5s_containing_np[t]:
                idxs = k5s[k5_idx]
                is_blue_before = all(colors[i] == 1 for i in idxs)
                if is_blue_before:
                    if new_color != 1:
                        delta_cost -= 1
                else:
                    if new_color == 1:
                        others_are_blue = True
                        for i in idxs:
                            if i != t:
                                if colors[i] != 1:
                                    others_are_blue = False
                                    break
                        if others_are_blue:
                            delta_cost += 1
            
            new_cost = current_cost + delta_cost
            
            if new_cost < current_cost:
                accept = True
            elif new_cost == current_cost:
                accept = np.random.rand() < 0.3
            else:
                if temp > min_temp:
                    accept = np.random.rand() < np.exp(-new_cost / temp)
                else:
                    accept = False
            
            if accept:
                colors[t] = new_color
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    last_improve_time = time.time()
                    no_improve_steps = 0
                    if best_cost == 0:
                        return best_colors, True
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
            
            temp *= cool_rate
            
            if no_improve_steps > max_no_improve:
                temp = max(temp, 0.5)
                no_improve_steps = 0
            
            if (time.time() - last_improve_time) > 8.0 and best_cost > 0:
                if np.random.rand() < 0.15:
                    colors = best_colors.copy()
                    current_cost = best_cost
                    temp = 1.0
                    no_improve_steps = 0
        
        return best_colors, (best_cost == 0)

    best_n = 9
    best_result_colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9),3)]
    
    target_ns = [34, 33, 32, 31, 30, 28, 25, 20]
    
    for n in target_ns:
        if time.time() > deadline - 12:
            break
        
        # Allocate time for multiple attempts - give more time to larger n
        if n == 34:
            num_attempts = 4
            time_per_attempt = 28.0
        elif n == 33:
            num_attempts = 4
            time_per_attempt = 18.0
        elif n == 32:
            num_attempts = 3
            time_per_attempt = 12.0
        elif n == 31:
            num_attempts = 3
            time_per_attempt = 10.0
        elif n == 30:
            num_attempts = 2
            time_per_attempt = 8.0
        else:
            num_attempts = 2
            time_per_attempt = 5.0
        
        found = False
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            time_limit = time.time() + time_per_attempt
            seed = attempt * 7919  # different seed per attempt
            
            try:
                colors, is_valid = simulate_annealing(n, time_limit, seed=seed)
                if is_valid:
                    best_n = n
                    best_result_colors = colors.tolist()
                    found = True
                    break
                else:
                    # Try local search repair
                    if time.time() < deadline - 8:
                        triples, num_triples, k4s, k5s, k4s_containing, k5s_containing, k4_array, k5_array, k4s_containing_np, k5s_containing_np = build_indices(n)
                        repair_time = min(6.0, deadline - 8 - time.time())
                        if repair_time > 1.0:
                            colors, is_valid = local_search_repair(
                                colors, k4s, k5s, k4s_containing, k5s_containing, 
                                k4_array, k5_array, k4s_containing_np, k5s_containing_np,
                                num_triples, time.time() + repair_time
                            )
                            if is_valid:
                                best_n = n
                                best_result_colors = colors.tolist()
                                found = True
                                break
            except Exception:
                pass
        
        if found and n == 34:
            break
    
    return {'n': best_n, 'colors': best_result_colors}
