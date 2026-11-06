import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin under 120s

    def check_valid(n, colors):
        """Verify a coloring has no red K4 and no blue K5."""
        triples = list(combinations(range(n), 3))
        tidx = {t: i for i, t in enumerate(triples)}
        for k4 in combinations(range(n), 4):
            if all(colors[tidx[t]] == 0 for t in combinations(k4, 3)):
                return False
        for k5 in combinations(range(n), 5):
            if all(colors[tidx[t]] == 1 for t in combinations(k5, 3)):
                return False
        return True

    def solve_n(n, time_budget):
        """Find a valid coloring for n vertices using Conflict-Driven Local Search."""
        triples = list(combinations(range(n), 3))
        num_t = len(triples)
        tidx = {t: i for i, t in enumerate(triples)}

        # Precompute incidence
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)

        k4_triples = []
        for k4 in k4s:
            k4_triples.append([tidx[t] for t in combinations(k4, 3)])
        k5_triples = []
        for k5 in k5s:
            k5_triples.append([tidx[t] for t in combinations(k5, 3)])

        triple_to_k4 = [[] for _ in range(num_t)]
        for i, ts in enumerate(k4_triples):
            for t in ts:
                triple_to_k4[t].append(i)
        triple_to_k5 = [[] for _ in range(num_t)]
        for i, ts in enumerate(k5_triples):
            for t in ts:
                triple_to_k5[t].append(i)

        def init_state(colors):
            rc4 = [0] * num_k4
            rc5 = [0] * num_k5
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for k in triple_to_k4[i]:
                        rc4[k] += 1
                else: # Blue
                    for k in triple_to_k5[i]:
                        rc5[k] += 1
            viol_k4 = sum(1 for r in rc4 if r == 4)
            viol_k5 = sum(1 for r in rc5 if r == 10)
            near_k4 = sum(1 for r in rc4 if r == 3)
            near_k5 = sum(1 for r in rc5 if r == 9)
            return rc4, rc5, viol_k4, viol_k5, near_k4, near_k5

        def random_colors():
            return [random.randint(0, 1) for _ in range(num_t)]

        def pattern_colors(seed_pattern):
            """Structured initial coloring based on vertex patterns."""
            cols = []
            for (a, b, c) in triples:
                if seed_pattern == 0:
                    col = (a + b + c) % 2
                elif seed_pattern == 1:
                    col = (a * b + c) % 2
                elif seed_pattern == 2:
                    col = (a ^ b ^ c) & 1
                elif seed_pattern == 3:
                    col = (a + 2*b + 3*c) % 3 < 2
                else:
                    col = random.randint(0, 1)
                cols.append(col)
            return cols

        def local_search(colors, time_limit):
            """Conflict-Driven Local Search with incremental state updates."""
            rc4, rc5, viol_k4, viol_k5, near_k4, near_k5 = init_state(colors)
            
            def objective(viol, near):
                return viol * 1000 + near
            
            best_obj = objective(viol_k4 + viol_k5, near_k4 + near_k5)
            best_colors = list(colors)
            
            t_end = time.time() + time_limit
            steps = 0
            max_steps = 500000
            no_improve_count = 0
            threshold = 5000
            
            while steps < max_steps and time.time() < t_end:
                if time.time() > deadline - 5:
                    break
                
                candidates = []
                
                if random.random() < 0.8:
                    viol_k4_indices = [i for i, r in enumerate(rc4) if r == 4]
                    viol_k5_indices = [i for i, r in enumerate(rc5) if r == 10]
                    
                    if not viol_k4_indices and not viol_k5_indices:
                        near_k4_indices = [i for i, r in enumerate(rc4) if r == 3]
                        near_k5_indices = [i for i, r in enumerate(rc5) if r == 9]
                        
                        if near_k4_indices:
                            k = random.choice(near_k4_indices)
                            red_edges_in_k = [t for t in k4_triples[k] if colors[t] == 0]
                            if red_edges_in_k:
                                candidates = red_edges_in_k
                        elif near_k5_indices:
                            k = random.choice(near_k5_indices)
                            blue_edges_in_k = [t for t in k5_triples[k] if colors[t] == 1]
                            if blue_edges_in_k:
                                candidates = blue_edges_in_k
                        
                        if not candidates:
                            candidates = list(range(num_t))
                    else:
                        if viol_k4_indices and (not viol_k5_indices or random.random() < 0.5):
                            k = random.choice(viol_k4_indices)
                            red_edges_in_k = [t for t in k4_triples[k] if colors[t] == 0]
                            candidates = red_edges_in_k
                        else:
                            k = random.choice(viol_k5_indices)
                            blue_edges_in_k = [t for t in k5_triples[k] if colors[t] == 1]
                            candidates = blue_edges_in_k
                else:
                    candidates = list(range(num_t))
                
                if not candidates:
                    candidates = list(range(num_t))
                
                t = random.choice(candidates)
                old_color = colors[t]
                new_color = 1 - old_color
                
                d_obj = 0
                d_viol_k4 = 0
                d_viol_k5 = 0
                d_near_k4 = 0
                d_near_k5 = 0
                
                for k in triple_to_k4[t]:
                    r = rc4[k]
                    if old_color == 0: # R -> B
                        if r == 4:
                            d_viol_k4 -= 1
                            d_near_k4 += 1
                            d_obj -= 1000
                            d_obj += 1
                        elif r == 3:
                            d_near_k4 -= 1
                            d_obj -= 1
                    else: # B -> R
                        if r == 3:
                            d_viol_k4 += 1
                            d_near_k4 -= 1
                            d_obj += 1000
                            d_obj -= 1
                        elif r == 2:
                            d_near_k4 += 1
                            d_obj += 1
                
                for k in triple_to_k5[t]:
                    r = rc5[k]
                    if old_color == 0: # R -> B
                        if r == 9:
                            d_viol_k5 += 1
                            d_near_k5 -= 1
                            d_obj += 1000
                            d_obj -= 1
                        elif r == 8:
                            d_near_k5 += 1
                            d_obj += 1
                    else: # B -> R
                        if r == 10:
                            d_viol_k5 -= 1
                            d_near_k5 += 1
                            d_obj -= 1000
                            d_obj += 1
                        elif r == 9:
                            d_near_k5 -= 1
                            d_obj -= 1

                if d_obj < 0:
                    accept = True
                elif d_obj == 0:
                    accept = random.random() < 0.5
                else:
                    if no_improve_count > threshold:
                        p = 0.1 if d_obj < 100 else 0.01
                        accept = random.random() < p
                    else:
                        accept = False

                if accept:
                    colors[t] = new_color
                    
                    for k in triple_to_k4[t]:
                        if old_color == 0:
                            rc4[k] -= 1
                        else:
                            rc4[k] += 1
                    for k in triple_to_k5[t]:
                        if old_color == 0:
                            rc5[k] += 1
                        else:
                            rc5[k] -= 1
                    
                    viol_k4 += d_viol_k4
                    viol_k5 += d_viol_k5
                    near_k4 += d_near_k4
                    near_k5 += d_near_k5

                    if d_obj < 0:
                        no_improve_count = 0
                        current_obj = objective(viol_k4 + viol_k5, near_k4 + near_k5)
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_colors = list(colors)
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

                steps += 1

            return best_colors, best_obj

        num_starts = 5
        start_budget = time_budget / num_starts

        for s in range(num_starts):
            if time.time() > deadline - 10:
                break
            if s < 3:
                colors = pattern_colors(s)
            else:
                colors = random_colors()

            result, obj = local_search(colors, start_budget)
            if obj < 1000:
                if check_valid(n, result):
                    return result

        return None

    for n in range(34, 9, -1):
        if time.time() > deadline - 20:
            break
        budget = min(15.0, deadline - 20 - time.time())
        if budget < 3:
            break
        result = solve_n(n, budget)
        if result is not None:
            if check_valid(n, result):
                return {'n': n, 'colors': result}

    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
