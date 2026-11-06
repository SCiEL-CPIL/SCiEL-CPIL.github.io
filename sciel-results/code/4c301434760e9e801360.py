import random
import math
import time
from itertools import combinations

def construct():
    n = 34
    triples = list(combinations(range(n), 3))
    T = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    k4_triples = []
    for k4 in k4s:
        t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
        k4_triples.append(t_indices)
        
    k5_triples = []
    for k5 in k5s:
        t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
        k5_triples.append(t_indices)
        
    triple_to_k4s = [[] for _ in range(T)]
    triple_to_k5s = [[] for _ in range(T)]
    
    for i, t_inds in enumerate(k4_triples):
        for t in t_inds:
            triple_to_k4s[t].append(i)
            
    for i, t_inds in enumerate(k5_triples):
        for t in t_inds:
            triple_to_k5s[t].append(i)

    # Track number of red edges in each K4 and blue edges in each K5
    # k4_red_count[k4_idx] = number of red triples in that K4
    # k5_blue_count[k5_idx] = number of blue triples in that K5
    
    def init_counts():
        k4_red_count = [0] * len(k4s)
        k5_blue_count = [0] * len(k5s)
        for t_idx, c in enumerate(colors):
            for k4_idx in triple_to_k4s[t_idx]:
                if c == 0:
                    k4_red_count[k4_idx] += 1
            for k5_idx in triple_to_k5s[t_idx]:
                if c == 1:
                    k5_blue_count[k5_idx] += 1
        return k4_red_count, k5_blue_count

    def check_valid(k4_red_count, k5_blue_count):
        for c in k4_red_count:
            if c == 4:
                return False
        for c in k5_blue_count:
            if c == 10:
                return False
        return True

    start_time = time.time()
    timeout = 115
    
    # Try multiple random restarts with different biases
    best_result = None
    
    for p_red in [0.3, 0.25, 0.35, 0.2, 0.4]:
        if time.time() - start_time > timeout:
            break
            
        colors = [0 if random.random() < p_red else 1 for _ in range(T)]
        k4_red_count, k5_blue_count = init_counts()
        
        if check_valid(k4_red_count, k5_blue_count):
            return {'n': n, 'colors': colors}
        
        # Count violations
        violations = 0
        violated_k4s = set()
        violated_k5s = set()
        for i, c in enumerate(k4_red_count):
            if c == 4:
                violations += 1
                violated_k4s.add(i)
        for i, c in enumerate(k5_blue_count):
            if c == 10:
                violations += 1
                violated_k5s.add(i)
        
        # Local search: flip triples involved in violations
        T_curr = 1.0
        cooling = 0.999
        
        while violations > 0 and time.time() - start_time < timeout:
            # Pick a triple from a violated structure
            if violated_k4s:
                k4_idx = random.choice(list(violated_k4s))
                t_idx = random.choice(k4_triples[k4_idx])
            elif violated_k5s:
                k5_idx = random.choice(list(violated_k5s))
                t_idx = random.choice(k5_triples[k5_idx])
            else:
                t_idx = random.randint(0, T - 1)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta in violations
            delta = 0
            
            # Update K4 counts
            for k4_idx in triple_to_k4s[t_idx]:
                old_cnt = k4_red_count[k4_idx]
                if old_color == 0:
                    # Red -> Blue
                    new_cnt = old_cnt - 1
                    if old_cnt == 4:
                        delta -= 1
                    if new_cnt == 4:
                        delta += 1
                else:
                    # Blue -> Red
                    new_cnt = old_cnt + 1
                    if old_cnt == 4:
                        delta -= 1
                    if new_cnt == 4:
                        delta += 1
                k4_red_count[k4_idx] = new_cnt
            
            # Update K5 counts
            for k5_idx in triple_to_k5s[t_idx]:
                old_cnt = k5_blue_count[k5_idx]
                if old_color == 1:
                    # Blue -> Red
                    new_cnt = old_cnt - 1
                    if old_cnt == 10:
                        delta -= 1
                    if new_cnt == 10:
                        delta += 1
                else:
                    # Red -> Blue
                    new_cnt = old_cnt + 1
                    if old_cnt == 10:
                        delta -= 1
                    if new_cnt == 10:
                        delta += 1
                k5_blue_count[k5_idx] = new_cnt
            
            # Acceptance
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                prob = math.exp(-delta / T_curr)
                accept = random.random() < prob
            
            if accept:
                colors[t_idx] = new_color
                violations += delta
                # Update violated sets approximately
                if delta < 0:
                    # Some violations resolved
                    pass
                # Rebuild violated sets periodically or on the fly is complex
                # For simplicity, just track violations count
            else:
                # Roll back
                for k4_idx in triple_to_k4s[t_idx]:
                    if old_color == 0:
                        k4_red_count[k4_idx] += 1
                    else:
                        k4_red_count[k4_idx] -= 1
                for k5_idx in triple_to_k5s[t_idx]:
                    if old_color == 1:
                        k5_blue_count[k5_idx] += 1
                    else:
                        k5_blue_count[k5_idx] -= 1
            
            T_curr *= cooling
        
        if violations == 0:
            return {'n': n, 'colors': colors}
    
    # Fallback: try to find valid coloring for smaller n
    for n_fallback in [33, 32, 31, 30]:
        if time.time() - start_time > timeout - 5:
            break
        triples_f = list(combinations(range(n_fallback), 3))
        T_f = len(triples_f)
        t2i_f = {t: i for i, t in enumerate(triples_f)}
        k4s_f = list(combinations(range(n_fallback), 4))
        k5s_f = list(combinations(range(n_fallback), 5))
        k4t_f = [[t2i_f[t] for t in combinations(k4, 3)] for k4 in k4s_f]
        k5t_f = [[t2i_f[t] for t in combinations(k5, 3)] for k5 in k5s_f]
        t2k4_f = [[] for _ in range(T_f)]
        t2k5_f = [[] for _ in range(T_f)]
        for i, inds in enumerate(k4t_f):
            for t in inds:
                t2k4_f[t].append(i)
        for i, inds in enumerate(k5t_f):
            for t in inds:
                t2k5_f[t].append(i)
        
        colors_f = [0 if random.random() < 0.3 else 1 for _ in range(T_f)]
        k4rc_f = [0] * len(k4s_f)
        k5bc_f = [0] * len(k5s_f)
        for t_idx, c in enumerate(colors_f):
            for k4i in t2k4_f[t_idx]:
                if c == 0:
                    k4rc_f[k4i] += 1
            for k5i in t2k5_f[t_idx]:
                if c == 1:
                    k5bc_f[k5i] += 1
        
        viol_f = sum(1 for c in k4rc_f if c == 4) + sum(1 for c in k5bc_f if c == 10)
        
        T_curr_f = 1.0
        cooling_f = 0.999
        while viol_f > 0 and time.time() - start_time < timeout - 2:
            t_idx = random.randint(0, T_f - 1)
            old_c = colors_f[t_idx]
            new_c = 1 - old_c
            delta = 0
            for k4i in t2k4_f[t_idx]:
                oc = k4rc_f[k4i]
                if old_c == 0:
                    nc = oc - 1
                    if oc == 4: delta -= 1
                    if nc == 4: delta += 1
                else:
                    nc = oc + 1
                    if oc == 4: delta -= 1
                    if nc == 4: delta += 1
                k4rc_f[k4i] = nc
            for k5i in t2k5_f[t_idx]:
                oc = k5bc_f[k5i]
                if old_c == 1:
                    nc = oc - 1
                    if oc == 10: delta -= 1
                    if nc == 10: delta += 1
                else:
                    nc = oc + 1
                    if oc == 10: delta -= 1
                    if nc == 10: delta += 1
                k5bc_f[k5i] = nc
            
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < math.exp(-delta / T_curr_f)
            
            if accept:
                colors_f[t_idx] = new_c
                viol_f += delta
            else:
                for k4i in t2k4_f[t_idx]:
                    if old_c == 0: k4rc_f[k4i] += 1
                    else: k4rc_f[k4i] -= 1
                for k5i in t2k5_f[t_idx]:
                    if old_c == 1: k5bc_f[k5i] += 1
                    else: k5bc_f[k5i] -= 1
            T_curr_f *= cooling_f
        
        if viol_f == 0:
            return {'n': n_fallback, 'colors': colors_f}
    
    # Final safe fallback
    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
