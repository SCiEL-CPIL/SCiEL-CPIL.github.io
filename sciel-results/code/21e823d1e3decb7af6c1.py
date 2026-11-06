import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def solve_for_n(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        k4_indices = []
        for quad in combinations(range(n), 4):
            ts = [triple_idx[t] for t in combinations(quad, 3)]
            k4_indices.append(ts)
            
        k5_indices = []
        for quint in combinations(range(n), 5):
            ts = [triple_idx[t] for t in combinations(quint, 3)]
            k5_indices.append(ts)
            
        triple_in_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_indices):
            for t in ts:
                triple_in_k4[t].append(i)
                
        triple_in_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_indices):
            for t in ts:
                triple_in_k5[t].append(i)
                
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        red_k4_count = [0] * len(k4_indices)
        blue_k5_count = [0] * len(k5_indices)
        
        for i, ts in enumerate(k4_indices):
            for t in ts:
                if colors[t] == 0:
                    red_k4_count[i] += 1
                    
        for i, ts in enumerate(k5_indices):
            for t in ts:
                if colors[t] == 1:
                    blue_k5_count[i] += 1
                    
        cost = sum(1 for c in red_k4_count if c == 4) + sum(1 for c in blue_k5_count if c == 10)
        
        max_iters = 100000
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if cost == 0:
                return n, colors
                
            best_delta = 0
            best_t = -1
            best_color = -1
            
            # Optimization: Only check a random subset of triples if num_triples is large?
            # For n=20, 1140 is small. Check all.
            for t in range(num_triples):
                current_color = colors[t]
                new_color = 1 - current_color
                
                delta = 0
                
                for k4_i in triple_in_k4[t]:
                    curr_count = red_k4_count[k4_i]
                    if current_color == 0:
                        if curr_count == 4:
                            delta -= 1
                    else:
                        if curr_count == 3:
                            delta += 1
                            
                for k5_i in triple_in_k5[t]:
                    curr_count = blue_k5_count[k5_i]
                    if current_color == 1:
                        if curr_count == 10:
                            delta -= 1
                    else:
                        if curr_count == 9:
                            delta += 1
                            
                if delta < best_delta:
                    best_delta = delta
                    best_t = t
                    best_color = new_color
            if best_t == -1:
                t = random.randint(0, num_triples - 1)
                current_color = colors[t]
                new_color = 1 - current_color
                delta = 0
                for k4_i in triple_in_k4[t]:
                    curr_count = red_k4_count[k4_i]
                    if current_color == 0:
                        if curr_count == 4: delta -= 1
                    else:
                        if curr_count == 3: delta += 1
                for k5_i in triple_in_k5[t]:
                    curr_count = blue_k5_count[k5_i]
                    if current_color == 1:
                        if curr_count == 10: delta -= 1
                    else:
                        if curr_count == 9: delta += 1
                best_delta = delta
                best_t = t
                best_color = new_color
                
            t = best_t
            old_color = colors[t]
            new_color = best_color
            
            for k4_i in triple_in_k4[t]:
                if old_color == 0:
                    red_k4_count[k4_i] -= 1
                else:
                    red_k4_count[k4_i] += 1
                    
            for k5_i in triple_in_k5[t]:
                if old_color == 1:
                    blue_k5_count[k5_i] -= 1
                else:
                    blue_k5_count[k5_i] += 1
                    
            colors[t] = new_color
            cost += best_delta
            
        return None, None

    # Try n=20 first, then 25, then 30
    # Allocate time wisely.
    # n=20 is fast.
    for n in [20, 25, 30]:
        if time.time() > deadline - 10:
            break
        # Give 30s to n=20, 40s to n=25, rest to n=30?
        # Let's just run until deadline or solution found.
        res_n, res_colors = solve_for_n(n)
        if res_colors is not None:
            return {'n': res_n, 'colors': res_colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
