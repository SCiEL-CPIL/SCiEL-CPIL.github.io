import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, keep 35s safety margin
    
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s (red forbidden) and K5s (blue forbidden)
    k4_triple_indices = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_index[(a, b, c)]
        t2 = triple_index[(a, b, d)]
        t3 = triple_index[(a, c, d)]
        t4 = triple_index[(b, c, d)]
        k4_triple_indices.append((t1, t2, t3, t4))
    
    k5_triple_indices = []
    for pent in combinations(range(n), 5):
        indices = [triple_index[t] for t in combinations(pent, 3)]
        k5_triple_indices.append(tuple(indices))
    
    num_k4 = len(k4_triple_indices)
    num_k5 = len(k5_triple_indices)
    
    # Convert to numpy arrays for fast access
    k4_arr = np.array(k4_triple_indices, dtype=np.int32)  # (num_k4, 4)
    k5_arr = np.array(k5_triple_indices, dtype=np.int32)  # (num_k5, 10)
    
    # For each triple, which K4s and K5s contain it
    triple_to_k4s = [[] for _ in range(num_triples)]
    for i in range(num_k4):
        for t in k4_arr[i]:
            triple_to_k4s[t].append(i)
    
    triple_to_k5s = [[] for _ in range(num_triples)]
    for i in range(num_k5):
        for t in k5_arr[i]:
            triple_to_k5s[t].append(i)
    
    # Convert to numpy arrays
    triple_to_k4s_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                          for lst in triple_to_k4s]
    triple_to_k5s_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) 
                          for lst in triple_to_k5s]
    
    def count_violations(colors):
        """Count red K4s and blue K5s using vectorized operations."""
        # Red K4: all 4 triples are red (0)
        red_k4 = np.sum(np.all(colors[k4_arr] == 0, axis=1))
        # Blue K5: all 10 triples are blue (1)
        blue_k5 = np.sum(np.all(colors[k5_arr] == 1, axis=1))
        return red_k4 + blue_k5
    
    def incremental_flip_delta(colors, t_idx):
        """Calculate delta in violation count when flipping triple t_idx."""
        old_color = colors[t_idx]
        new_color = 1 - old_color
        delta = 0
        
        # Check K4s containing t_idx
        k4_indices = triple_to_k4s_arr[t_idx]
        if len(k4_indices) > 0:
            if new_color == 0:  # Becomes red
                # Count K4s where other 3 are already red
                for k4_i in k4_indices:
                    others = k4_arr[k4_i]
                    mask = others != t_idx
                    if np.all(colors[others[mask]] == 0):
                        delta += 1
            else:  # Becomes blue
                # Count K4s where other 3 are all red (destroys red K4)
                for k4_i in k4_indices:
                    others = k4_arr[k4_i]
                    mask = others != t_idx
                    if np.all(colors[others[mask]] == 0):
                        delta -= 1
        
        # Check K5s containing t_idx
        k5_indices = triple_to_k5s_arr[t_idx]
        if len(k5_indices) > 0:
            if new_color == 1:  # Becomes blue
                # Count K5s where other 9 are already blue
                for k5_i in k5_indices:
                    others = k5_arr[k5_i]
                    mask = others != t_idx
                    if np.all(colors[others[mask]] == 1):
                        delta += 1
            else:  # Becomes red
                # Count K5s where other 9 are all blue (destroys blue K5)
                for k5_i in k5_indices:
                    others = k5_arr[k5_i]
                    mask = others != t_idx
                    if np.all(colors[others[mask]] == 1):
                        delta -= 1
        
        return delta
    
    def simulated_annealing(colors, steps, start_temp=1.0, end_temp=0.01):
        """Run simulated annealing on the coloring."""
        current_cost = count_violations(colors)
        best_colors = colors.copy()
        best_cost = current_cost
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick random triple to flip
            t_idx = random.randint(0, num_triples - 1)
            
            # Calculate delta
            delta = incremental_flip_delta(colors, t_idx)
            
            # Simulated annealing acceptance
            if delta <= 0:
                # Always accept improvements or equal
                colors[t_idx] = 1 - colors[t_idx]
                current_cost += delta
            else:
                # Accept worse with probability exp(-delta/temp)
                temp = start_temp * (end_temp / start_temp) ** (step / steps)
                if random.random() < np.exp(-delta / max(temp, 1e-10)):
                    colors[t_idx] = 1 - colors[t_idx]
                    current_cost += delta
            
            # Track best
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    return best_colors, 0
        
        return best_colors, best_cost
    
    # Try multiple initializations and restarts
    best_colors = None
    best_cost = float('inf')
    
    # Different initial strategies
    initializations = []
    
    # Random
    initializations.append(np.random.randint(0, 2, num_triples).astype(np.int8))
    
    # Structured: based on vertex indices
    for seed_val in [0, 1, 2]:
        c = np.zeros(num_triples, dtype=np.int8)
        for i, (a, b, c_idx) in enumerate(triples):
            c[i] = (a + b + c_idx + seed_val) % 2
        initializations.append(c)
    
    # Bias towards one color
    for bias in [0.3, 0.5, 0.7]:
        c = (np.random.rand(num_triples) < bias).astype(np.int8)
        initializations.append(c)
    
    # Run SA from each initialization
    for init_idx, init_colors in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        c = init_colors.copy()
        c, cost = simulated_annealing(c, steps=20000, start_temp=2.0, end_temp=0.01)
        
        if cost < best_cost:
            best_cost = cost
            best_colors = c.copy()
            if best_cost == 0:
                return {'n': n, 'colors': best_colors.tolist()}
    
    # If we found a valid solution, return it
    if best_cost == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    
    # Try more aggressive restarts if time permits
    while time.time() < deadline - 15:
        if best_colors is None:
            c = np.random.randint(0, 2, num_triples).astype(np.int8)
        else:
            c = best_colors.copy()
            # Perturb
            num_perturb = random.randint(10, 50)
            for _ in range(num_perturb):
                c[random.randint(0, num_triples - 1)] = 1 - c[random.randint(0, num_triples - 1)]
        
        c, cost = simulated_annealing(c, steps=10000, start_temp=3.0, end_temp=0.01)
        
        if cost < best_cost:
            best_cost = cost
            best_colors = c.copy()
            if best_cost == 0:
                return {'n': n, 'colors': best_colors.tolist()}
    
    # If we still haven't found a valid coloring for n=34,
    # try to find valid colorings for smaller n
    for target_n in [33, 32, 31, 30, 25, 20, 15, 10, 9]:
        if time.time() > deadline - 5:
            break
        
        t = list(combinations(range(target_n), 3))
        nt = len(t)
        t_idx = {x: i for i, x in enumerate(t)}
        
        k4t = []
        for quad in combinations(range(target_n), 4):
            a, b, c, d = quad
            k4t.append((t_idx[(a,b,c)], t_idx[(a,b,d)], t_idx[(a,c,d)], t_idx[(b,c,d)]))
        
        k5t = []
        for pent in combinations(range(target_n), 5):
            k5t.append(tuple(t_idx[x] for x in combinations(pent, 3)))
        
        k4t_arr = np.array(k4t, dtype=np.int32) if k4t else np.empty((0,4), dtype=np.int32)
        k5t_arr = np.array(k5t, dtype=np.int32) if k5t else np.empty((0,10), dtype=np.int32)
        
        t2k4 = [[] for _ in range(nt)]
        for i in range(len(k4t)):
            for x in k4t[i]:
                t2k4[x].append(i)
        t2k5 = [[] for _ in range(nt)]
        for i in range(len(k5t)):
            for x in k5t[i]:
                t2k5[x].append(i)
        
        t2k4_arr = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in t2k4]
        t2k5_arr = [np.array(x, dtype=np.int32) if x else np.array([], dtype=np.int32) for x in t2k5]
        
        def count_v(target_n, colors, k4t_arr, k5t_arr):
            red = np.sum(np.all(colors[k4t_arr] == 0, axis=1)) if len(k4t_arr) > 0 else 0
            blue = np.sum(np.all(colors[k5t_arr] == 1, axis=1)) if len(k5t_arr) > 0 else 0
            return red + blue
        
        def inc_delta(colors, t_idx, nt, t2k4_arr, t2k5_arr, k4t_arr, k5t_arr):
            old = colors[t_idx]
            new = 1 - old
            delta = 0
            k4s = t2k4_arr[t_idx]
            if len(k4s) > 0:
                for k4_i in k4s:
                    others = k4t_arr[k4_i]
                    mask = others != t_idx
                    if np.all(colors[others[mask]] == (0 if new == 0 else 0)):
                        if new == 0:
                            delta += 1
                        else:
                            delta -= 1
            k5s = t2k5_arr[t_idx]
            if len(k5s) > 0:
                for k5_i in k5s:
                    others = k5t_arr[k5_i]
                    mask = others != t_idx
                    if np.all(colors[others[mask]] == (1 if new == 1 else 1)):
                        if new == 1:
                            delta += 1
                        else:
                            delta -= 1
            return delta
        
        def sa_small(colors, nt, steps, t2k4_arr, t2k5_arr, k4t_arr, k5t_arr):
            cur = count_v(target_n, colors, k4t_arr, k5t_arr)
            best_c = colors.copy()
            best_cost = cur
            for s in range(steps):
                if time.time() > deadline - 3:
                    break
                ti = random.randint(0, nt - 1)
                d = inc_delta(colors, ti, nt, t2k4_arr, t2k5_arr, k4t_arr, k5t_arr)
                if d <= 0 or random.random() < np.exp(-d / max(1.0 * (steps-s)/steps, 1e-10)):
                    colors[ti] = 1 - colors[ti]
                    cur += d
                if cur < best_cost:
                    best_cost = cur
                    best_c = colors.copy()
                    if best_cost == 0:
                        return best_c, 0
            return best_c, best_cost
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            c = np.random.randint(0, 2, nt).astype(np.int8)
            c, cost = sa_small(c, nt, 10000, t2k4_arr, t2k5_arr, k4t_arr, k5t_arr)
            if cost == 0:
                return {'n': target_n, 'colors': c.tolist()}
            found = True
        
        if found:
            break
    
    # Last resort: return a known small valid coloring
    # n=9 with sum%3==0 gives red
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
