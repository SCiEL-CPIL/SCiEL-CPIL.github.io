"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def generate_triples(n):
        return list(combinations(range(n), 3))

    def precompute_structures(n):
        triples = generate_triples(n)
        m = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        red_k4s = []
        for quad in combinations(range(n), 4):
            red_k4s.append((
                triple_idx[(quad[0], quad[1], quad[2])],
                triple_idx[(quad[0], quad[1], quad[3])],
                triple_idx[(quad[0], quad[2], quad[3])],
                triple_idx[(quad[1], quad[2], quad[3])]
            ))
            
        blue_k5s = []
        for quint in combinations(range(n), 5):
            k5_triples = [triple_idx[tri] for tri in combinations(quint, 3)]
            blue_k5s.append(k5_triples)
            
        # Precompute flat arrays for vectorized access
        # For K4s: shape (num_k4s, 4)
        k4_arr = np.array(red_k4s, dtype=np.int32) if red_k4s else np.empty((0,4), dtype=np.int32)
        # For K5s: shape (num_k5s, 10)
        k5_arr = np.array(blue_k5s, dtype=np.int32) if blue_k5s else np.empty((0,10), dtype=np.int32)
        
        # Inverse mapping: triple_idx -> list of k4_indices and k5_indices
        triple_to_k4s = [[] for _ in range(m)]
        for i, k4 in enumerate(red_k4s):
            for t in k4:
                triple_to_k4s[t].append(i)
                
        triple_to_k5s = [[] for _ in range(m)]
        for i, k5 in enumerate(blue_k5s):
            for t in k5:
                triple_to_k5s[t].append(i)
                
        return m, triple_idx, k4_arr, k5_arr, triple_to_k4s, triple_to_k5s

    def vectorized_local_search(n, seed, init_strategy="random"):
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        m, triple_idx, k4_arr, k5_arr, triple_to_k4s, triple_to_k5s = precompute_structures(n)
        
        # Initialize colors
        if init_strategy == "random":
            colors = np_rng.integers(0, 2, size=m, dtype=np.uint8)
        elif init_strategy == "modular":
            # Try a pattern: color(a,b,c) = (a+b+c) % 2
            colors = np.zeros(m, dtype=np.uint8)
            for i, (a,b,c) in enumerate(generate_triples(n)):
                colors[i] = (a + b + c) % 2
        elif init_strategy == "modular3":
            # color(a,b,c) = 0 if (a+b+c) % 3 == 0 else 1
            colors = np.zeros(m, dtype=np.uint8)
            for i, (a,b,c) in enumerate(generate_triples(n)):
                colors[i] = 0 if (a + b + c) % 3 == 0 else 1
        else:
            colors = np_rng.integers(0, 2, size=m, dtype=np.uint8)
            
        num_k4s = len(k4_arr)
        num_k5s = len(k5_arr)
        
        is_red_viol = np.zeros(num_k4s, dtype=bool)
        is_blue_viol = np.zeros(num_k5s, dtype=bool)
        
        def update_violations():
            nonlocal is_red_viol, is_blue_viol
            if num_k4s > 0:
                # Check all K4s
                # k4_arr shape (K, 4)
                # colors[k4_arr] shape (K, 4)
                red_check = np.all(colors[k4_arr] == 0, axis=1)
                is_red_viol = red_check
            if num_k5s > 0:
                blue_check = np.all(colors[k5_arr] == 1, axis=1)
                is_blue_viol = blue_check
                
        update_violations()
        red_viol_count = np.sum(is_red_viol)
        blue_viol_count = np.sum(is_blue_viol)
        
        best_red = red_viol_count
        best_blue = blue_viol_count
        best_colors = colors.copy()
        
        max_iters = 200000
        step_size = 1.0
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            if red_viol_count == 0 and blue_viol_count == 0:
                return colors.tolist()
                
            # Pick a violation to flip
            # If both exist, pick randomly. If only one, pick that.
            if red_viol_count > 0 and (blue_viol_count == 0 or (rng.random() < 0.5)):
                # Find a random red violation
                # Instead of scanning, we can maintain a list of active violations?
                # For now, random sampling from indices where is_red_viol is True
                # Efficient way: get indices of True, pick one.
                # np.nonzero(is_red_viol)[0]
                active_reds = np.nonzero(is_red_viol)[0]
                if len(active_reds) == 0:
                    continue
                vi = active_reds[rng.randint(0, len(active_reds)-1)]
                # Get the 4 triples in this K4
                k4_triples = k4_arr[vi]
                # Flip one random triple in this K4
                flip_t = k4_triples[rng.randint(0, 3)]
                old_c = colors[flip_t]
                new_c = 1 - old_c
                colors[flip_t] = new_c
                
                # Update affected K4s
                affected_k4s = triple_to_k4s[flip_t]
                if affected_k4s:
                    # Get the triples for these K4s
                    # k4_arr[affected_k4s] shape (num_affected, 4)
                    sub_k4s = k4_arr[affected_k4s]
                    # Check if all red
                    new_viol = np.all(colors[sub_k4s] == 0, axis=1)
                    # Update is_red_viol for these indices
                    old_viol = is_red_viol[affected_k4s]
                    # Count changes
                    red_viol_count += np.sum(new_viol) - np.sum(old_viol)
                    is_red_viol[affected_k4s] = new_viol
                    
                # Update affected K5s
                affected_k5s = triple_to_k5s[flip_t]
                if affected_k5s:
                    sub_k5s = k5_arr[affected_k5s]
                    new_viol = np.all(colors[sub_k5s] == 1, axis=1)
                    old_viol = is_blue_viol[affected_k5s]
                    blue_viol_count += np.sum(new_viol) - np.sum(old_viol)
                    is_blue_viol[affected_k5s] = new_viol
            else:
                if blue_viol_count == 0:
                    continue
                active_blues = np.nonzero(is_blue_viol)[0]
                if len(active_blues) == 0:
                    continue
                vi = active_blues[rng.randint(0, len(active_blues)-1)]
                k5_triples = k5_arr[vi]
                flip_t = k5_triples[rng.randint(0, 9)]
                old_c = colors[flip_t]
                new_c = 1 - old_c
                colors[flip_t] = new_c
                
                affected_k4s = triple_to_k4s[flip_t]
                if affected_k4s:
                    sub_k4s = k4_arr[affected_k4s]
                    new_viol = np.all(colors[sub_k4s] == 0, axis=1)
                    old_viol = is_red_viol[affected_k4s]
                    red_viol_count += np.sum(new_viol) - np.sum(old_viol)
                    is_red_viol[affected_k4s] = new_viol
                    
                affected_k5s = triple_to_k5s[flip_t]
                if affected_k5s:
                    sub_k5s = k5_arr[affected_k5s]
                    new_viol = np.all(colors[sub_k5s] == 1, axis=1)
                    old_viol = is_blue_viol[affected_k5s]
                    blue_viol_count += np.sum(new_viol) - np.sum(old_viol)
                    is_blue_viol[affected_k5s] = new_viol
            
            if red_viol_count < best_red or (red_viol_count == best_red and blue_viol_count < best_blue):
                best_red = red_viol_count
                best_blue = blue_viol_count
                best_colors = colors.copy()
                
        if best_red == 0 and best_blue == 0:
            return best_colors.tolist()
        return None

    best_n = 0
    best_colors = None
    
    # Try n=34 with multiple strategies and seeds
    strategies = ["random", "modular", "modular3"]
    for strat in strategies:
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            res = vectorized_local_search(34, seed, strat)
            if res:
                best_n = 34
                best_colors = res
                break
        if best_n == 34:
            break
            
    if best_n < 33:
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            res = vectorized_local_search(33, seed, "random")
            if res:
                best_n = 33
                best_colors = res
                break
                
    if best_n < 32:
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            res = vectorized_local_search(32, seed, "random")
            if res:
                best_n = 32
                best_colors = res
                break
                
    if best_n < 32:
        for n in range(31, 10, -1):
            if time.time() > deadline - 5:
                break
            res = vectorized_local_search(n, 12345, "random")
            if res:
                best_n = n
                best_colors = res
                break
                
    if best_n == 0:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
