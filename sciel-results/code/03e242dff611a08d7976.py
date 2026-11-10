import time
import math
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 295.0
    
    # Precompute
    xs = np.array([i // N for i in range(N * N)], dtype=np.int32)
    ys = np.array([i % N for i in range(N * N)], dtype=np.int32)
    
    # dist_matrix[i, j] = squared distance between i and j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    all_points = list(range(N * N))
    
    def solve_once(seed, time_limit):
        start = time.time()
        limit = start + time_limit
        
        rng = random.Random(seed)
        
        selected = set()
        selected_list = []
        dist_sets = {}
        
        def can_add(p):
            if p in selected:
                return False
            
            # Fast check: distances from p to all selected
            # We use the precomputed dist_matrix row p
            # selected_list is a list of indices
            # We can extract distances quickly if we had an array of selected indices
            # But selected_list changes.
            # Let's stick to the loop but optimize the inner check.
            
            seen_dists = set()
            # Optimization: if len(selected_list) is large, we might want to use numpy
            # dists = dist_matrix[p, selected_list] # This creates a new array each time, might be slow due to allocation
            # if len(set(dists)) < len(dists): return False
            
            # For now, python loop is okay if K is small (< 50)
            # If K grows, numpy is better.
            
            for q in selected_list:
                d = dist_matrix[p, q]
                
                if d in seen_dists:
                    return False
                seen_dists.add(d)
                
                if d in dist_sets[q]:
                    return False
                    
            return True

        def add(p):
            selected.add(p)
            selected_list.append(p)
            dist_sets[p] = set()
            for q in selected_list[:-1]:
                d = dist_matrix[p, q]
                dist_sets[p].add(d)
                dist_sets[q].add(d)

        def remove(p):
            if p not in selected:
                return
            selected.discard(p)
            selected_list.remove(p)
            for q in selected_list:
                d = dist_matrix[p, q]
                dist_sets[q].discard(d)
            del dist_sets[p]

        # Greedy Init
        # Try a few orders
        orders = []
        ord1 = all_points.copy()
        rng.shuffle(ord1)
        orders.append(ord1)
        
        center = N // 2
        ord2 = sorted(all_points, key=lambda i: (xs[i]-center)**2 + (ys[i]-center)**2)
        orders.append(ord2)
        
        best_greedy_size = 0
        best_greedy_state = None

        for order in orders:
            selected.clear()
            selected_list.clear()
            dist_sets.clear()
            
            for p in order:
                if can_add(p):
                    add(p)
            
            if len(selected) > best_greedy_size:
                best_greedy_size = len(selected)
                best_greedy_state = (set(selected), list(selected_list), {k: set(v) for k, v in dist_sets.items()})

        if not best_greedy_state:
            return []
            
        selected, selected_list, dist_sets = best_greedy_state
        unselected = [p for p in all_points if p not in selected]
        
        best_size = len(selected)
        best_solution = [(i // N, i % N) for i in selected]
        
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < limit:
            improved = False
            
            # 1. Try Add
            if unselected:
                rng.shuffle(unselected)
                for p in unselected[:10]:
                    if can_add(p):
                        add(p)
                        unselected.remove(p)
                        improved = True
                        break
            
            if improved:
                no_improve = 0
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = [(i // N, i % N) for i in selected]
                continue
            
            # 2. Try 1-Swap
            if selected_list and unselected:
                p_out = rng.choice(selected_list)
                found = False
                for _ in range(10):
                    if not unselected: break
                    p_in = rng.choice(unselected)
                    remove(p_out)
                    if can_add(p_in):
                        add(p_in)
                        unselected.remove(p_in)
                        unselected.append(p_out)
                        found = True
                        improved = True
                        break
                    else:
                        add(p_out)
                if found:
                    no_improve = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_solution = [(i // N, i % N) for i in selected]
                    continue
            
            # 3. Try 2-Swap (Remove 2, Add 2)
            if len(selected_list) >= 2 and len(unselected) >= 2:
                p_out1 = rng.choice(selected_list)
                p_out2 = rng.choice([p for p in selected_list if p != p_out1])
                
                found = False
                for _ in range(10):
                    if len(unselected) < 2: break
                    p_in1 = rng.choice(unselected)
                    p_in2 = rng.choice([p for p in unselected if p != p_in1])
                    
                    # Remove p_out1, p_out2
                    remove(p_out1)
                    remove(p_out2)
                    
                    # Try add p_in1
                    if can_add(p_in1):
                        add(p_in1)
                        # Try add p_in2
                        if can_add(p_in2):
                            add(p_in2)
                            unselected.remove(p_in1)
                            unselected.remove(p_in2)
                            unselected.append(p_out1)
                            unselected.append(p_out2)
                            found = True
                            improved = True
                            break
                        else:
                            # Revert p_in1
                            remove(p_in1)
                    
                    # Revert p_out1, p_out2 if not found
                    if not found:
                        add(p_out1)
                        add(p_out2)
                    
                if found:
                    no_improve = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_solution = [(i // N, i % N) for i in selected]
                    continue
            
            # 4. Kick (Perturbation)
            no_improve += 1
            if no_improve >= max_no_improve:
                no_improve = 0
                k = rng.randint(5, 10)
                if len(selected_list) > k:
                    to_remove = rng.sample(selected_list, k)
                    for p in to_remove:
                        remove(p)
                        unselected.append(p)
                    
                    # Refill
                    rng.shuffle(unselected)
                    for p in unselected:
                        if can_add(p):
                            add(p)
                            unselected.remove(p)
                            # Limit refill time
                            if time.time() > limit - 5:
                                break

        return best_solution

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        runs_left = len(seeds) - seeds.index(seed)
        time_per_run = min(15.0, max(5.0, (remaining - 5.0) / runs_left))
        
        if time.time() + time_per_run > deadline - 2:
            time_per_run = max(1.0, deadline - time.time() - 2)
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
