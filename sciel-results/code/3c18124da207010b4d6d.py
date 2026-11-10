import time
import math
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin
    
    # Precompute coordinates
    xs = np.array([i // N for i in range(N * N)], dtype=np.int32)
    ys = np.array([i % N for i in range(N * N)], dtype=np.int32)
    
    all_points = list(range(N * N))
    total_points = N * N
    
    # Precompute all pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)
    
    def solve_once(seed, time_limit):
        start = time.time()
        limit = start + time_limit
        
        rng = random.Random(seed)
        
        # Data structures
        selected = set()
        selected_list = []
        selected_idx = {}  # point -> index in selected_list for O(1) removal
        dist_sets = {}  # dist_sets[i] = set of squared distances from i to other selected points
        in_unselected = [False] * total_points  # boolean array for fast unselected membership
        
        def can_add(p):
            if p in selected:
                return False
            
            seen_dists = set()
            for q in selected_list:
                d = dist_matrix[p, q]
                
                # Check if d is already seen (p as apex)
                if d in seen_dists:
                    return False
                seen_dists.add(d)
                
                # Check if d is in dist_sets[q] (q as apex)
                if d in dist_sets[q]:
                    return False
                    
            return True

        def add(p):
            selected.add(p)
            idx = len(selected_list)
            selected_idx[p] = idx
            selected_list.append(p)
            dist_sets[p] = set()
            # Update dist_sets for p and existing points
            for q in selected_list[:-1]:
                d = dist_matrix[p, q]
                dist_sets[p].add(d)
                dist_sets[q].add(d)
            in_unselected[p] = False

        def remove(p):
            if p not in selected:
                return
            selected.discard(p)
            idx = selected_idx[p]
            last_idx = len(selected_list) - 1
            if idx != last_idx:
                last_p = selected_list[last_idx]
                selected_list[idx] = last_p
                selected_idx[last_p] = idx
            selected_list.pop()
            del selected_idx[p]
            # Update dist_sets
            for q in selected_list:
                d = dist_matrix[p, q]
                dist_sets[q].discard(d)
            del dist_sets[p]
            in_unselected[p] = True

        # Greedy Initialization
        orders = []
        
        # 1. Random
        ord1 = all_points.copy()
        rng.shuffle(ord1)
        orders.append(ord1)
        
        # 2. Center-out
        center = N // 2
        ord2 = sorted(all_points, key=lambda i: (xs[i]-center)**2 + (ys[i]-center)**2)
        orders.append(ord2)
        
        # 3. Corner-in
        ord3 = sorted(all_points, key=lambda i: min(xs[i], N-1-xs[i]) + min(ys[i], N-1-ys[i]))
        orders.append(ord3)
        
        # 4. Diagonal
        ord4 = sorted(all_points, key=lambda i: abs(xs[i] - ys[i]))
        orders.append(ord4)

        best_greedy_size = 0
        best_greedy_selected = set()
        best_greedy_dist_sets = {}
        best_greedy_list = []
        best_greedy_idx = {}

        for order in orders:
            selected.clear()
            selected_list.clear()
            selected_idx.clear()
            dist_sets.clear()
            
            for p in order:
                if can_add(p):
                    add(p)
            
            if len(selected) > best_greedy_size:
                best_greedy_size = len(selected)
                best_greedy_selected = set(selected)
                best_greedy_dist_sets = {k: set(v) for k, v in dist_sets.items()}
                best_greedy_list = list(selected_list)
                best_greedy_idx = dict(selected_idx)

        # Restore best greedy state
        selected = best_greedy_selected
        selected_list = best_greedy_list
        selected_idx = best_greedy_idx
        dist_sets = best_greedy_dist_sets
        
        # Build unselected list using boolean array
        unselected = [p for p in all_points if p not in selected]
        for p in unselected:
            in_unselected[p] = True
        for p in selected:
            in_unselected[p] = False
        
        best_size = len(selected)
        best_solution = [(i // N, i % N) for i in selected]
        
        no_improve_count = 0
        max_no_improve = 30
        
        # Tabu list
        tabu_list = {}
        tabu_tenure = 50
        
        while time.time() < limit:
            improved = False
            
            # 1. Try to add any unselected point
            if unselected:
                # Shuffle a portion of unselected for randomness
                shuffle_len = min(50, len(unselected))
                for _ in range(shuffle_len):
                    i = rng.randint(0, len(unselected)-1)
                    unselected[0], unselected[i] = unselected[i], unselected[0]
                
                # Try to add the first few that fit
                for p in unselected[:shuffle_len]:
                    if in_unselected[p] and can_add(p):
                        add(p)
                        # Remove from unselected list (swap with last)
                        idx = unselected.index(p)
                        last = len(unselected) - 1
                        if idx != last:
                            unselected[idx] = unselected[last]
                        unselected.pop()
                        improved = True
                        break
            
            if improved:
                no_improve_count = 0
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_solution = [(i // N, i % N) for i in selected]
                continue
            
            # 2. Try Swap (1-out, 1-in)
            if selected_list and unselected:
                # Pick a random point to remove
                p_out = rng.choice(selected_list)
                
                # Find points that were blocked by p_out (share a distance conflict)
                # These are good candidates for p_in
                blocked_candidates = []
                for q in unselected:
                    if not in_unselected[q]:
                        continue
                    d = dist_matrix[p_out, q]
                    # Check if d is in dist_sets for any selected point (conflict)
                    # or if d would create a duplicate distance from q
                    # Simplified: just check if d is in dist_sets[q] after removing p_out
                    # For now, use a heuristic: points with small distance to p_out are more likely to be unblocked
                    if d < 100:  # Small distance threshold
                        blocked_candidates.append(q)
                
                # If no blocked candidates found, use random
                if not blocked_candidates:
                    candidates = [q for q in unselected[:min(20, len(unselected))] if in_unselected[q]]
                else:
                    candidates = blocked_candidates[:min(20, len(blocked_candidates))]
                
                found_swap = False
                for p_in in candidates:
                    if not in_unselected[p_in]:
                        continue
                    
                    # Check if swap is possible
                    remove(p_out)
                    if can_add(p_in):
                        add(p_in)
                        # Remove p_in from unselected
                        idx = unselected.index(p_in)
                        last = len(unselected) - 1
                        if idx != last:
                            unselected[idx] = unselected[last]
                        unselected.pop()
                        # Add p_out to unselected
                        unselected.append(p_out)
                        in_unselected[p_out] = True
                        found_swap = True
                        improved = True
                        # Update tabu
                        tabu_list[p_out] = tabu_tenure
                        break
                    else:
                        add(p_out)
                
                if found_swap:
                    no_improve_count = 0
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_solution = [(i // N, i % N) for i in selected]
                    continue
            
            # 3. Kick (Perturbation)
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                # Kick: remove k random points
                k = rng.randint(3, 7)
                if len(selected_list) > k:
                    to_remove = rng.sample(selected_list, k)
                    for p in to_remove:
                        remove(p)
                        unselected.append(p)
                        in_unselected[p] = True
                        tabu_list[p] = tabu_tenure
                    
                    # Refill greedily
                    shuffle_len = min(100, len(unselected))
                    for _ in range(shuffle_len):
                        i = rng.randint(0, len(unselected)-1)
                        unselected[0], unselected[i] = unselected[i], unselected[0]
                    
                    for p in unselected[:shuffle_len]:
                        if not in_unselected[p]:
                            continue
                        if can_add(p):
                            add(p)
                            # Remove from unselected
                            idx = unselected.index(p)
                            last = len(unselected) - 1
                            if idx != last:
                                unselected[idx] = unselected[last]
                            unselected.pop()
                            if len(unselected) == 0:
                                break
                            
            # Decay tabu
            for p in list(tabu_list.keys()):
                if tabu_list[p] > 0:
                    tabu_list[p] -= 1
                    if tabu_list[p] == 0:
                        del tabu_list[p]

        return best_solution

    best_result = []
    best_size = 0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        runs_left = len(seeds) - seeds.index(seed)
        # Allocate more time to later seeds? Or equal?
        # Let's give each seed a fair share, but ensure we finish.
        time_per_run = min(15.0, max(5.0, (remaining - 5.0) / runs_left))
        
        if time.time() + time_per_run > deadline - 2:
            time_per_run = max(1.0, deadline - time.time() - 2)
            
        result = solve_once(seed, time_per_run)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
