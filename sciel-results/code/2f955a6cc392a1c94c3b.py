import numpy as np
import time
import random
from collections import defaultdict

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline, 40s margin
    
    best_points = []
    best_count = 0
    
    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    def check_valid_addition(S, dist_conflicts, p):
        """Check if p can be added to S without creating isosceles triangle."""
        if not S:
            return True
        px, py = p
        new_dists = []
        for q in S:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            if d_sq in dist_conflicts[q]:
                return False
            new_dists.append(d_sq)
        if len(set(new_dists)) != len(new_dists):
            return False
        return True
    
    def try_add_point(S, dist_conflicts, p):
        """Add point p to S if valid, update structures, return True if added."""
        if not check_valid_addition(S, dist_conflicts, p):
            return False
        
        px, py = p
        new_dists = []
        for q in S:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            new_dists.append(d_sq)
        
        for i, q in enumerate(S):
            dist_conflicts[q].add(new_dists[i])
        
        dist_conflicts[p] = set(new_dists)
        S.append(p)
        return True
    
    def remove_point(S, dist_conflicts, p):
        """Remove point p from S and update structures."""
        px, py = p
        S.remove(p)
        for q in S:
            qx, qy = q
            d_sq = (px - qx)**2 + (py - qy)**2
            if d_sq in dist_conflicts[q]:
                dist_conflicts[q].discard(d_sq)
        del dist_conflicts[p]
    
    def greedy_construct(seed_val):
        """Greedy construction to get an initial solution."""
        random.seed(seed_val)
        S = []
        dist_conflicts = {}
        candidates = all_points[:]
        random.shuffle(candidates)
        
        for p in candidates:
            if time.time() > deadline - 30:
                break
            try_add_point(S, dist_conflicts, p)
        
        return S, dist_conflicts
    
    def swap_local_search(S, dist_conflicts, time_limit):
        """
        Structural change: swap-based local search.
        Instead of only adding points, we can swap an existing point with a new one.
        This allows escaping local optima where no single addition works,
        but a removal+addition sequence does.
        """
        start_ls = time.time()
        current_count = len(S)
        best_count_ls = current_count
        best_S_ls = S[:]
        best_dist_ls = {k: set(v) for k, v in dist_conflicts.items()}
        
        S_set = set(S)
        
        # Temperature for simulated annealing
        temperature = 5.0
        cooling_rate = 0.9995
        min_temperature = 0.1
        
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() - start_ls < time_limit and no_improve < max_no_improve:
            if temperature < min_temperature:
                temperature = min_temperature
            
            # Choose operation: add (50%), swap (30%), remove (20%)
            op = random.random()
            
            if op < 0.5:
                # Try to add a random point
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p in S_set:
                    continue
                
                if check_valid_addition(S, dist_conflicts, p):
                    try_add_point(S, dist_conflicts, p)
                    S_set.add(p)
                    new_count = len(S)
                    no_improve = 0
                    
                    if new_count > best_count_ls:
                        best_count_ls = new_count
                        best_S_ls = S[:]
                        best_dist_ls = {k: set(v) for k, v in dist_conflicts.items()}
                else:
                    no_improve += 1
                    
            elif op < 0.8:
                # Try a swap: remove one existing point, add a new one
                if not S:
                    continue
                
                old_p = random.choice(S)
                # Temporarily remove old_p
                remove_point(S, dist_conflicts, old_p)
                S_set.discard(old_p)
                
                # Try to add a new point
                p = (random.randint(0, N-1), random.randint(0, N-1))
                if p in S_set:
                    # Restore and continue
                    try_add_point(S, dist_conflicts, old_p)
                    S_set.add(old_p)
                    no_improve += 1
                    continue
                
                if check_valid_addition(S, dist_conflicts, p):
                    try_add_point(S, dist_conflicts, p)
                    S_set.add(p)
                    new_count = len(S)
                    no_improve = 0
                    
                    if new_count > best_count_ls:
                        best_count_ls = new_count
                        best_S_ls = S[:]
                        best_dist_ls = {k: set(v) for k, v in dist_conflicts.items()}
                else:
                    # Reject swap, restore old_p
                    try_add_point(S, dist_conflicts, old_p)
                    S_set.add(old_p)
                    no_improve += 1
                    
            else:
                # Try a removal (to escape bad configurations)
                if not S:
                    continue
                
                old_p = random.choice(S)
                remove_point(S, dist_conflicts, old_p)
                S_set.discard(old_p)
                new_count = len(S)
                
                # Only keep removal if it doesn't reduce count below best
                # and we can potentially add more later
                if new_count < best_count_ls - 2:
                    # Restore
                    try_add_point(S, dist_conflicts, old_p)
                    S_set.add(old_p)
                    no_improve += 1
                else:
                    no_improve = 0
                    # Try to add a few points after removal
                    for _ in range(3):
                        if time.time() - start_ls > time_limit:
                            break
                        p = (random.randint(0, N-1), random.randint(0, N-1))
                        if p in S_set:
                            continue
                        if check_valid_addition(S, dist_conflicts, p):
                            try_add_point(S, dist_conflicts, p)
                            S_set.add(p)
                    
                    if len(S) > best_count_ls:
                        best_count_ls = len(S)
                        best_S_ls = S[:]
                        best_dist_ls = {k: set(v) for k, v in dist_conflicts.items()}
            
            # Cool down
            temperature *= cooling_rate
            
            # Periodically reset to best if stuck
            if no_improve > max_no_improve // 2:
                # Reset to best found so far
                S.clear()
                dist_conflicts.clear()
                S_set.clear()
                S.extend(best_S_ls)
                S_set = set(S)
                for k, v in best_dist_ls.items():
                    dist_conflicts[k] = set(v)
                no_improve = 0
                temperature = max(temperature, 2.0)
        
        # Restore best solution
        S.clear()
        dist_conflicts.clear()
        S_set.clear()
        S.extend(best_S_ls)
        for k, v in best_dist_ls.items():
            dist_conflicts[k] = set(v)
        
        return S, dist_conflicts
    
    # Run multiple trials
    trial_count = 0
    max_trials = 20
    
    while time.time() < deadline - 15 and trial_count < max_trials:
        trial_count += 1
        seed_val = trial_count * 7919 + 12345
        
        S, dist_conflicts = greedy_construct(seed_val)
        current_count = len(S)
        
        if current_count > best_count:
            best_points = S[:]
            best_count = current_count
        
        # Apply swap-based local search
        if time.time() < deadline - 20:
            time_for_ls = min(10.0, (deadline - 20) - time.time())
            if time_for_ls > 2.0:
                S_ls, dist_conflicts_ls = swap_local_search(S, dist_conflicts, time_for_ls)
                if len(S_ls) > best_count:
                    best_points = S_ls[:]
                    best_count = len(S_ls)
        
        # Early exit if we've used most of our time
        if time.time() > deadline - 25:
            break
    
    return best_points
