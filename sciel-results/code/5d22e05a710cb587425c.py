import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 585  # safety margin
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)  # C(11,4) = 330
    
    # Precompute bitmask for each 4-set
    masks = []
    for s in all_4sets:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # Precompute conflict: two 4-sets conflict if they share >= 3 indices
    # Use bitmask for fast intersection count
    conflict = [0] * n_sets  # bitmask of conflicting sets
    for i in range(n_sets):
        mi = masks[i]
        for j in range(i + 1, n_sets):
            mj = masks[j]
            inter = mi & mj
            # Count set bits
            c = bin(inter).count('1')
            if c >= 3:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n_sets)]
    for i in range(n_sets):
        cm = conflict[i]
        while cm:
            j = (cm & (-cm)).bit_length() - 1
            adj[i].add(j)
            cm &= cm - 1
    
    rng = random.Random(42)
    
    def greedy_random(order):
        """Greedy: process sets in given order, add if no conflict with selected."""
        selected = set()
        selected_mask = 0
        for s in order:
            if (conflict[s] & selected_mask) == 0:
                selected.add(s)
                selected_mask |= (1 << s)
        return selected
    
    def local_search(initial, time_limit=20):
        """Local search: try to add non-conflicting sets, or swap."""
        current = set(initial)
        current_mask = 0
        for s in current:
            current_mask |= (1 << s)
        best = set(current)
        best_size = len(current)
        best_mask = current_mask
        
        start = time.time()
        no_improve_count = 0
        
        while time.time() - start < time_limit:
            improved = False
            
            # Try to add a random unselected set that has no conflicts
            # First, find candidates: sets not in current and not conflicting with any in current
            # For efficiency, iterate through unselected
            unselected_list = []
            for s in range(n_sets):
                if s not in current:
                    if (conflict[s] & current_mask) == 0:
                        unselected_list.append(s)
            
            if unselected_list:
                s = rng.choice(unselected_list)
                current.add(s)
                current_mask |= (1 << s)
                improved = True
                if len(current) > best_size:
                    best = set(current)
                    best_size = len(current)
                    best_mask = current_mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Try swaps: for a random selected set, try replacing with a non-selected
                # that conflicts with exactly that one selected set
                selected_list = list(current)
                if selected_list:
                    s_remove = rng.choice(selected_list)
                    # Find candidates that conflict only with s_remove among current
                    # A candidate s_add must: not be in current, conflict[s_add] & (current_mask without s_remove) == 0
                    mask_without = current_mask & ~(1 << s_remove)
                    candidates = []
                    for s_add in range(n_sets):
                        if s_add in current:
                            continue
                        if (conflict[s_add] & mask_without) == 0:
                            candidates.append(s_add)
                    
                    if candidates:
                        s_add = rng.choice(candidates)
                        current.discard(s_remove)
                        current_mask &= ~(1 << s_remove)
                        current.add(s_add)
                        current_mask |= (1 << s_add)
                        improved = True
                        if len(current) > best_size:
                            best = set(current)
                            best_size = len(current)
                            best_mask = current_mask
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 500:
                    break
            
            if time.time() - start > time_limit:
                break
        
        return best
    
    # Multiple restarts
    best_solution = set()
    best_size = 0
    
    num_restarts = 30
    for trial in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random ordering for greedy init
        order = list(range(n_sets))
        rng.shuffle(order)
        
        # Greedy init
        selected = greedy_random(order)
        
        # Local search refinement
        improved = local_search(selected, time_limit=15)
        
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = improved
    
    # Generate points from best_solution
    points = []
    for s in best_solution:
        i, j, k, l = all_4sets[s]
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
                        
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
