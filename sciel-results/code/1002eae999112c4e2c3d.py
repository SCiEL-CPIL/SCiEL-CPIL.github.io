import time
import random
import math
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin

    # Precompute coordinates and squared distances
    # dist2[i, j] is squared distance between point i and point j
    coords = np.array([(i // N, i % N) for i in range(NUM_POINTS)], dtype=np.int32)
    xs = coords[:, 0]
    ys = coords[:, 1]
    
    # Compute full distance matrix (4096x4096 is ~16MB, fits in L2/L3 cache usually)
    diff_x = xs[:, None] - xs[None, :]
    diff_y = ys[:, None] - ys[None, :]
    dist2 = (diff_x**2 + diff_y**2).astype(np.int32)
    
    best_set = set()
    best_dist_sets = {}
    best_size = 0
    
    # Helper to check if a point p can be added to a set represented by indices
    # selected: list of indices
    # dist_sets: dict {index: set_of_squares}
    def can_add_point(p, selected_indices, dist_sets):
        if not selected_indices:
            return True, set()
        
        # Vectorized distance calculation from p to all selected
        d_to_sel = dist2[p, selected_indices]
        
        # Check 1: Distances from p to selected points must be distinct (no isosceles with apex p)
        # np.unique is fast for small arrays
        if len(np.unique(d_to_sel)) != len(d_to_sel):
            return False, set()
        
        # Check 2: For each selected q, dist(p, q) must not be in dist_sets[q]
        # This part is harder to vectorize completely because dist_sets is a list of sets.
        # However, we can iterate. The number of selected points is small (< 100 usually).
        new_dists = set()
        for i, q in enumerate(selected_indices):
            d = int(d_to_sel[i])
            if d in dist_sets[q]:
                return False, set()
            new_dists.add(d)
            
        return True, new_dists

    def add_point(p, selected_set, selected_list, dist_sets):
        selected_set.add(p)
        selected_list.append(p)
        
        # Calculate distances from p to all existing
        d_to_sel = dist2[p, selected_list[:-1]]
        new_dists = set(int(d) for d in d_to_sel)
        dist_sets[p] = new_dists
        
        # Update existing points' dist_sets
        for i, q in enumerate(selected_list[:-1]):
            dist_sets[q].add(int(d_to_sel[i]))

    def remove_point(p, selected_set, selected_list, dist_sets):
        selected_set.discard(p)
        if p in selected_list:
            idx = selected_list.index(p)
            selected_list.pop(idx)
        
        # Remove p from dist_sets
        if p in dist_sets:
            del dist_sets[p]
            
        # Remove p's distances from all other points
        for q in selected_list:
            dist_sets[q].discard(int(dist2[p, q]))

    def greedy_construction(order, time_budget):
        selected_set = set()
        selected_list = []
        dist_sets = {}
        
        for p in order:
            if time.time() > time_budget:
                break
            if p in selected_set:
                continue
            
            ok, new_dists = can_add_point(p, selected_list, dist_sets)
            if ok:
                add_point(p, selected_set, selected_list, dist_sets)
                
        return selected_set, dist_sets, selected_list

    def local_search(selected_set, dist_sets, selected_list, max_time, rng):
        """
        Randomized local search with additions and swaps.
        """
        current_size = len(selected_set)
        no_improve_count = 0
        max_no_improve = 1000 # Reset if stuck for too long? Or just keep going?
        
        while time.time() < max_time:
            # Randomly choose between Add and Swap
            # Bias towards Add if set is small, Swap if large
            if current_size < 50:
                action = 'add' if rng.random() < 0.7 else 'swap'
            else:
                action = 'add' if rng.random() < 0.3 else 'swap'
            
            if action == 'add':
                # Try to add a random unselected point
                # We can pick a random point from 0..NUM_POINTS-1
                p_add = rng.randint(0, NUM_POINTS - 1)
                
                if p_add not in selected_set:
                    ok, new_dists = can_add_point(p_add, selected_list, dist_sets)
                    if ok:
                        add_point(p_add, selected_set, selected_list, dist_sets)
                        current_size += 1
                        no_improve_count = 0
                        continue
            
            # If 'add' failed or action was 'swap', try a swap
            if current_size > 0:
                # Pick a random point to remove
                p_remove = selected_list[rng.randint(0, current_size - 1)]
                
                # Pick a random point to add
                p_add = rng.randint(0, NUM_POINTS - 1)
                
                if p_add in selected_set or p_add == p_remove:
                    continue
                
                # To check if swap is valid, we need to check p_add against (selected - {p_remove})
                # Let's create a temporary view or just check carefully.
                # Efficient check:
                # 1. Distances from p_add to all selected except p_remove must be distinct.
                # 2. For each q in selected except p_remove, dist(p_add, q) must not be in dist_sets[q] \ {dist(p_remove, q)}
                
                # Get indices of selected except p_remove
                # selected_list contains p_remove.
                # We can just iterate and skip p_remove.
                
                # First, check distinctness from p_add
                # d_to_sel_except_remove = dist2[p_add, selected_list]
                # We need to exclude the distance to p_remove.
                
                d_to_all = dist2[p_add, selected_list]
                
                # Find index of p_remove in selected_list for distance exclusion
                # This is O(N). N is small.
                try:
                    idx_remove = selected_list.index(p_remove)
                except ValueError:
                    continue # Should not happen
                
                # Create a mask or list of distances excluding the one to p_remove
                # Actually, we can just iterate.
                
                valid = True
                new_dists = set()
                
                for i, q in enumerate(selected_list):
                    if q == p_remove:
                        continue
                    
                    d = int(d_to_all[i])
                    
                    # Check distinctness from p_add
                    if d in new_dists:
                        valid = False
                        break
                    new_dists.add(d)
                    
                    # Check against dist_sets[q]
                    # dist_sets[q] contains dist(q, p_remove).
                    # If d == dist(q, p_remove), it's "free" because p_remove is gone.
                    # Otherwise, d must not be in dist_sets[q].
                    
                    d_removed = int(dist2[p_remove, q])
                    
                    if d == d_removed:
                        continue
                    
                    if d in dist_sets[q]:
                        valid = False
                        break
                
                if valid:
                    # Perform the swap
                    remove_point(p_remove, selected_set, selected_list, dist_sets)
                    add_point(p_add, selected_set, selected_list, dist_sets)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If stuck for a long time, we could try a random restart or just continue.
                    # For now, just continue.
            
            # If we get here, no move was made
            no_improve_count += 1
            if no_improve_count > 5000:
                # Break out to try a different restart? 
                # Or just keep going until time runs out.
                # Let's keep going, maybe a lucky random pick works.
                pass

        return selected_set, dist_sets, selected_list

    # Define different orderings for greedy initialization
    rng_master = random.Random(42)
    
    # Ordering 1: Random
    order1 = list(range(NUM_POINTS))
    rng_master.shuffle(order1)
    
    # Ordering 2: Spiral from center
    order2 = []
    cx, cy = N // 2, N // 2
    max_r = max(N // 2, N // 2) + 1
    visited = set()
    for r in range(max_r):
        # Square ring of radius r
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) == r:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < N and 0 <= y < N:
                        idx = y * N + x
                        if idx not in visited:
                            order2.append(idx)
                            visited.add(idx)
    # Add any remaining (should be none if N is even and center is integer, but safe)
    for i in range(NUM_POINTS):
        if i not in visited:
            order2.append(i)
            
    # Ordering 3: Corner bias (start from corners)
    order3 = []
    corners = [(0,0), (0, N-1), (N-1, 0), (N-1, N-1)]
    for cx, cy in corners:
        order3.append(cy * N + cx)
    # Rest random
    rest = [i for i in range(NUM_POINTS) if i not in order3]
    rng_master.shuffle(rest)
    order3.extend(rest)
    
    # Ordering 4: Random seed 123
    order4 = list(range(NUM_POINTS))
    random.Random(123).shuffle(order4)
    
    orderings = [order1, order2, order3, order4]
    
    # Allocate time to each trial
    # Reserve 10s for final validation
    available_time = deadline - start_time - 10.0
    time_per_trial = available_time / len(orderings)
    
    for i, order in enumerate(orderings):
        if time.time() > deadline - 10.0:
            break
            
        trial_start = time.time()
        trial_deadline = trial_start + time_per_trial
        
        # Phase 1: Greedy
        greedy_deadline = trial_start + time_per_trial * 0.2
        selected_set, dist_sets, selected_list = greedy_construction(order, greedy_deadline)
        
        # Phase 2: Local Search
        ls_deadline = trial_deadline - 1.0 # Leave 1s for bookkeeping
        selected_set, dist_sets, selected_list = local_search(
            selected_set, dist_sets, selected_list, ls_deadline, random.Random(i*1000+42)
        )
        
        # Update best
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_set = set(selected_set)
            # We don't strictly need to save dist_sets for best_set unless we do more search on it later.
            # Since we return a list of tuples, we just need the set.
            
    # Final Validation
    # It's possible (though unlikely with our checks) that a bug exists.
    # Let's do a quick check on the best set.
    if best_size > 2:
        # Convert to list for easy indexing in validation
        best_list = list(best_set)
        valid = True
        for i, p in enumerate(best_list):
            d_to_others = dist2[p, best_list]
            # Check distinct distances from p to others
            # Note: dist2[p, p] is 0. We should ignore self.
            # But our sets don't contain self.
            # In validation, we check all pairs.
            # Simpler: check all triples? No, check apex condition for all p.
            
            # Distances from p to all q != p
            # d_to_others includes dist2[p, p] = 0.
            # We want distinct values for q != p.
            # Since p is in best_list, d_to_others has a 0 at index i.
            # We can just check unique count of d_to_others excluding the 0?
            # Actually, if there is a duplicate non-zero distance, it's invalid.
            
            # Let's just use the set logic from can_add_point logic but on the whole set.
            # For each p in best_set:
            #   dists = { dist2[p, q] for q in best_set if q != p }
            #   if len(dists) != len(best_set) - 1: invalid
            
            # This is O(K^2). K is ~50-100. 100^2 = 10000 ops. Fast.
            
            for p in best_list:
                dists = set()
                for q in best_list:
                    if p == q:
                        continue
                    d = int(dist2[p, q])
                    if d in dists:
                        valid = False
                        break
                    dists.add(d)
                if not valid:
                    break
        
        if not valid:
            # Fallback to a known small valid set
            # 3 points: (0,0), (1,0), (0,2)
            # d(0,0)-(1,0)=1
            # d(0,0)-(0,2)=4
            # d(1,0)-(0,2)=1+4=5
            # Distinct.
            best_set = {0, 1, 2*N}
            best_size = 3

    return [(i // N, i % N) for i in best_set]

# EVOLVE-BLOCK-END
