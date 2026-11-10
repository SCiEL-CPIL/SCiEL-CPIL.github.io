import time
import random
import numpy as np
from collections import defaultdict

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute all squared distances between grid points
    xs = np.arange(N)
    ys = np.arange(N)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Compute squared distance matrix
    diff_x = x_flat[:, None] - x_flat[None, :]
    diff_y = y_flat[:, None] - y_flat[None, :]
    dist2 = (diff_x**2 + diff_y**2).astype(np.int32)
    
    best_set = set()
    rng = random.Random(42)
    
    def greedy(order):
        selected = []
        selected_set = set()
        # For each selected point q, track the set of squared distances from q to other selected points
        dist_sets = {}
        
        for p in order:
            if p in selected_set:
                continue
            
            if not selected:
                can_add = True
                new_dists = set()
            else:
                # Get distances from p to all selected points
                d_to_sel = dist2[p, selected]
                
                # Check 1: No duplicate distances from p (apex at p)
                if len(set(d_to_sel)) != len(d_to_sel):
                    continue
                
                # Check 2: For each selected point q, the distance d(p,q) must not already be in dist_sets[q]
                can_add = True
                for i, q in enumerate(selected):
                    d = int(d_to_sel[i])
                    if d in dist_sets[q]:
                        can_add = False
                        break
                
                if not can_add:
                    continue
                new_dists = set(int(d) for d in d_to_sel)
            
            # Add p
            selected.append(p)
            selected_set.add(p)
            dist_sets[p] = new_dists
            
            # Update dist_sets for existing points
            for q in selected[:-1]:  # all except p
                dist_sets[q].add(int(dist2[p, q]))
        
        return selected_set, dist_sets, selected

    def local_search(initial_set, initial_dists, initial_list, max_iters=5000):
        """
        Perform local search starting from the initial state.
        """
        selected_set = set(initial_set)
        dist_sets = {p: set(d) for p, d in initial_dists.items()}
        selected_list = list(initial_list)
        
        # We need to be able to remove and add efficiently.
        # For removal, we just update dist_sets.
        
        best_local = len(selected_set)
        best_local_set = set(selected_set)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Strategy: Try to add a point. If fail, try to swap.
            # To be efficient, we pick random candidates.
            
            # 1. Try Add
            # Pick a random point not in set
            # We can precompute a list of unselected points, but it changes.
            # Just pick a random index.
            p_add = rng.randint(0, NUM_POINTS - 1)
            
            if p_add not in selected_set:
                # Check validity
                d_to_sel = dist2[p_add, selected_list]
                
                # Check distinctness from p_add
                if len(set(d_to_sel)) == len(d_to_sel):
                    # Check against existing dist_sets
                    valid = True
                    for i, q in enumerate(selected_list):
                        d = int(d_to_sel[i])
                        if d in dist_sets[q]:
                            valid = False
                            break
                    
                    if valid:
                        # Accept add
                        selected_set.add(p_add)
                        selected_list.append(p_add)
                        
                        new_dists = set(int(d) for d in d_to_sel)
                        dist_sets[p_add] = new_dists
                        
                        for q in selected_list[:-1]:
                            dist_sets[q].add(int(dist2[p_add, q]))
                        
                        if len(selected_set) > best_local:
                            best_local = len(selected_set)
                            best_local_set = set(selected_set)
                            # Update global best if better
                            if len(selected_set) > len(best_set):
                                best_set = set(selected_set)
                        continue
            
            # 2. Try Swap (Remove one, Add one)
            # Pick a random point to remove
            if selected_list:
                idx_remove = rng.randint(0, len(selected_list) - 1)
                p_remove = selected_list[idx_remove]
                
                # Pick a random point to add
                p_add = rng.randint(0, NUM_POINTS - 1)
                
                if p_add not in selected_set and p_add != p_remove:
                    # Temporarily remove p_remove to check if p_add can be added to the rest
                    # We need to check p_add against selected_set \ {p_remove}
                    
                    # Optimization: Instead of physically removing, we can just check
                    # distances to all selected except p_remove.
                    
                    # Get indices of selected_list except p_remove
                    # This is O(N) to build the list, which is fine.
                    
                    sel_except = [p for p in selected_list if p != p_remove]
                    
                    if not sel_except:
                        # If removing leaves empty set, any add is valid
                        # But we are swapping, so we add p_add
                        # This case is rare in local search on large sets
                        pass
                    
                    d_to_sel = dist2[p_add, sel_except]
                    
                    # Check distinctness from p_add
                    if len(set(d_to_sel)) == len(d_to_sel):
                        # Check against existing dist_sets, but we must exclude the distance to p_remove
                        # dist_sets[q] contains distance to p_remove.
                        # We need to check if d(p_add, q) is in dist_sets[q] \ {d(p_remove, q)}
                        
                        valid = True
                        for i, q in enumerate(sel_except):
                            d = int(d_to_sel[i])
                            # The distance d(p_remove, q) is in dist_sets[q]
                            # If d == d(p_remove, q), it's allowed because p_remove is gone.
                            # If d != d(p_remove, q), it must not be in dist_sets[q].
                            
                            # Actually, simpler:
                            # d is invalid if it is in dist_sets[q] AND it is not the distance to the removed point.
                            # But wait, dist_sets[q] contains distances to ALL other points in S.
                            # One of them is p_remove.
                            # So if d == d(p_remove, q), it's "free".
                            # If d != d(p_remove, q), it must not be in dist_sets[q].
                            
                            d_removed = int(dist2[p_remove, q])
                            
                            if d == d_removed:
                                continue
                            
                            if d in dist_sets[q]:
                                valid = False
                                break
                        
                        if valid:
                            # Perform Swap
                            # Remove p_remove
                            selected_set.remove(p_remove)
                            selected_list.remove(p_remove)
                            
                            # Update dist_sets for remaining points
                            for q in selected_list:
                                dist_sets[q].discard(int(dist2[p_remove, q]))
                            
                            del dist_sets[p_remove]
                            
                            # Add p_add
                            selected_set.add(p_add)
                            selected_list.append(p_add)
                            
                            # Calculate new dists for p_add
                            d_to_sel_final = dist2[p_add, selected_list[:-1]] # selected_list[:-1] is the set before adding p_add? No, we removed p_remove, so len is same.
                            # Actually, selected_list now contains p_add at the end.
                            # The other points are selected_list[:-1]
                            
                            new_dists = set(int(d) for d in d_to_sel_final)
                            dist_sets[p_add] = new_dists
                            
                            for q in selected_list[:-1]:
                                dist_sets[q].add(int(dist2[p_add, q]))
                            
                            if len(selected_set) > best_local:
                                best_local = len(selected_set)
                                best_local_set = set(selected_set)
                                if len(selected_set) > len(best_set):
                                    best_set = set(selected_set)

        return selected_set

    # 1. Run Greedy with different orders to get good starting points
    orders = []
    
    # Random orders
    for _ in range(5):
        o = list(range(NUM_POINTS))
        rng.shuffle(o)
        orders.append(o)
    
    # Spiral order
    def spiral_order():
        result = []
        left, right = 0, N - 1
        top, bottom = 0, N - 1
        while left <= right and top <= bottom:
            for x in range(left, right + 1):
                result.append(top * N + x)
            top += 1
            for y in range(top, bottom + 1):
                result.append(y * N + right)
            right -= 1
            if top <= bottom:
                for x in range(right, left - 1, -1):
                    result.append(bottom * N + x)
                bottom -= 1
            if left <= right:
                for y in range(bottom, top - 1, -1):
                    result.append(y * N + left)
                left += 1
        return result
    
    orders.append(spiral_order())
    
    # Checkerboard
    def checkerboard_order():
        result = []
        for y in range(N):
            for x in range(N):
                if (x + y) % 2 == 0:
                    result.append(y * N + x)
        for y in range(N):
            for x in range(N):
                if (x + y) % 2 == 1:
                    result.append(y * N + x)
        return result
    
    orders.append(checkerboard_order())
    
    # Run greedy on each order
    for o in orders:
        if time.time() > deadline:
            break
        s, d, l = greedy(o)
        if len(s) > len(best_set):
            best_set = s
    
    # 2. Run Local Search on the best set found so far
    # We need to reconstruct dist_sets and list for the best_set
    # To do this efficiently, we can just re-run greedy on the order that produced best_set?
    # Or we can just run local search on the current best_set by rebuilding the state.
    
    # Let's rebuild state for best_set
    if best_set:
        # Sort points for deterministic list?
        best_list = list(best_set)
        best_dists = {}
        for p in best_list:
            best_dists[p] = set()
        for i, p in enumerate(best_list):
            for j, q in enumerate(best_list):
                if i != j:
                    best_dists[p].add(int(dist2[p, q]))
        
        # Run local search
        # Use a large number of iterations, but guarded by time
        # The local search function above doesn't take a deadline, so we wrap it or modify it.
        
        # Let's run a loop of local search restarts
        # Each restart starts from the current best and does a random walk.
        
        # We will run local search until deadline
        # We need a version of local_search that checks time.
        
        def local_search_with_deadline(initial_set, initial_dists, initial_list):
            selected_set = set(initial_set)
            dist_sets = {p: set(d) for p, d in initial_dists.items()}
            selected_list = list(initial_list)
            
            best_local = len(selected_set)
            best_local_set = set(selected_set)
            
            while time.time() < deadline:
                # Try Add
                p_add = rng.randint(0, NUM_POINTS - 1)
                
                if p_add not in selected_set:
                    d_to_sel = dist2[p_add, selected_list]
                    
                    if len(set(d_to_sel)) == len(d_to_sel):
                        valid = True
                        for i, q in enumerate(selected_list):
                            d = int(d_to_sel[i])
                            if d in dist_sets[q]:
                                valid = False
                                break
                        
                        if valid:
                            selected_set.add(p_add)
                            selected_list.append(p_add)
                            
                            new_dists = set(int(d) for d in d_to_sel)
                            dist_sets[p_add] = new_dists
                            
                            for q in selected_list[:-1]:
                                dist_sets[q].add(int(dist2[p_add, q]))
                            
                            if len(selected_set) > best_local:
                                best_local = len(selected_set)
                                best_local_set = set(selected_set)
                                if len(selected_set) > len(best_set):
                                    best_set = set(selected_set)
                            continue
                
                # Try Swap
                if selected_list:
                    idx_remove = rng.randint(0, len(selected_list) - 1)
                    p_remove = selected_list[idx_remove]
                    
                    p_add = rng.randint(0, NUM_POINTS - 1)
                    
                    if p_add not in selected_set and p_add != p_remove:
                        sel_except = [p for p in selected_list if p != p_remove]
                        
                        d_to_sel = dist2[p_add, sel_except]
                        
                        if len(set(d_to_sel)) == len(d_to_sel):
                            valid = True
                            for i, q in enumerate(sel_except):
                                d = int(d_to_sel[i])
                                d_removed = int(dist2[p_remove, q])
                                
                                if d == d_removed:
                                    continue
                                
                                if d in dist_sets[q]:
                                    valid = False
                                    break
                            
                            if valid:
                                selected_set.remove(p_remove)
                                selected_list.remove(p_remove)
                                
                                for q in selected_list:
                                    dist_sets[q].discard(int(dist2[p_remove, q]))
                                
                                del dist_sets[p_remove]
                                
                                selected_set.add(p_add)
                                selected_list.append(p_add)
                                
                                d_to_sel_final = dist2[p_add, selected_list[:-1]]
                                new_dists = set(int(d) for d in d_to_sel_final)
                                dist_sets[p_add] = new_dists
                                
                                for q in selected_list[:-1]:
                                    dist_sets[q].add(int(dist2[p_add, q]))
                                
                                if len(selected_set) > best_local:
                                    best_local = len(selected_set)
                                    best_local_set = set(selected_set)
                                    if len(selected_set) > len(best_set):
                                        best_set = set(selected_set)
            
            return selected_set

        local_search_with_deadline(best_set, best_dists, best_list)

    return [(p // N, p % N) for p in best_set]
