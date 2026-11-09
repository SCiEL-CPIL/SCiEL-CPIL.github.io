import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    best_set = []
    best_size = 0
    
    def is_valid_addition(p, selected, dist_from):
        px, py = p
        d_p = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_p.append(d_sq)
            
        # Check 1: P is apex. All d_p must be unique.
        if len(set(d_p)) != len(d_p):
            return False, None
            
        # Check 2: For each A, d(P,A) must not be equal to d(A,B) for any B
        for i in range(len(selected)):
            d_pa = d_p[i]
            # dist_from[i] contains distances from selected[i] to all OTHER selected points
            if d_pa in dist_from[i]:
                return False, None
                
        return True, d_p

    def add_point(p, selected, dist_from, d_p):
        selected.append(p)
        new_dist_set = set()
        for i in range(len(selected) - 1):
            new_dist_set.add(d_p[i])
        dist_from.append(new_dist_set)
        for i in range(len(selected) - 1):
            dist_from[i].add(d_p[i])

    def rebuild_dist_from(selected):
        dist_from = []
        for i in range(len(selected)):
            s = set()
            ax, ay = selected[i]
            for j in range(len(selected)):
                if i == j:
                    continue
                bx, by = selected[j]
                s.add((ax-bx)**2 + (ay-by)**2)
            dist_from.append(s)
        return dist_from

    def greedy_select(points_order):
        selected = []
        dist_from = []
        
        for p in points_order:
            valid, d_p = is_valid_addition(p, selected, dist_from)
            if valid:
                add_point(p, selected, dist_from, d_p)
                
        return selected, dist_from

    def local_improve(selected, dist_from):
        # Try to add more points
        # We maintain a set of selected points for O(1) lookup
        selected_set = set(selected)
        
        # First pass: try to add any point from all_points
        # We can be smarter by checking points that are "close" or random
        # But simple random sampling of candidates is robust.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Check a random subset of points
            # If we have K points, there are 4096-K candidates.
            # We can just iterate through a shuffled list of all points not in selected
            # But that's O(N^2). Let's sample.
            
            # A better strategy: iterate through all points once, try to add.
            # If we find one, add it and restart the scan (or continue).
            # Since N=64, N^2=4096. Checking one point is O(K).
            # Total O(N^2 * K). If K=10, that's 40k ops. Very fast.
            # So we can just do a full scan every time we add a point.
            
            # However, if we are stuck, we need to do swaps.
            
            # Let's do a full scan first for additions.
            added_any = False
            # To make it deterministic-ish but varied, we can shuffle the points list
            # But shuffling 4096 items is cheap.
            # Actually, just iterating in order is fine, or we can use a random permutation.
            # Let's use a random permutation of all points to check.
            
            check_order = all_points[:]
            random.shuffle(check_order)
            
            for p in check_order:
                if p in selected_set:
                    continue
                
                valid, d_p = is_valid_addition(p, selected, dist_from)
                if valid:
                    add_point(p, selected, dist_from, d_p)
                    selected_set.add(p)
                    improved = True
                    added_any = True
                    # Restart the loop to check again with the new set
                    # This ensures we don't miss points that became valid or were blocked before?
                    # Actually, adding a point can only make it *harder* to add others.
                    # So if a point was invalid, it remains invalid.
                    # If it was valid, we added it.
                    # So we don't need to restart. We can just continue.
                    # Wait, if we add P, it might conflict with Q that was valid before?
                    # No, if Q was valid before, it means Q didn't form an isosceles with existing.
                    # Adding P might form an isosceles with Q and some R.
                    # So Q might become invalid. But we already passed Q.
                    # So we should probably restart or just accept that we might miss some.
                    # Given the greedy nature, missing some is acceptable if we do swaps later.
                    # But restarting is safer to find a maximal set.
                    # Let's restart the scan.
                    break
            
            if not added_any:
                # No additions possible. Try swaps.
                # Pick a point to remove
                if not selected:
                    break
                
                # We want to try swapping points to free up space.
                # Strategy: Remove a random point, then try to add a random point.
                # Or: Try to swap a specific point with a specific candidate.
                
                # Let's try a few random swaps.
                swap_attempts = 0
                max_swaps = 200 # Limit swaps per stall phase to avoid infinite loops if stuck
                
                while swap_attempts < max_swaps and time.time() < deadline - 5:
                    swap_attempts += 1
                    
                    # Pick a point to remove
                    idx_to_remove = random.randint(0, len(selected) - 1)
                    p_remove = selected[idx_to_remove]
                    
                    # Pick a point to add
                    # Random from all_points not in current_set
                    p_add = None
                    attempts = 0
                    while attempts < 20:
                        rand_idx = random.randint(0, len(all_points) - 1)
                        candidate = all_points[rand_idx]
                        if candidate not in selected_set:
                            p_add = candidate
                            break
                        attempts += 1
                    
                    if p_add is None:
                        continue
                    
                    # Check if p_add is valid in selected - {p_remove}
                    # Create temp list
                    temp_selected = [p for i, p in enumerate(selected) if i != idx_to_remove]
                    temp_dist_from = rebuild_dist_from(temp_selected)
                    
                    valid, d_p_add = is_valid_addition(p_add, temp_selected, temp_dist_from)
                    
                    if valid:
                        # Perform swap
                        # Remove p_remove
                        selected.pop(idx_to_remove)
                        dist_from.pop(idx_to_remove)
                        
                        # We need to update dist_from for remaining points?
                        # Since we popped, the indices shifted.
                        # And we need to remove the distance to p_remove from their sets.
                        # This is messy.
                        # Easier: Rebuild dist_from for the new set (temp_selected + [p_add])
                        # Actually, we can just add p_add to temp_selected and rebuild.
                        
                        temp_selected.append(p_add)
                        selected = temp_selected
                        dist_from = rebuild_dist_from(selected)
                        
                        selected_set.remove(p_remove)
                        selected_set.add(p_add)
                        
                        # Now, try to add more points again (restart the improvement loop)
                        improved = True
                        break # Break out of swap loop, go back to addition loop
                
                if not improved:
                    # Stuck even after swaps
                    break

        return selected, dist_from

    # Main loop
    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_points)
        # We need a fresh copy of all_points for shuffling? 
        # random.shuffle modifies in place. That's fine.
        
        current_set, current_dist = greedy_select(all_points)
        
        # Local improvement
        current_set, current_dist = local_improve(current_set, current_dist)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # If we have a lot of time, we could try more sophisticated moves
        # But for now, let's maximize restarts.
        
    return best_set
