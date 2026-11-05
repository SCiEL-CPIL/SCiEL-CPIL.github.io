import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer for 60s limit (evaluator kills at 90s, but we want to be safe)
    
    N = 30
    RANGE = 30
    
    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    # Initial configurations - focus on promising ones
    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    
    # Add a few random ones
    for _ in range(3):
        pts = []
        while len(pts) < N:
            p = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
            if p not in pts:
                pts.append(p)
        inits.append(pts)

    best_overall_score = 0
    best_points = inits[0]
    
    # Helper to compute full score and structures using NumPy
    def get_initial_state(points):
        pts = np.array(points)
        # Compute all triangle areas
        # i < j < k
        # We can vectorize by creating arrays of indices
        idx_i, idx_j, idx_k = np.indices((N, N, N))
        mask = (idx_i < idx_j) & (idx_j < idx_k)
        i_idx = idx_i[mask]
        j_idx = idx_j[mask]
        k_idx = idx_k[mask]
        
        pi = pts[i_idx]
        pj = pts[j_idx]
        pk = pts[k_idx]
        
        # det = (xj-xi)*(yk-yi) - (xk-xi)*(yj-yi)
        # Using cross product of (pj-pi) and (pk-pi)
        v1 = pj - pi
        v2 = pk - pi
        dets = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        # Filter out zero areas
        dets = dets[dets > 0]
        
        area_counts = Counter(dets.tolist())
        freq_counts = Counter(area_counts.values())
        max_score = max(freq_counts.keys()) if freq_counts else 0
        
        return area_counts, freq_counts, max_score

    # Helper to evaluate a move efficiently
    # We update the area_counts and freq_counts for a single point move
    def evaluate_move(points, idx, new_pt, area_counts, freq_counts, current_score):
        # points is a list of tuples
        # We need to calculate the change in area counts
        # Triangles involving idx: (idx, j, k) for all j != k, j,k != idx
        # Actually, to avoid double counting and maintain consistency with init_state,
        # we consider all pairs (j, k) from the other N-1 points.
        # In init_state, we only counted i<j<k.
        # When we move point idx, we need to update the counts for all triangles containing idx.
        # Let's just recalculate the affected triangles and update the counters.
        
        old_pt = points[idx]
        other_indices = [i for i in range(N) if i != idx]
        
        # We need to iterate over all pairs (j, k) from other_indices
        # And compute det_old and det_new
        
        # To be consistent with the "i<j<k" counting in init_state:
        # If idx is the smallest index, triangles are (idx, j, k) with j<k
        # If idx is the middle index, triangles are (j, idx, k) with j<idx<k
        # If idx is the largest index, triangles are (j, k, idx) with j<k<idx
        
        # Instead of complex logic, we can just compute the determinant for all pairs (j,k) in other_indices
        # and determine if they were counted in the original state.
        # Actually, it's easier to just compute the delta in the *multiset* of areas.
        # The init_state counts each unique set of 3 points once.
        # So for a move of point idx, we look at all sets {idx, j, k} where j,k != idx.
        
        deltas = {}
        
        # To speed this up, we can vectorize the calculation of dets for all j,k pairs
        other_pts = np.array([points[i] for i in other_indices])
        # Pairs of other_pts
        # There are (N-1)*(N-2)/2 such pairs
        # Let's just loop in Python for simplicity if N is small, but N=30 -> 435 pairs.
        # 435 iterations per move is fast enough if we do it in Python?
        # Let's try to vectorize it.
        
        j_idx_arr, k_idx_arr = np.indices((len(other_indices), len(other_indices)))
        mask = j_idx_arr < k_idx_arr
        j_idx = j_idx_arr[mask]
        k_idx = k_idx_arr[mask]
        
        pj = other_pts[j_idx]
        pk = other_pts[k_idx]
        
        old_pt_arr = np.array(old_pt)
        new_pt_arr = np.array(new_pt)
        
        v1_old = pj - old_pt_arr
        v2_old = pk - old_pt_arr
        dets_old = np.abs(v1_old[:, 0] * v2_old[:, 1] - v1_old[:, 1] * v2_old[:, 0])
        
        v1_new = pj - new_pt_arr
        v2_new = pk - new_pt_arr
        dets_new = np.abs(v1_new[:, 0] * v2_new[:, 1] - v1_new[:, 1] * v2_new[:, 0])
        
        # Now we need to update area_counts
        # For each det_old > 0, decrement
        # For each det_new > 0, increment
        
        # We can use Counter for this
        # But we need to return the new max_score
        
        # Let's copy the counters to avoid modifying the original if we reject
        # But for efficiency, we might modify in place and revert if rejected.
        # However, reverting Counter updates is tricky.
        # Let's just calculate the new counters from scratch? No, too slow.
        
        # Let's modify a copy of the counters.
        # Actually, let's just return the delta and let the caller handle it?
        # Or better, let's just calculate the new max score directly.
        
        # We'll create a temporary dictionary of changes
        # But we need to update freq_counts as well.
        
        # Let's try a different approach:
        # 1. Calculate new area_counts for the affected triangles.
        # 2. Update the global area_counts.
        # 3. Update freq_counts based on the changes in area_counts.
        # 4. Determine new max score.
        
        # Since we are doing a local search, we can just calculate the score change.
        # But score is max(freq_counts.keys()).
        
        # Let's update area_counts and freq_counts in a trial run.
        # We'll pass in the counters and modify them, then revert if needed.
        # This is risky if we don't revert correctly.
        
        # Alternative: Just recompute the score from scratch? No, O(N^3).
        
        # Let's stick to the delta approach but do it carefully.
        
        # We'll create a list of (area, delta)
        # For each old det > 0: (det, -1)
        # For each new det > 0: (det, +1)
        
        # Combine these deltas
        delta_counts = {}
        for d in dets_old:
            if d > 0:
                d = int(d)
                delta_counts[d] = delta_counts.get(d, 0) - 1
        for d in dets_new:
            if d > 0:
                d = int(d)
                delta_counts[d] = delta_counts.get(d, 0) + 1
        
        # Now apply these deltas to area_counts and freq_counts
        # We need to track the changes to freq_counts to update the max score
        
        new_max_score = current_score
        # We need to know which freq counts are changing
        # Let's just update the counters and then find the new max
        
        # To allow reverting, we should not modify the original counters directly in this function
        # unless we have a revert mechanism.
        # Let's return the new area_counts and freq_counts? That's expensive (copying).
        
        # Let's modify the original counters and return a "revert" function or the previous state.
        # Actually, let's just modify them and if the move is rejected, we revert.
        # We'll store the old values in a list to revert.
        
        revert_info = []
        
        for area, delta in delta_counts.items():
            if delta == 0:
                continue
                
            old_count = area_counts.get(area, 0)
            new_count = old_count + delta
            
            # Update freq_counts
            if old_count > 0:
                revert_info.append(('freq', old_count, 1))
                freq_counts[old_count] -= 1
                if freq_counts[old_count] == 0:
                    del freq_counts[old_count]
            
            if new_count > 0:
                revert_info.append(('freq', new_count, -1))
                freq_counts[new_count] += 1
                area_counts[area] = new_count
                if new_count > new_max_score:
                    new_max_score = new_count
            else:
                if area in area_counts:
                    revert_info.append(('area', area, 0)) # marker to delete
                    del area_counts[area]
        
        # If the current max_score is no longer in freq_counts, we need to recalculate
        # This is a potential bug source.
        # Let's just recalculate max if freq_counts.get(current_score, 0) == 0
        if freq_counts.get(current_score, 0) == 0:
            new_max_score = max(freq_counts.keys()) if freq_counts else 0
        
        return new_max_score, revert_info

    def revert_move(points, idx, old_pt, new_pt, area_counts, freq_counts, revert_info):
        # Revert the counters
        for item in revert_info:
            if item[0] == 'freq':
                _, count, sign = item
                if sign == 1: # we decremented, so increment back
                    freq_counts[count] = freq_counts.get(count, 0) + 1
                else: # we incremented, so decrement back
                    freq_counts[count] -= 1
                    if freq_counts[count] == 0:
                        del freq_counts[count]
            elif item[0] == 'area':
                _, area, _ = item
                # We deleted it, so we need to restore it?
                # Wait, if we deleted it, it means new_count was 0.
                # So old_count was 0? No, old_count was > 0.
                # So we need to restore area_counts[area] = old_count.
                # But we didn't store old_count in revert_info for 'area' deletion.
                # We need to fix the revert_info structure.
                pass
        
        # This revert logic is getting complicated and error-prone.
        # Let's simplify: Just recalculate the state from scratch after each accepted move?
        # No, that's O(N^3).
        
        # Let's try a different strategy:
        # Keep the points array.
        # For each move, calculate the new score by updating the counters.
        # If accepted, keep the counters.
        # If rejected, we need to revert.
        
        # To make revert safe, let's store the old values of the counters in revert_info.
        
        pass # We will implement a cleaner version below

    def local_search(initial_points, time_limit, T_init=1.0):
        current_points = initial_points[:]
        area_counts, freq_counts, current_score = get_initial_state(current_points)
        
        best_local_score = current_score
        best_local_points = current_points[:]
        
        T = T_init
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < time_limit and no_improve < max_no_improve:
            idx = random.randint(0, N-1)
            old_pt = current_points[idx]
            
            # Generate candidate moves
            moves = []
            
            # Local moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                        # Check distinctness
                        valid = True
                        for j in range(N):
                            if j != idx and current_points[j] == new_pt:
                                valid = False
                                break
                        if valid:
                            moves.append(new_pt)
                        
            # Add some random global moves
            for _ in range(2):
                new_pt = (random.randint(-RANGE, RANGE), random.randint(-RANGE, RANGE))
                valid = True
                for j in range(N):
                    if j != idx and current_points[j] == new_pt:
                        valid = False
                        break
                if valid:
                    moves.append(new_pt)
                    
            if not moves:
                no_improve += 1
                continue
                
            random.shuffle(moves)
            
            improved = False
            for new_pt in moves:
                # Evaluate move
                # We need a robust way to update and revert
                # Let's just try the move, update counters, check score.
                # If we accept, we keep the counters.
                # If we reject, we revert the counters.
                
                # To do this cleanly, let's store the old state of the counters
                # for the keys that are modified.
                
                # We'll call a helper that returns the new score and the revert info
                # We need to fix the evaluate_move function to return proper revert info.
                
                # Let's implement a simpler version:
                # 1. Save old counts for all areas that will change.
                # 2. Apply changes.
                # 3. Calculate new max score.
                # 4. If reject, restore old counts.
                
                # Let's identify the areas that change
                other_indices = [i for i in range(N) if i != idx]
                other_pts = np.array([current_points[i] for i in other_indices])
                j_idx_arr, k_idx_arr = np.indices((len(other_indices), len(other_indices)))
                mask = j_idx_arr < k_idx_arr
                j_idx = j_idx_arr[mask]
                k_idx = k_idx_arr[mask]
                
                pj = other_pts[j_idx]
                pk = other_pts[k_idx]
                
                old_pt_arr = np.array(old_pt)
                new_pt_arr = np.array(new_pt)
                
                v1_old = pj - old_pt_arr
                v2_old = pk - old_pt_arr
                dets_old = np.abs(v1_old[:, 0] * v2_old[:, 1] - v1_old[:, 1] * v2_old[:, 0])
                
                v1_new = pj - new_pt_arr
                v2_new = pk - new_pt_arr
                dets_new = np.abs(v1_new[:, 0] * v2_new[:, 1] - v1_new[:, 1] * v2_new[:, 0])
                
                # Collect deltas
                delta_counts = {}
                for d in dets_old:
                    if d > 0:
                        d = int(d)
                        delta_counts[d] = delta_counts.get(d, 0) - 1
                for d in dets_new:
                    if d > 0:
                        d = int(d)
                        delta_counts[d] = delta_counts.get(d, 0) + 1
                
                # Prepare revert info
                revert_info = []
                new_max_score = current_score
                
                for area, delta in delta_counts.items():
                    if delta == 0:
                        continue
                    
                    old_count = area_counts.get(area, 0)
                    new_count = old_count + delta
                    
                    # Store old state for revert
                    # We need to revert area_counts[area] and freq_counts[old_count], freq_counts[new_count]
                    # If old_count > 0, freq_counts[old_count] decreases
                    # If new_count > 0, freq_counts[new_count] increases
                    
                    revert_info.append({
                        'area': area,
                        'old_count': old_count,
                        'new_count': new_count,
                        'old_freq_old': freq_counts.get(old_count, 0) if old_count > 0 else 0,
                        'old_freq_new': freq_counts.get(new_count, 0) if new_count > 0 else 0
                    })
                    
                    # Apply changes
                    if old_count > 0:
                        freq_counts[old_count] -= 1
                        if freq_counts[old_count] == 0:
                            del freq_counts[old_count]
                    
                    if new_count > 0:
                        freq_counts[new_count] = freq_counts.get(new_count, 0) + 1
                        area_counts[area] = new_count
                        if new_count > new_max_score:
                            new_max_score = new_count
                    else:
                        if area in area_counts:
                            del area_counts[area]
                
                # Recalculate max if current max is gone
                if freq_counts.get(current_score, 0) == 0:
                    new_max_score = max(freq_counts.keys()) if freq_counts else 0
                
                delta_score = new_max_score - current_score
                
                # Acceptance
                if delta_score > 0:
                    accept = True
                else:
                    if delta_score == 0:
                        accept = random.random() < 0.3
                    else:
                        accept = random.random() < (2.718 ** (delta_score / T))
                
                if accept:
                    current_points[idx] = new_pt
                    current_score = new_max_score
                    improved = True
                    no_improve = 0
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_points = current_points[:]
                    break
                else:
                    # Revert
                    for info in revert_info:
                        area = info['area']
                        old_count = info['old_count']
                        new_count = info['new_count']
                        
                        # Restore area_counts
                        if old_count > 0:
                            area_counts[area] = old_count
                        else:
                            if area in area_counts:
                                del area_counts[area]
                        
                        # Restore freq_counts
                        if old_count > 0:
                            freq_counts[old_count] = info['old_freq_old']
                        if new_count > 0:
                            freq_counts[new_count] = info['old_freq_new']
                    
                    # Revert points
                    current_points[idx] = old_pt
                    
            if not improved:
                no_improve += 1
                
            # Cool down
            T *= 0.995
            if T < 0.1:
                T = 0.1

        return best_local_points, best_local_score

    # Main loop over initial configurations
    num_inits = len(inits)
    time_budget = deadline - start_time - 5.0
    time_per_init = time_budget / num_inits
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        # Validate init
        if len(set(init_pts)) != N:
            continue
        valid = True
        for p in init_pts:
            if p[0] < -RANGE or p[0] > RANGE or p[1] < -RANGE or p[1] > RANGE:
                valid = False
                break
        if not valid:
            continue
            
        # Run local search
        limit = min(deadline - 5.0, time.time() + time_per_init)
        best_pts, best_score = local_search(init_pts, limit)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_points = best_pts[:]

    # Final polish
    if time.time() < deadline - 3.0 and best_overall_score > 0:
        final_limit = deadline - 3.0
        final_pts, final_score = local_search(best_points, final_limit, T_init=0.5)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_points = final_pts[:]

    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
