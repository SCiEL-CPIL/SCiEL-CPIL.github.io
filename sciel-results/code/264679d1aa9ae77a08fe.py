import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer
    
    N = 30
    RANGE = 30
    
    # Precompute all valid points? No, too many.
    
    def gen_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for i in range(w):
            for j in range(h):
                pts.append((i + offset_x, j + offset_y))
        return pts

    # Initial configurations
    inits = []
    inits.append(gen_grid(6, 5, -2, -2))
    inits.append(gen_grid(5, 6, -2, -2))
    inits.append(gen_grid(10, 3, -4, -1))
    inits.append(gen_grid(3, 10, -1, -4))
    inits.append(gen_grid(15, 2, -6, 0))
    inits.append(gen_grid(2, 15, 0, -6))
    
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
    
    # Helper to compute full score and structures
    def init_state(points):
        area_counts = Counter()
        freq_counts = Counter()
        # area_counts[area] = count
        # freq_counts[count] = number of areas with that count
        
        p = points
        for i in range(N-2):
            xi, yi = p[i]
            for j in range(i+1, N-1):
                xj, yj = p[j]
                for k in range(j+1, N):
                    xk, yk = p[k]
                    det = abs((xj-xi)*(yk-yi) - (xk-xi)*(yj-yi))
                    if det > 0:
                        area_counts[det] += 1
        
        for area, cnt in area_counts.items():
            freq_counts[cnt] += 1
            
        max_score = max(freq_counts.keys()) if freq_counts else 0
        return area_counts, freq_counts, max_score

    # Helper to calculate delta when moving point idx from old_pt to new_pt
    # Returns: new_area_counts, new_freq_counts, new_max_score
    # We modify in place for speed? No, better to return new state or update in place carefully.
    # Given we need to rollback often, in-place with undo is tricky.
    # Let's just copy the counters? Counter is a dict. Copying 400 items is fast.
    # But we can do better: apply changes, check score, if bad, revert.
    
    def apply_move(points, idx, old_pt, new_pt, area_counts, freq_counts, max_score):
        # Apply the move to points
        points[idx] = new_pt
        
        # Update counts
        # We need to remove triangles with old_pt and add triangles with new_pt
        # Triangles are (idx, j, k)
        
        # We iterate j, k
        # To avoid double counting, j < k, and j, k != idx
        
        # Optimization: Precompute coordinates of all other points?
        # No, just loop.
        
        # We need to be careful with the order of updates to freq_counts
        # If we decrement an area and increment the same area, net change 0.
        # If we decrement A and increment B (A != B), we update two entries.
        
        # Let's collect the changes
        # delta_area_counts: {area: delta_count}
        # This is safer.
        
        delta_area_counts = {}
        
        p = points # points[idx] is already new_pt now, so we use old_pt explicitly for calculation
        
        # Remove triangles with old_pt
        for j in range(N):
            if j == idx: continue
            xj, yj = p[j]
            for k in range(j+1, N):
                if k == idx: continue
                xk, yk = p[k]
                
                # Area with old_pt
                det_old = abs((xj - old_pt[0]) * (yk - old_pt[1]) - (xk - old_pt[0]) * (yj - old_pt[1]))
                if det_old > 0:
                    delta_area_counts[det_old] = delta_area_counts.get(det_old, 0) - 1
                
                # Area with new_pt
                det_new = abs((xj - new_pt[0]) * (yk - new_pt[1]) - (xk - new_pt[0]) * (yj - new_pt[1]))
                if det_new > 0:
                    delta_area_counts[det_new] = delta_area_counts.get(det_new, 0) + 1
        
        # Apply deltas to area_counts and freq_counts
        new_max_score = max_score
        for area, delta in delta_area_counts.items():
            if delta == 0:
                continue
                
            old_count = area_counts.get(area, 0)
            new_count = old_count + delta
            
            # Update freq_counts
            # Decrement freq of old_count
            freq_counts[old_count] -= 1
            if freq_counts[old_count] == 0:
                del freq_counts[old_count]
            
            # Increment freq of new_count
            freq_counts[new_count] += 1
            
            # Update area_counts
            if new_count > 0:
                area_counts[area] = new_count
            else:
                if area in area_counts:
                    del area_counts[area]
            
            if new_count > new_max_score:
                new_max_score = new_count
        
        # It's possible max_score decreased if the area that had max_score was decreased
        # and no other area reached that level.
        # We should re-calculate max if the area that had max_score was touched?
        # Simpler: if max_score was in freq_counts and freq_counts[max_score] became 0,
        # then we need to find the new max.
        # Since we update freq_counts, we can check:
        if max_score in freq_counts and freq_counts[max_score] == 0:
             # Find new max
             # freq_counts keys are small integers.
             new_max_score = max(freq_counts.keys()) if freq_counts else 0
        elif new_max_score < max_score:
             # If we only incremented, max_score can't decrease.
             # If we decremented the max area, it might decrease.
             # The logic above handles the case where the count of 'max_score' becomes 0.
             # But what if max_score was 10, and we decremented an area from 10 to 9,
             # but there was another area with 10? Then freq_counts[10] is still > 0.
             # So max_score stays 10.
             # If we decremented the *only* area with 10, freq_counts[10] becomes 0.
             # Then we recalculate.
             # If we incremented an area to 11, new_max_score becomes 11.
             
             # So the logic holds.
             pass
             
        return new_max_score

    def revert_move(points, idx, new_pt, old_pt, area_counts, freq_counts, max_score):
        # This is the inverse of apply_move
        # We can just call apply_move with new_pt as old and old_pt as new?
        # No, the logic in apply_move assumes points[idx] is the NEW point.
        # If we revert, points[idx] is currently new_pt.
        # We want to go back to old_pt.
        # So we call apply_move(points, idx, new_pt, old_pt, ...)
        # This will remove triangles with new_pt and add triangles with old_pt.
        # This is correct.
        return apply_move(points, idx, new_pt, old_pt, area_counts, freq_counts, max_score)

    for init_pts in inits:
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
            
        current_points = init_pts[:]
        area_counts, freq_counts, current_score = init_state(current_points)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_points = current_points[:]
            
        # Local Search with Simulated Annealing
        T = 1.0
        no_improve = 0
        max_no_improve = 5000 # With fast eval, we can do more steps
        
        while time.time() < deadline - 2.0 and no_improve < max_no_improve:
            idx = random.randint(0, N-1)
            old_pt = current_points[idx]
            
            # Generate candidate moves
            # Local moves
            moves = []
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx == 0 and dy == 0: continue
                    new_pt = (old_pt[0] + dx, old_pt[1] + dy)
                    if -RANGE <= new_pt[0] <= RANGE and -RANGE <= new_pt[1] <= RANGE:
                        # Check distinctness
                        # We can check later or now. Now is safer.
                        # But O(N) check per move is slow.
                        # We'll check in apply_move? No, apply_move assumes valid.
                        # Let's check here.
                        valid = True
                        for j in range(N):
                            if j != idx and current_points[j] == new_pt:
                                valid = False
                                break
                        if valid:
                            moves.append(new_pt)
                            
            # Add some random global moves
            for _ in range(3):
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
                # Apply move
                new_score = apply_move(current_points, idx, old_pt, new_pt, area_counts, freq_counts, current_score)
                
                delta_score = new_score - current_score
                
                # Acceptance
                if delta_score > 0:
                    accept = True
                else:
                    # Simulated Annealing
                    # delta_score is usually -1, 0, 1
                    # If delta_score == 0, accept with prob 0.5?
                    # If delta_score < 0, accept with prob exp(delta_score / T)
                    if delta_score == 0:
                        accept = random.random() < 0.5
                    else:
                        accept = random.random() < (2.718 ** (delta_score / T))
                
                if accept:
                    current_score = new_score
                    old_pt = new_pt # Update old_pt for potential revert? No, we keep current_points updated.
                    improved = True
                    no_improve = 0
                    if current_score > best_overall_score:
                        best_overall_score = current_score
                        best_points = current_points[:]
                    break
                else:
                    # Revert
                    current_score = revert_move(current_points, idx, new_pt, old_pt, area_counts, freq_counts, current_score)
                    # Note: revert_move returns the score, which should be the original current_score
                    # We ignore the return value and keep the old current_score variable?
                    # Actually, revert_move updates area_counts and freq_counts back.
                    # The score should be back to what it was.
                    
            if not improved:
                no_improve += 1
                
            # Cool down
            T *= 0.99
            if T < 0.1:
                T = 0.1

    # Final validation
    if len(best_points) != N or len(set(best_points)) != N:
        return gen_grid(6, 5, -2, -2)
    for p in best_points:
        if p[0] < -30 or p[0] > 30 or p[1] < -30 or p[1] > 30:
            return gen_grid(6, 5, -2, -2)

    return best_points
