<<<<<<< SEARCH
def _local_search(points, current_score, deadline):
    """Local search to improve the score."""
    current_points = [p for p in points]
    best_score = current_score
    best_points = [p for p in points]
    no_improve_count = 0
    max_no_improve = 100
    
    t_start = time.time()
    
    while time.time() < deadline:
        improved = False
        indices = list(range(30))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            x, y = current_points[idx]
            # Try various moves
            moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1),
                     (2, 0), (-2, 0), (0, 2), (0, -2),
                     (2, 1), (-2, -1), (2, -1), (-2, 1),
                     (1, 2), (-1, -2), (1, -2), (-1, 2)]
            
            for dx, dy in moves:
                if time.time() > deadline:
                    break
                    
                new_x, new_y = x + dx, y + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pt = (new_x, new_y)
                # Check if new_pt is already in current_points (excluding idx)
                if new_pt in current_points[:idx] + current_points[idx+1:]:
                    continue
                
                # Apply move
                old_pt = current_points[idx]
                current_points[idx] = new_pt
                
                s = _score_np(current_points)
                
                if s > best_score:
                    best_score = s
                    best_points = [p for p in current_points]
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert
                    current_points[idx] = old_pt
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        break
        
        if not improved:
            # Try a random restart from best
            if random.random() < 0.3:
                current_points = [p for p in best_points]
                no_improve_count = 0
            else:
                break
    
    return best_points, best_score
=======
def _local_search(points, current_score, deadline):
    """Local search with simulated annealing to improve the score."""
    current_points = [p for p in points]
    best_score = current_score
    best_points = [p for p in points]
    no_improve_count = 0
    max_no_improve = 200
    temperature = 2.0
    cooling = 0.995
    
    t_start = time.time()
    all_pts_set = set(current_points)
    
    while time.time() < deadline:
        improved = False
        indices = list(range(30))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            x, y = current_points[idx]
            # Try various moves including larger jumps
            moves = [(1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (-1, -1), (1, -1), (-1, 1),
                     (2, 0), (-2, 0), (0, 2), (0, -2),
                     (2, 1), (-2, -1), (2, -1), (-2, 1),
                     (1, 2), (-1, -2), (1, -2), (-1, 2),
                     (3, 0), (-3, 0), (0, 3), (0, -3),
                     (3, 1), (-3, -1), (3, -1), (-3, 1),
                     (1, 3), (-1, -3), (1, -3), (-1, 3)]
            
            for dx, dy in moves:
                if time.time() > deadline:
                    break
                    
                new_x, new_y = x + dx, y + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pt = (new_x, new_y)
                # Check if new_pt is already in current_points (excluding idx)
                if new_pt in all_pts_set and new_pt != current_points[idx]:
                    continue
                
                # Apply move
                old_pt = current_points[idx]
                all_pts_set.discard(old_pt)
                current_points[idx] = new_pt
                all_pts_set.add(new_pt)
                
                s = _score_np(current_points)
                
                if s > best_score:
                    best_score = s
                    best_points = [p for p in current_points]
                    improved = True
                    no_improve_count = 0
                elif s == best_score or (temperature > 0.1 and random.random() < np.exp((s - best_score) / temperature)):
                    # Accept non-improving move with probability (simulated annealing)
                    no_improve_count += 1
                else:
                    # Revert
                    current_points[idx] = old_pt
                    all_pts_set.discard(new_pt)
                    all_pts_set.add(old_pt)
                    no_improve_count += 1
                
                if no_improve_count >= max_no_improve:
                    break
        
        temperature *= cooling
        
        if not improved:
            # Try a random restart from best
            if random.random() < 0.3:
                current_points = [p for p in best_points]
                all_pts_set = set(current_points)
                no_improve_count = 0
                temperature = max(temperature, 0.5)
            elif no_improve_count > 500:
                break
    
    return best_points, best_score
>>>>>>> REPLACE

