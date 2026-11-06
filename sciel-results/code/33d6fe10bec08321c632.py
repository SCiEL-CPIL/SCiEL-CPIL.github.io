<<<<<<< SEARCH
def _local_search_add_remove(pts, time_budget=60.0):
    """
    Local search: try to add points that fit, and if stuck, remove a point and try again.
    Uses numpy for faster checking.
    """
    start_time = time.time()
    pts = list(pts)
    pts_set = set(pts)
    
    coords = np.array([(i // N, i % N) for i in range(N * N)], dtype=np.int32)
    all_indices = list(range(N * N))
    random.shuffle(all_indices)
    
    best_pts = list(pts)
    best_size = len(pts)
    
    stuck_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try to add any point from all_indices
        for idx in all_indices:
            if time.time() - start_time > time_budget:
                break
            if idx in pts_set:
                continue
            
            # Check if idx can be added
            if len(pts) < 2:
                pts.append(idx)
                pts_set.add(idx)
                improved = True
                continue
            
            sel_coords = coords[pts]
            cx, cy = coords[idx]
            dx = sel_coords[:, 0] - cx
            dy = sel_coords[:, 1] - cy
            dist_to_cand = dx * dx + dy * dy
            
            n_sel = len(pts)
            valid = True
            for i in range(n_sel):
                if not valid:
                    break
                for j in range(i + 1, n_sel):
                    dij = (sel_coords[i, 0] - sel_coords[j, 0])**2 + (sel_coords[i, 1] - sel_coords[j, 1])**2
                    dic = dist_to_cand[i]
                    djc = dist_to_cand[j]
                    if dij == dic or dij == djc or dic == djc:
                        valid = False
                        break
            
            if valid:
                pts.append(idx)
                pts_set.add(idx)
                improved = True
        
        if improved:
            stuck_count = 0
            if len(pts) > best_size:
                best_size = len(pts)
                best_pts = list(pts)
        else:
            stuck_count += 1
            # If stuck, try removing a random point and restarting
            if stuck_count >= 5 and len(pts) > 5:
                # Remove a random point
                remove_idx = random.randint(0, len(pts) - 1)
                removed = pts.pop(remove_idx)
                pts_set.remove(removed)
                stuck_count = 0
                # Re-shuffle
                random.shuffle(all_indices)
    
    return best_pts
=======
def _local_search_add_remove(pts, time_budget=60.0):
    """
    Local search: try to add points that fit, and if stuck, remove a point and try again.
    """
    start_time = time.time()
    pts = list(pts)
    pts_set = set(pts)
    
    all_indices = list(range(N * N))
    random.shuffle(all_indices)
    
    best_pts = list(pts)
    best_size = len(pts)
    
    stuck_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try to add any point from all_indices
        for idx in all_indices:
            if time.time() - start_time > time_budget:
                break
            if idx in pts_set:
                continue
            
            # Check if idx can be added
            if _can_add_point_fast(idx, pts):
                pts.append(idx)
                pts_set.add(idx)
                improved = True
        
        if improved:
            stuck_count = 0
            if len(pts) > best_size:
                best_size = len(pts)
                best_pts = list(pts)
        else:
            stuck_count += 1
            # If stuck, try removing a random point and restarting
            if stuck_count >= 5 and len(pts) > 5:
                # Remove a random point
                remove_idx = random.randint(0, len(pts) - 1)
                removed = pts.pop(remove_idx)
                pts_set.remove(removed)
                stuck_count = 0
                # Re-shuffle
                random.shuffle(all_indices)
    
    return best_pts
>>>>>>> REPLACE

