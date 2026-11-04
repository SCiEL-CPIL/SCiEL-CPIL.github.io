def _fast_coordinate_descent_polish(points, deadline, max_iter=5000):
    pts = points.copy()
    n = N
    
    # Precompute indices for each point in the flat areas array
    # Order: i < j < k
    # The flat array is filled by:
    # for i in range(n):
    #   for j in range(i+1, n):
    #     for k in range(j+1, n):
    
    point_indices = {i: [] for i in range(n)}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                point_indices[i].append(idx)
                point_indices[j].append(idx)
                point_indices[k].append(idx)
                idx += 1
                
    def compute_areas(pts):
        areas = np.empty(idx)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[j + 1:] - pts[i]
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                count = len(cross)
                areas[idx:idx + count] = 0.5 * np.abs(cross)
                idx += count
        return areas

    areas = compute_areas(pts)
    current_min = np.min(areas)
    step = 0.02
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            
            orig = pts[i].copy()
            best_pos = orig.copy()
            best_min = current_min
            
            # Store original areas for point i to revert if needed
            orig_indices = point_indices[i]
            orig_areas = areas[orig_indices].copy()
            
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = orig + np.array([dx, dy])
                    pts[i] = new_pt
                    
                    # Compute new areas for triangles involving i
                    others = np.delete(pts, i, axis=0)
                    M = len(others)
                    if M >= 2:
                        v1 = others[:, np.newaxis, :] - pts[i]
                        v2 = others[np.newaxis, :, :] - pts[i]
                        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
                        new_areas_matrix = 0.5 * np.abs(cross)
                        mask = np.triu(np.ones((M, M), dtype=bool), k=1)
                        new_areas_flat = new_areas_matrix[mask]
                        
                        # Update areas in place
                        areas[orig_indices] = new_areas_flat
                        
                        new_min = np.min(areas)
                        
                        if new_min > best_min:
                            best_min = new_min
                            best_pos = new_pt.copy()
                            # We don't update 'areas' permanently yet, 
                            # we just track the best. 
                            # But 'areas' is modified. 
                            # If we try another move, we need to revert to 'orig' first?
                            # No, we are testing moves from 'orig'.
                            # So we should revert 'areas' to 'orig_areas' before testing the next (dx, dy)?
                            # Yes.
                            
            # After testing all moves, if we found a better one, apply it.
            # But 'areas' is currently in the state of the LAST tested move.
            # We need to restore 'areas' to 'orig_areas' if we don't accept the best move?
            # Actually, we should just recompute 'areas' for point i from 'best_pos' if we accept.
            # Or, simpler:
            # 1. Save orig_areas.
            # 2. For each move:
            #    a. Restore areas to orig_areas.
            #    b. Compute new areas for the move.
            #    c. Update areas.
            #    d. Check min.
            # 3. If best move found, keep the areas from that move?
            #    No, the areas from the best move are in 'areas' only if it was the last one tested.
            
            # Let's refactor:
            # Keep 'areas' clean (corresponding to 'orig').
            # For each candidate:
            #   compute new_areas_flat
            #   candidate_min = min( min(new_areas_flat), min(areas[not in orig_indices]) )
            #   if candidate_min > best_min:
            #       best_min = candidate_min
            #       best_pos = candidate
            #       best_new_areas_flat = new_areas_flat
            
            # This avoids modifying 'areas' during testing.
            
            # Let's re-implement the inner loop with this logic.
            
            best_new_areas_flat = None
            
            # Restore areas to orig state (in case previous iteration messed it up, though we plan to keep it clean)
            # Actually, we should ensure 'areas' corresponds to 'pts' (which is 'orig' at start of i-loop).
            # It does, because we only modify 'pts' and 'areas' together if we accept.
            
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = orig + np.array([dx, dy])
                    pts[i] = new_pt
                    
                    others = np.delete(pts, i, axis=0)
                    M = len(others)
                    if M >= 2:
                        v1 = others[:, np.newaxis, :] - pts[i]
                        v2 = others[np.newaxis, :, :] - pts[i]
                        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
                        new_areas_matrix = 0.5 * np.abs(cross)
                        mask = np.triu(np.ones((M, M), dtype=bool), k=1)
                        new_areas_flat = new_areas_matrix[mask]
                        
                        # Min of areas NOT involving i
                        # This is min(areas[not in orig_indices])
                        # We can precompute this? 
                        # min_not_involving = np.min(np.delete(areas, orig_indices))
                        # This is O(N^3). 
                        # But we can just use current_min? 
                        # No, current_min might be in orig_indices.
                        
                        # Let's just compute min of the rest.
                        # For N=14, 364-91=273 elements.
                        min_not_involving = np.min(np.delete(areas, orig_indices))
                        
                        candidate_min = min(np.min(new_areas_flat), min_not_involving)
                        
                        if candidate_min > best_min:
                            best_min = candidate_min
                            best_pos = new_pt.copy()
                            best_new_areas_flat = new_areas_flat
                            
            # Revert pts[i] to orig
            pts[i] = orig
            
            if best_min > current_min:
                pts[i] = best_pos
                areas[orig_indices] = best_new_areas_flat
                current_min = best_min
                improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
                
    return pts
