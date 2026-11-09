<<<<<<< SEARCH
    # Reinsertion pass: try to fit left-out boxes into gaps
    if best_result is not None:
        placed_boxes = [(i, best_result[i][1], best_result[i][2], best_result[i][3],
                        best_result[i][4], best_result[i][5], best_result[i][6])
                       for i in range(n) if best_result[i][0] == 0]
        
        for pi, pos_gen in enumerate(pos_generators):
            if time.time() - start_time > DEADLINE:
                break
            for si, sort_key in enumerate(sort_strategies[:3]):  # Use top 3 sort strategies
                if time.time() - start_time > DEADLINE:
                    break
                # Try to insert left-out boxes
                left_out = [i for i in range(n) if best_result[i][0] == -1]
                left_out.sort(key=lambda i: sort_key(boxes[i]), reverse=True)
                
                new_placed = list(placed_boxes)
                new_vol = best_vol
                improved = False
                
                for idx in left_out:
                    best_placement = None
                    best_score = float('inf')
                    
                    for ow, oh, od in box_orientations[idx]:
                        candidate_positions = pos_gen(new_placed, ow, oh, od)
                        
                        for cx, cy, cz in candidate_positions:
                            if cx + ow > W or cy + oh > H or cz + od > D:
                                continue
                            
                            valid = True
                            for _, px, py, pz, pw, ph, pd in new_placed:
                                if overlaps(cx, cy, cz, ow, oh, od, px, py, pz, pw, ph, pd):
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            score = cx + cy + cz
                            if score < best_score:
                                best_score = score
                                best_placement = (idx, cx, cy, cz, ow, oh, od)
                    
                    if best_placement:
                        new_placed.append(best_placement)
                        new_vol += best_placement[4] * best_placement[5] * best_placement[6]
                        improved = True
                
                if improved and new_vol > best_vol:
                    best_vol = new_vol
                    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
                    for idx, x, y, z, w, h, d in new_placed:
                        result[idx] = (0, x, y, z, w, h, d)
                    best_result = result
=======
    # Reinsertion pass: try to fit left-out boxes into gaps using voxel grid
    if best_result is not None:
        placed_boxes = [(i, best_result[i][1], best_result[i][2], best_result[i][3],
                        best_result[i][4], best_result[i][5], best_result[i][6])
                       for i in range(n) if best_result[i][0] == 0]
        
        # Build occupancy grid from current placement
        occupancy[:] = 0
        for _, x, y, z, w, h, d in placed_boxes:
            mark_occupied(x, y, z, w, h, d)
        
        for pi, pos_gen in enumerate(pos_generators):
            if time.time() - start_time > DEADLINE:
                break
            for si, sort_key in enumerate(sort_strategies[:5]):  # Use top 5 sort strategies
                if time.time() - start_time > DEADLINE:
                    break
                # Try to insert left-out boxes
                left_out = [i for i in range(n) if best_result[i][0] == -1]
                left_out.sort(key=lambda i: sort_key(boxes[i]), reverse=True)
                
                new_placed = list(placed_boxes)
                new_vol = best_vol
                improved = False
                
                for idx in left_out:
                    best_placement = None
                    best_score = float('inf')
                    
                    for ow, oh, od in box_orientations[idx]:
                        candidate_positions = pos_gen(new_placed, ow, oh, od)
                        
                        for cx, cy, cz in candidate_positions:
                            if cx + ow > W or cy + oh > H or cz + od > D:
                                continue
                            
                            if check_collision(cx, cy, cz, ow, oh, od):
                                continue
                            
                            score = cx + cy + cz
                            if score < best_score:
                                best_score = score
                                best_placement = (idx, cx, cy, cz, ow, oh, od)
                    
                    if best_placement:
                        new_placed.append(best_placement)
                        mark_occupied(best_placement[1], best_placement[2], best_placement[3],
                                     best_placement[4], best_placement[5], best_placement[6])
                        new_vol += best_placement[4] * best_placement[5] * best_placement[6]
                        improved = True
                
                if improved and new_vol > best_vol:
                    best_vol = new_vol
                    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
                    for idx, x, y, z, w, h, d in new_placed:
                        result[idx] = (0, x, y, z, w, h, d)
                    best_result = result
                
                # Reset grid for next iteration
                occupancy[:] = 0
                for _, x, y, z, w, h, d in placed_boxes:
                    mark_occupied(x, y, z, w, h, d)
>>>>>>> REPLACE

