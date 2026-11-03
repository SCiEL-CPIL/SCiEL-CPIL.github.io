<<<<<<< SEARCH
    # ---- Local Element Refinement ----
    def local_refine(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        used_mask = 0
        row_masks = []
        for r in rows:
            mask = get_diff_mask(r)
            row_masks.append(mask)
            used_mask |= mask
            
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            
            old_val = row[pos]
            old_mask = row_masks[r_idx]
            
            # Try to move element down or up slightly
            # Focus on moving down to reduce scope
            for delta in [-1, -2, 1, 2]:
                new_val = old_val + delta
                if new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                    
                test_row = row[:]
                test_row[pos] = new_val
                
                # Check internal validity
                new_mask = 0
                ok = True
                n = len(test_row)
                for i in range(n):
                    ri = test_row[i]
                    for j in range(i + 1, n):
                        d = test_row[j] - ri
                        bit = 1 << d
                        if new_mask & bit:
                            ok = False
                            break
                        new_mask |= bit
                    if not ok:
                        break
                        
                if not ok:
                    continue
                
                # Check cross-row validity
                # Remove old row mask from used, check conflict, add new
                global_without_old = used_mask & ~old_mask
                if global_without_old & new_mask:
                    continue
                    
                # Valid move
                other_max = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row[-1])
                
                if new_scope <= current_max:
                    # Accept
                    rows[r_idx] = test_row
                    row_masks[r_idx] = new_mask
                    used_mask = global_without_old | new_mask
                    current_max = new_scope
                    
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [list(r) for r in rows]
                    break  # One change per iteration
                    
        return best_rows
=======
    # ---- Local Element Refinement with Exhaustive Pair Optimization ----
    def local_refine(rows, rng, max_iter=5000):
        rows = [list(r) for r in rows]
        used_mask = 0
        row_masks = []
        for r in rows:
            mask = get_diff_mask(r)
            row_masks.append(mask)
            used_mask |= mask
            
        current_max = scope_of(rows)
        best_max = current_max
        best_rows = [list(r) for r in rows]
        
        def try_move_single(rows, row_masks, used_mask, r_idx, pos, new_val):
            """Try moving a single element and return (success, new_mask, new_scope)."""
            row = rows[r_idx]
            if new_val <= row[pos - 1]:
                return False, 0, current_max
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                return False, 0, current_max
            
            test_row = row[:]
            test_row[pos] = new_val
            
            # Check internal validity
            new_mask = 0
            ok = True
            n = len(test_row)
            for i in range(n):
                ri = test_row[i]
                for j in range(i + 1, n):
                    d = test_row[j] - ri
                    bit = 1 << d
                    if new_mask & bit:
                        ok = False
                        break
                    new_mask |= bit
                if not ok:
                    break
            
            if not ok:
                return False, 0, current_max
            
            # Check cross-row validity
            global_without_old = used_mask & ~row_masks[r_idx]
            if global_without_old & new_mask:
                return False, 0, current_max
            
            # Calculate new scope
            other_max = 0
            for i in range(n_rows):
                if i != r_idx:
                    other_max = max(other_max, rows[i][-1])
            new_scope = max(other_max, test_row[-1])
            
            return True, new_mask, new_scope
        
        def try_move_pair(rows, row_masks, used_mask, r1, p1, v1, r2, p2, v2):
            """Try moving two elements (possibly in different rows) simultaneously."""
            row1 = rows[r1]
            row2 = rows[r2]
            
            # Check bounds for row1
            if v1 <= row1[p1 - 1]:
                return False, 0, 0, current_max
            if p1 < len(row1) - 1 and v1 >= row1[p1 + 1]:
                return False, 0, 0, current_max
            
            # Check bounds for row2 (if different row)
            if r2 != r1:
                if v2 <= row2[p2 - 1]:
                    return False, 0, 0, current_max
                if p2 < len(row2) - 1 and v2 >= row2[p2 + 1]:
                    return False, 0, 0, current_max
            
            # Build test rows
            test_row1 = row1[:]
            test_row1[p1] = v1
            new_mask1 = 0
            ok1 = True
            n1 = len(test_row1)
            for i in range(n1):
                ri = test_row1[i]
                for j in range(i + 1, n1):
                    d = test_row1[j] - ri
                    bit = 1 << d
                    if new_mask1 & bit:
                        ok1 = False
                        break
                    new_mask1 |= bit
                if not ok1:
                    break
            
            if not ok1:
                return False, 0, 0, current_max
            
            if r2 != r1:
                test_row2 = row2[:]
                test_row2[p2] = v2
                new_mask2 = 0
                ok2 = True
                n2 = len(test_row2)
                for i in range(n2):
                    ri = test_row2[i]
                    for j in range(i + 1, n2):
                        d = test_row2[j] - ri
                        bit = 1 << d
                        if new_mask2 & bit:
                            ok2 = False
                            break
                        new_mask2 |= bit
                    if not ok2:
                        break
                
                if not ok2:
                    return False, 0, 0, current_max
                
                # Check no conflict between new masks
                if new_mask1 & new_mask2:
                    return False, 0, 0, current_max
                
                # Check against other rows
                global_without_old = used_mask & ~row_masks[r1] & ~row_masks[r2]
                if global_without_old & (new_mask1 | new_mask2):
                    return False, 0, 0, current_max
                
                # Calculate new scope
                other_max = 0
                for i in range(n_rows):
                    if i != r1 and i != r2:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row1[-1], test_row2[-1])
                
                return True, new_mask1, new_mask2, new_scope
            else:
                # Same row, two positions
                # Already checked internal validity in new_mask1
                global_without_old = used_mask & ~row_masks[r1]
                if global_without_old & new_mask1:
                    return False, 0, 0, current_max
                
                other_max = 0
                for i in range(n_rows):
                    if i != r1:
                        other_max = max(other_max, rows[i][-1])
                new_scope = max(other_max, test_row1[-1])
                
                return True, new_mask1, 0, new_scope
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Single element moves
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            
            best_move = None
            best_scope = current_max
            best_mask = None
            
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                success, new_mask, new_scope = try_move_single(rows, row_masks, used_mask, r_idx, pos, new_val)
                if success and new_scope < best_scope:
                    best_move = (r_idx, pos, new_val)
                    best_scope = new_scope
                    best_mask = new_mask
            
            if best_move is not None:
                r_idx, pos, new_val = best_move
                row = rows[r_idx]
                row[pos] = new_val
                row_masks[r_idx] = best_mask
                used_mask = (used_mask & ~row_masks[r_idx]) | best_mask
                current_max = best_scope
                if current_max < best_max:
                    best_max = current_max
                    best_rows = [list(r) for r in rows]
                continue
            
            # Pair moves (exhaustive over small neighborhoods)
            r1 = rng.randint(0, n_rows - 1)
            p1 = rng.randint(1, len(rows[r1]) - 1)
            v1_old = rows[r1][p1]
            
            for d1 in [-2, -1, 1, 2]:
                v1 = v1_old + d1
                if v1 <= rows[r1][p1 - 1]:
                    continue
                if p1 < len(rows[r1]) - 1 and v1 >= rows[r1][p1 + 1]:
                    continue
                
                # Try with other rows
                r2 = rng.randint(0, n_rows - 1)
                p2 = rng.randint(1, len(rows[r2]) - 1)
                v2_old = rows[r2][p2]
                
                for d2 in [-2, -1, 1, 2]:
                    v2 = v2_old + d2
                    if r2 != r1:
                        if v2 <= rows[r2][p2 - 1]:
                            continue
                        if p2 < len(rows[r2]) - 1 and v2 >= rows[r2][p2 + 1]:
                            continue
                    
                    success, m1, m2, new_scope = try_move_pair(rows, row_masks, used_mask, r1, p1, v1, r2, p2, v2)
                    if success and new_scope < current_max:
                        # Apply the move
                        rows[r1][p1] = v1
                        if r2 != r1:
                            rows[r2][p2] = v2
                        
                        row_masks[r1] = m1
                        if r2 != r1:
                            row_masks[r2] = m2
                        
                        used_mask = (used_mask & ~row_masks[r1])
                        if r2 != r1:
                            used_mask = used_mask & ~row_masks[r2]
                        used_mask |= row_masks[r1]
                        if r2 != r1:
                            used_mask |= row_masks[r2]
                        
                        current_max = new_scope
                        if current_max < best_max:
                            best_max = current_max
                            best_rows = [list(r) for r in rows]
                        break
                else:
                    continue
                break
        
        return best_rows
>>>>>>> REPLACE

