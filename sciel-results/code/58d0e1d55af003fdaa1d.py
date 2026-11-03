<<<<<<< SEARCH
    # 4. Final Compaction
    final_rows = [list(r) for r in best_rows_final]
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)
    
    # Try to reduce scope by moving elements to smaller positions
    for _ in range(100):
        if time.time() > deadline - 1:
            break
        row_idx = random.randint(0, n_rows - 1)
        row = final_rows[row_idx]
        if len(row) < 2:
            continue
        last = row[-1]
        if last <= 1:
            continue
        
        removed = row.pop()
        for a in row:
            used_set.discard(removed - a)
        
        found = False
        # Try to find a smaller position for the removed element
        for x in range(last - 1, 0, -1):
            if x in row:
                continue
            valid = True
            for a in row:
                d = abs(x - a)
                if d == 0 or d in used_set:
                    valid = False
                    break
            if valid:
                pos = 0
                while pos < len(row) and row[pos] < x:
                    pos += 1
                row.insert(pos, x)
                for a in row:
                    if a != x:
                        used_set.add(abs(x - a))
                found = True
                break
        
        if not found:
            row.append(removed)
            row.sort()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_set.add(row[j] - row[i])
    
    final_scope = max(max(r) for r in final_rows)
    
    # Validate final result
    if is_valid(final_rows) and final_scope < best_scope_final:
        best_rows_final = final_rows
        best_scope_final = final_scope
=======
    # 4. Final Compaction - more aggressive, multiple passes
    final_rows = [list(r) for r in best_rows_final]
    used_set = get_used_diffs(final_rows)
    final_scope = max(max(r) for r in final_rows)
    
    # Multiple compaction passes
    for pass_num in range(5):
        if time.time() > deadline - 1:
            break
        improved = False
        for _ in range(200):
            if time.time() > deadline - 1:
                break
            row_idx = random.randint(0, n_rows - 1)
            row = final_rows[row_idx]
            if len(row) < 2:
                continue
            # Pick a random element (not the first)
            p_idx = random.randint(1, len(row) - 1)
            removed = row[p_idx]
            if removed <= 1:
                continue
            
            # Remove from row and update used_set
            row.pop(p_idx)
            for a in row:
                used_set.discard(abs(removed - a))
            
            found = False
            # Try to find a smaller position for the removed element
            # Scan from 1 upward, prefer smallest valid position
            for x in range(1, removed):
                if x in row:
                    continue
                valid = True
                for a in row:
                    d = abs(x - a)
                    if d == 0 or d in used_set:
                        valid = False
                        break
                if valid:
                    pos = 0
                    while pos < len(row) and row[pos] < x:
                        pos += 1
                    row.insert(pos, x)
                    for a in row:
                        if a != x:
                            used_set.add(abs(x - a))
                    found = True
                    improved = True
                    break
            
            if not found:
                # Put back at original position
                pos = 0
                while pos < len(row) and row[pos] < removed:
                    pos += 1
                row.insert(pos, removed)
                for a in row:
                    if a != removed:
                        used_set.add(abs(removed - a))
        
        if not improved:
            break
    
    final_scope = max(max(r) for r in final_rows)
    
    # Validate final result
    if is_valid(final_rows) and final_scope < best_scope_final:
        best_rows_final = final_rows
        best_scope_final = final_scope
>>>>>>> REPLACE

