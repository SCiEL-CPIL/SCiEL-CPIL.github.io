def construct(n_rows, k):
    """
    Build n_rows rows of k+1 increasing integers starting at 0,
    such that all positive within-row differences are globally distinct.
    Minimize the largest entry.
    """
    used_diffs = set()
    rows = []

    for r in range(n_rows):
        row = [0]
        for _ in range(k):
            # Try candidate x starting from current max + 1
            # Look at the smallest few feasible candidates and pick the best
            best_x = None
            best_max_diff = float('inf')
            candidates_checked = 0
            x = row[-1] + 1
            max_candidates = 50  # check up to 50 feasible candidates
            
            while candidates_checked < max_candidates:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs:
                    # Feasible: evaluate how much this extends the max used diff
                    new_max = max(max(used_diffs) if used_diffs else 0, max(new_diffs))
                    if new_max < best_max_diff:
                        best_max_diff = new_max
                        best_x = x
                    # Also prefer smaller x if same new_max
                    candidates_checked += 1
                    if candidates_checked >= max_candidates:
                        break
                x += 1
            
            if best_x is None:
                # Fallback: just take the first feasible
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        best_x = x
                        break
                    x += 1
            
            row.append(best_x)
            used_diffs |= {best_x - a for a in row[:-1]}
        
        rows.append(row)
    
    # Local improvement: try to reduce the global max by shifting individual marks
    # For each row, for each position (except 0 and last), try small shifts
    for _ in range(50):  # limited iterations
        improved = False
        for r in range(n_rows):
            for pos in range(1, len(rows[r]) - 1):
                original = rows[r][pos]
                for delta in range(-3, 4):
                    if delta == 0:
                        continue
                    candidate = original + delta
                    # Check increasing: rows[r][pos-1] < candidate < rows[r][pos+1]
                    if not (rows[r][pos-1] < candidate < rows[r][pos+1]):
                        continue
                    # Compute new diffs for this row
                    new_row = rows[r][:]
                    new_row[pos] = candidate
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                    # Check against all other rows' diffs
                    other_diffs = set()
                    for rr in range(n_rows):
                        if rr == r:
                            continue
                        for i in range(len(rows[rr])):
                            for j in range(i+1, len(rows[rr])):
                                other_diffs.add(rows[rr][j] - rows[rr][i])
                    if not (new_diffs & other_diffs):
                        # Valid move
                        old_max = max(max(row) for row in rows)
                        new_max = max(max(row) for row in rows if row != new_row)
                        new_max = max(new_max, max(new_row))
                        if new_max < old_max:
                            rows[r] = new_row
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return rows
