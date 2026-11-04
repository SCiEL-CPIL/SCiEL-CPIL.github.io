import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for 120s limit
    
    best_rows = None
    best_score = float('inf')
    
    def row_diffs(row):
        return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
    
    def count_future_options(used, row, pos, cap):
        """Count how many valid next values exist for the current row position."""
        if pos >= k:
            return 1  # Row is complete
        count = 0
        last = row[-1]
        for x in range(last + 1, min(cap, 200) + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used) and len(new_diffs) == len(row):
                count += 1
                if count > 50:  # Cap to save time
                    return count
        return count
    
    def backtracking_with_mrv(used, rows, row_idx, current_scope_cap):
        """Backtracking search with minimum remaining values heuristic."""
        if time.time() - start_time > time_limit:
            return None
        
        if row_idx == n_rows:
            return rows
        
        row = [0]
        # Build this row element by element
        def build_row(row, pos, used_copy):
            if time.time() - start_time > time_limit:
                return None
            
            if pos == k:
                # Row complete, check if we can continue
                return (row, used_copy)
            
            last = row[-1]
            candidates = []
            for x in range(last + 1, current_scope_cap + 1):
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_copy) and len(new_diffs) == len(row):
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # MRV heuristic: try candidates that leave the most options for future
            # Sort candidates by how many options they leave for the rest of this row
            scored_candidates = []
            for x in candidates:
                new_row = row + [x]
                new_used = used_copy | {x - a for a in row}
                # Estimate remaining options
                opts = count_future_options(new_used, new_row, pos + 1, current_scope_cap)
                scored_candidates.append((opts, x))
            
            # Sort by options (descending - most constrained first)
            scored_candidates.sort(reverse=True)
            
            for opts, x in scored_candidates:
                new_row = row + [x]
                new_used = used_copy | {x - a for a in row}
                result = build_row(new_row, pos + 1, new_used)
                if result is not None:
                    return result
            
            return None
        
        result = build_row(row, 0, used)
        if result is None:
            return None
        
        new_row, new_used = result
        rows.append(new_row)
        return backtracking_with_mrv(new_used, rows, row_idx + 1, current_scope_cap)
    
    def local_search(rows):
        current_max = max(max(r) for r in rows)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            # Try to improve each row
            for i in range(n_rows):
                row = rows[i]
                # Iterate from last element to first (excluding 0)
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    # Try smaller values
                    min_val = row[pos-1] + 1
                    max_val = current_val - 1  # We want to reduce
                    
                    if max_val < min_val:
                        continue
                        
                    # Precompute diffs of other rows for faster checking
                    other_diffs = set()
                    for r_idx, r in enumerate(rows):
                        if r_idx != i:
                            other_diffs.update(row_diffs(r))
                    
                    # Try values from smallest to largest
                    for v in range(min_val, max_val + 1):
                        # Create a copy of the row with the change
                        candidate_row = list(row)
                        candidate_row[pos] = v
                        
                        # Check if the new row is internally valid and doesn't conflict
                        new_diffs = row_diffs(candidate_row)
                        
                        # Check internal validity (unique diffs within the row)
                        expected_diffs = len(candidate_row) * (len(candidate_row) - 1) // 2
                        if len(new_diffs) != expected_diffs:
                            continue
                        
                        # Check no conflicts with other rows
                        if new_diffs & other_diffs:
                            continue
                        
                        # Check strictly increasing
                        valid = True
                        for j in range(1, len(candidate_row)):
                            if candidate_row[j] <= candidate_row[j-1]:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Accept the change
                        rows[i] = candidate_row
                        new_max = max(max(r) for r in rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                        break # Break inner for v loop
                    if improved:
                        break # Break inner for pos loop
                if improved:
                    break # Break outer for i loop
        return rows, current_max

    # Main loop
    attempts = 0
    while time.time() - start_time < time_limit:
        # Strategy 1: Backtracking with MRV
        if attempts % 3 == 0:
            # Try backtracking with a scope cap
            scope_cap = min(150, best_score if best_score < float('inf') else 150)
            rows = backtracking_with_mrv(set(), [], 0, scope_cap)
        else:
            # Strategy 2: Randomized greedy with MRV-inspired weighting
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    valids = []
                    last = row[-1]
                    cap = 150
                    if best_score < float('inf'):
                        cap = min(cap, best_score)
                    
                    for x in range(last + 1, cap + 1):
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used) and len(new_diffs) == len(row):
                            valids.append(x)
                    
                    if not valids:
                        success = False
                        break
                    
                    # MRV-inspired: weight by how many options each choice leaves
                    scored = []
                    for x in valids[:30]:  # Limit to top 30 for speed
                        new_row = row + [x]
                        new_used = used | {x - a for a in row}
                        # Quick estimate of remaining options
                        opts = 0
                        for y in range(new_row[-1] + 1, min(new_row[-1] + 30, cap) + 1):
                            nd = {y - a for a in new_row}
                            if not (nd & new_used) and len(nd) == len(new_row):
                                opts += 1
                                if opts > 20:
                                    break
                        scored.append((opts, x))
                    
                    scored.sort(reverse=True)
                    # Pick from top candidates with some randomness
                    top_candidates = [x for _, x in scored[:10]]
                    if len(top_candidates) > 1:
                        chosen = random.choice(top_candidates)
                    else:
                        chosen = top_candidates[0] if top_candidates else valids[0]
                    
                    row.append(chosen)
                    used.update({chosen - a for a in row[:-1]})
                
                if not success:
                    break
                rows.append(row)
            
            if success:
                rows = rows
            else:
                rows = None
        
        if rows is not None:
            score = max(max(r) for r in rows)
            if score < best_score:
                best_score = score
                best_rows = [list(r) for r in rows]
                # Try local search
                if time.time() - start_time < time_limit - 5:
                    improved_rows, improved_score = local_search(best_rows)
                    # Validate improved_rows before accepting
                    if improved_score < best_score:
                        # Quick validation
                        all_diffs = set()
                        valid = True
                        for r in improved_rows:
                            d_set = row_diffs(r)
                            if all_diffs & d_set:
                                valid = False
                                break
                            all_diffs.update(d_set)
                            # Check internal validity (strictly increasing)
                            for j in range(1, len(r)):
                                if r[j] <= r[j-1]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            best_score = improved_score
                            best_rows = improved_rows
                if best_score <= 105:
                    break
        attempts += 1

    if best_rows is None:
        # Fallback to deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and len(new_diffs) == len(row):
                        break
                    x += 1
                row.append(x)
                used.update({x - a for a in row[:-1]})
            rows.append(row)
        best_rows = rows

    return best_rows
