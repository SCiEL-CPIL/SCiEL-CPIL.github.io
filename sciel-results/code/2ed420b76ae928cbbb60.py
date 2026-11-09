import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    total_diffs = n_rows * (k + 1) * k // 2
    
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def backtrack_search(M, seed_val, max_nodes=500000):
        """
        Randomized backtracking with forward checking.
        Build rows one element at a time.
        State: list of partial rows, used_diffs set.
        """
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = [[] for _ in range(n_rows)]
        # Each row starts with [0]
        for i in range(n_rows):
            rows[i].append(0)
        
        # Track which position we're filling in each row
        # We'll fill rows in order, and within a row, fill left to right.
        # Position (r, p) means row r, position p (0-indexed, so p=0 is already 0)
        # We need to fill positions 1..k for each row.
        
        # To add randomness, we can randomize the order in which we try candidate values.
        # But we must maintain increasing order within a row.
        
        # Let's use a stack-based backtracking.
        # State: (row_idx, pos_idx, rows_copy, used_diffs_copy)
        # But copying is expensive. Let's use undo/redo.
        
        # Alternative: iterative deepening or simple recursive backtracking with pruning.
        
        # For efficiency, let's try a different approach:
        # Build one complete row at a time, using randomized greedy within the row.
        # If stuck, backtrack to the last choice point in the row.
        
        # Actually, let's do full backtracking but with a node limit.
        
        def _backtrack(row_idx, pos_idx):
            """
            Try to fill row[row_idx][pos_idx] and subsequent positions.
            row_idx: which row we're currently building (0..n_rows-1)
            pos_idx: which position in that row (1..k, since pos 0 is always 0)
            Returns True if a complete valid solution is found.
            """
            if row_idx >= n_rows:
                return True
            
            if pos_idx > k:
                # Move to next row
                return _backtrack(row_idx + 1, 1)
            
            # Get the last value in the current row
            current_row = rows[row_idx]
            last_val = current_row[-1]
            
            # Candidate values: from last_val + 1 to M
            # But we need to ensure we can still fill remaining positions.
            # Remaining positions in this row: k - pos_idx + 1 (positions pos_idx..k)
            # Actually, pos_idx is 1-indexed for the positions to fill.
            # If pos_idx = 1, we need to fill positions 1..k (k positions).
            # If pos_idx = k, we need to fill position k (1 position).
            remaining_in_row = k - pos_idx + 1
            
            # The minimum value we can pick: last_val + 1
            # The maximum value we can pick: M - (remaining_in_row - 1)
            # Because we need room for the remaining positions.
            min_val = last_val + 1
            max_val = M - (remaining_in_row - 1)
            
            if min_val > max_val:
                return False
            
            # Generate candidate values. To add randomness, shuffle a subset?
            # Or try a random sample first, then fall back to systematic?
            # For now, let's try values in a randomized order.
            # But we must be careful: we want to find a solution, not just any.
            # Let's try a mix: some random candidates, then systematic.
            
            # For efficiency, let's just try all candidates from min_val to max_val,
            # but in a randomized order to break symmetry.
            candidates = list(range(min_val, max_val + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                # Check if val is valid:
                # 1. All diffs val - current_row[i] must be in [1, M] (they are, since val > current_row[i] and val <= M)
                # 2. These diffs must not be in used_diffs
                # 3. These diffs must be distinct from each other (they are, since current_row is strictly increasing)
                # 4. These diffs must not create internal conflicts with existing diffs in current_row
                
                new_diffs = set()
                valid = True
                for prev in current_row:
                    d = val - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check internal conflicts: new_diffs must not intersect with existing diffs in current_row
                # Existing diffs in current_row:
                existing_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        existing_diffs.add(current_row[j] - current_row[i])
                
                if new_diffs & existing_diffs:
                    continue
                
                # Also check: new_diffs must be distinct from each other (already guaranteed)
                # And must not be in used_diffs (already checked)
                
                # Tentatively add val
                current_row.append(val)
                used_diffs.update(new_diffs)
                
                # Recurse
                if _backtrack(row_idx, pos_idx + 1):
                    return True
                
                # Undo
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # We need to limit the number of nodes explored.
        # Let's add a counter.
        node_count = [0]
        max_nodes_local = max_nodes
        
        def _backtrack_limited(row_idx, pos_idx):
            if node_count[0] >= max_nodes_local:
                return False
            node_count[0] += 1
            
            if row_idx >= n_rows:
                return True
            
            if pos_idx > k:
                return _backtrack_limited(row_idx + 1, 1)
            
            current_row = rows[row_idx]
            last_val = current_row[-1]
            remaining_in_row = k - pos_idx + 1
            min_val = last_val + 1
            max_val = M - (remaining_in_row - 1)
            
            if min_val > max_val:
                return False
            
            # Try candidates in randomized order
            candidates = list(range(min_val, max_val + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                new_diffs = set()
                valid = True
                for prev in current_row:
                    d = val - prev
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                existing_diffs = set()
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        existing_diffs.add(current_row[j] - current_row[i])
                
                if new_diffs & existing_diffs:
                    continue
                
                current_row.append(val)
                used_diffs.update(new_diffs)
                
                if _backtrack_limited(row_idx, pos_idx + 1):
                    return True
                
                current_row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Try the search
        if _backtrack_limited(0, 1):
            return [r[:] for r in rows]
        return None
    
    best_rows = None
    best_M = float('inf')
    
    # Try M from lower bound upward
    # For (7,5): LB=105, known best=112
    # We'll try M from 105 to 130
    
    # First, try a quick greedy to get a baseline
    def greedy_construct(M_limit, seed_val=0):
        rng = random.Random(seed_val)
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                next_val = row[-1] + 1
                valid_found = False
                while next_val <= M_limit:
                    new_diffs = set()
                    valid = True
                    for prev in row:
                        d = next_val - prev
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        curr_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                curr_diffs.add(row[j] - row[i])
                        
                        if new_diffs & curr_diffs:
                            valid = False
                        if new_diffs & used_diffs:
                            valid = False
                        
                        if valid:
                            row.append(next_val)
                            used_diffs.update(new_diffs)
                            valid_found = True
                            break
                    next_val += 1
                
                if not valid_found:
                    return None
            
            rows.append(row)
        
        return rows
    
    # Get a baseline solution with greedy
    for M_base in range(total_diffs + 10, total_diffs + 40):
        if time.time() > deadline - 20:
            break
        rows = greedy_construct(M_base)
        if rows is not None and is_valid(rows):
            best_rows = rows
            best_M = max(max(r) for r in rows)
            break
    
    # Now try to improve with backtracking for smaller M values
    # Try M from total_diffs to best_M - 1
    upper_search = best_M - 1 if best_rows is not None else total_diffs + 20
    lower_search = total_diffs
    
    for M in range(lower_search, upper_search + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            rows = backtrack_search(M, seed, max_nodes=200000)
            if rows is not None and is_valid(rows):
                current_max = max(max(r) for r in rows)
                if current_max < best_M:
                    best_M = current_max
                    best_rows = [r[:] for r in rows]
                break  # Found a solution for this M, try smaller M
    
    # If no solution found, use the greedy baseline or a fallback
    if best_rows is None:
        M_fallback = total_diffs + 50
        rows = greedy_construct(M_fallback)
        if rows is not None and is_valid(rows):
            best_rows = rows
        else:
            # Emergency fallback
            M_huge = 2000
            rows = greedy_construct(M_huge)
            if rows is not None:
                best_rows = rows
            else:
                # Construct a valid solution with large gaps
                best_rows = []
                base = 0
                for i in range(n_rows):
                    row = [0]
                    for j in range(1, k + 1):
                        base += 100
                        row.append(base)
                    best_rows.append(row)
                # Verify
                if not is_valid(best_rows):
                    # Should not happen with large gaps
                    best_rows = [[0] + list(range(1, k + 1)) * (i + 1) for i in range(n_rows)]
    
    return best_rows
