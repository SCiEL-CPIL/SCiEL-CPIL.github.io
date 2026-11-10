import time
import random

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135  # Safety margin

    # Max value estimate: Best known is 112, counting bound 105.
    # We need a buffer for greedy overshoots. 200 is safe.
    MAX_VAL = 200
    
    def get_diffs_mask(row):
        """Return a bytearray mask of differences for the row."""
        mask = bytearray(MAX_VAL + 1)
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - a
                if d <= MAX_VAL:
                    mask[d] = 1
        return mask

    def is_valid_mask(row, used_mask):
        """Check if row's diffs are disjoint from used_mask."""
        for i in range(len(row)):
            a = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - a
                if d > MAX_VAL:
                    return False
                if used_mask[d]:
                    return False
        return True

    def greedy_row(used_mask, k, cap=MAX_VAL, randomize=False):
        """
        Construct a row using greedy strategy.
        used_mask: bytearray of currently used diffs.
        Returns (row, diffs_mask) or (None, None)
        """
        row = [0]
        # local_mask tracks diffs within this new row to ensure internal uniqueness
        local_mask = bytearray(MAX_VAL + 1)
        
        # Pre-compute potential candidates? No, just iterate.
        # Optimization: If randomize, we might pick from a small window of valid next steps
        while len(row) < k + 1:
            found = False
            # Determine range to search
            # In greedy, we usually look at the smallest available.
            # In randomized, we might look at a window.
            last_val = row[-1]
            
            # Standard greedy: smallest x > last_val
            # Randomized: pick x from [last_val+1, last_val+window]
            window = 1 if not randomize else random.randint(1, 5)
            
            for x in range(last_val + 1, min(last_val + window + 1, cap + 1)):
                if x > cap:
                    break
                
                # Check conflicts with global used_mask and local_mask
                conflict = False
                for a in row:
                    d = x - a
                    if d > MAX_VAL:
                        conflict = True
                        break
                    if used_mask[d] or local_mask[d]:
                        conflict = True
                        break
                
                if not conflict:
                    # Accept x
                    row.append(x)
                    for a in row[:-1]: # Update local mask with new diffs
                        d = x - a
                        local_mask[d] = 1
                    found = True
                    break
            
            if not found:
                return None, None
        
        return row, local_mask

    def dfs_row(row, used_mask, local_mask, cap, row_deadline, depth=0):
        """
        DFS to find a row with last element <= cap.
        Returns (row, local_mask) or (None, None)
        """
        if time.time() > row_deadline:
            return None, None
        
        if len(row) == k + 1:
            return row[:], local_mask[:]
        
        last = row[-1]
        needed = (k + 1) - len(row)
        upper = cap - (needed - 1)
        
        if upper < last + 1:
            return None, None
        
        # Iterate candidates
        # To improve diversity, we can shuffle the candidate order occasionally
        # But standard ascending is good for finding *small* rows quickly
        for x in range(last + 1, upper + 1):
            if time.time() > row_deadline:
                return None, None
            
            conflict = False
            for a in row:
                d = x - a
                if d > MAX_VAL:
                    conflict = True
                    break
                if used_mask[d] or local_mask[d]:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # Try this x
            new_row = row + [x]
            new_local_mask = local_mask[:]
            for a in row:
                new_local_mask[x - a] = 1
            
            res_row, res_mask = dfs_row(new_row, used_mask, new_local_mask, cap, row_deadline, depth + 1)
            if res_row is not None:
                return res_row, res_mask
        
        return None, None

    # ---- Phase 1: Initial Construction ----
    def construct_solution(randomize=False):
        used_global = bytearray(MAX_VAL + 1)
        rows = []
        for i in range(n_rows):
            row, mask = greedy_row(used_global, k, randomize=randomize)
            if row is None:
                return None
            rows.append(row)
            # Update global mask
            for d in range(MAX_VAL + 1):
                if mask[d]:
                    used_global[d] = 1
        return rows

    # Get a valid starting solution
    best_rows = construct_solution(randomize=False)
    if best_rows is None:
        # Fallback: try randomized
        for _ in range(10):
            best_rows = construct_solution(randomize=True)
            if best_rows is not None:
                break
    if best_rows is None:
        # Should not happen, but return empty or error
        return [[] for _ in range(n_rows)]

    best_max = max(r[-1] for r in best_rows)
    
    # ---- Phase 2: Local Search Improvement ----
    # We iterate, trying to replace the "worst" row (largest last element) with a smaller one.
    # We use DFS to find a smaller row that fits in the gap left by the current row.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Sort indices by last element descending, so we attack the largest first
        indices = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        
        for idx in indices:
            if time.time() > deadline - 10:
                break
            
            current_row = best_rows[idx]
            current_max = current_row[-1]
            
            # Calculate the mask of all OTHER rows
            used_others = bytearray(MAX_VAL + 1)
            for j in range(n_rows):
                if j == idx:
                    continue
                # Efficiently update mask
                r = best_rows[j]
                for i in range(len(r)):
                    a = r[i]
                    for m in range(i + 1, len(r)):
                        d = r[m] - a
                        if d <= MAX_VAL:
                            used_others[d] = 1
            
            # Try to find a row with max < current_max
            # We try caps from current_max - 1 downwards
            # To save time, we don't try every single cap, but a few steps down
            # Or just one DFS with cap = current_max - 1. If it fails, try lower.
            
            found = False
            # Limit search depth/time for this specific row replacement
            row_deadline = min(deadline - 10, time.time() + 2.0)
            
            # Try a few caps
            for cap in range(current_max - 1, max(current_max - 20, 0), -1):
                if time.time() > row_deadline:
                    break
                
                # Reset local mask for DFS start
                start_local = bytearray(MAX_VAL + 1)
                new_row, new_mask = dfs_row([0], used_others, start_local, cap, row_deadline)
                
                if new_row is not None:
                    # Check if it's actually better
                    if new_row[-1] < current_max:
                        best_rows[idx] = new_row
                        best_max = max(r[-1] for r in best_rows)
                        improved = True
                        found = True
                        break
            
            if found:
                # Restart the loop to re-evaluate other rows as the "others" mask changed
                break

    # ---- Phase 3: Randomized Restarts with Local Refinement ----
    # If we are stuck, try a completely new random construction and refine it.
    if time.time() < deadline - 20:
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            candidate_rows = construct_solution(randomize=True)
            if candidate_rows is None:
                continue
            
            cand_max = max(r[-1] for r in candidate_rows)
            
            # Quick local refinement on the candidate
            # (Simplified: just one pass of the local search above)
            improved_cand = True
            while improved_cand and time.time() < deadline - 15:
                improved_cand = False
                indices = sorted(range(n_rows), key=lambda i: candidate_rows[i][-1], reverse=True)
                for idx in indices:
                    if time.time() > deadline - 15:
                        break
                    current_row = candidate_rows[idx]
                    current_max = current_row[-1]
                    
                    used_others = bytearray(MAX_VAL + 1)
                    for j in range(n_rows):
                        if j == idx: continue
                        r = candidate_rows[j]
                        for i in range(len(r)):
                            a = r[i]
                            for m in range(i + 1, len(r)):
                                d = r[m] - a
                                if d <= MAX_VAL:
                                    used_others[d] = 1
                    
                    row_deadline = min(deadline - 15, time.time() + 1.0)
                    for cap in range(current_max - 1, max(current_max - 10, 0), -1):
                        if time.time() > row_deadline:
                            break
                        start_local = bytearray(MAX_VAL + 1)
                        new_row, _ = dfs_row([0], used_others, start_local, cap, row_deadline)
                        if new_row is not None and new_row[-1] < current_max:
                            candidate_rows[idx] = new_row
                            cand_max = max(r[-1] for r in candidate_rows)
                            improved_cand = True
                            break
                    if improved_cand:
                        break
            
            if cand_max < best_max:
                best_rows = candidate_rows
                best_max = cand_max

    return best_rows
