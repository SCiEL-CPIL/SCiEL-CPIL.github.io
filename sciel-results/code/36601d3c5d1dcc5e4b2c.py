import random
import time
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    def get_max_entry(rows):
        return max(max(row) for row in rows) if rows else 0

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    def backtrack_row(used_diffs, max_val):
        """
        Build a row of k+1 increasing integers starting at 0, all differences
        distinct within the row and not in used_diffs, with max entry <= max_val.
        Uses MRV (minimum remaining values) heuristic.
        """
        row = [0]
        
        def _bt(idx, last_val):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            min_val = last_val + 1
            max_val_here = min(max_val, min_val + (k - idx - 1))
            
            # MRV: try values that have fewer conflicts first
            candidates = []
            for v in range(min_val, max_val_here + 1):
                new_diffs = {v - a for a in row}
                if new_diffs & used_diffs:
                    continue
                # Count how many future values this would block (simple heuristic)
                candidates.append(v)
            
            if not candidates:
                return False
            
            # Try in order of smallest new max difference (favors compactness)
            candidates.sort()
            
            for v in candidates:
                new_diffs = {v - a for a in row}
                row.append(v)
                used_diffs.update(new_diffs)
                if _bt(idx + 1, v):
                    return True
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        if _bt(1, 0):
            return row[:]
        return None

    def local_search_row_rebuild(rows):
        """
        Local search: repeatedly try to rebuild each row with a smaller max entry.
        """
        current_rows = [row[:] for row in rows]
        improved = True
        max_iter = 200
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Shuffle row order
            row_order = list(range(n_rows))
            random.shuffle(row_order)
            
            for r in row_order:
                if time.time() > deadline:
                    break
                
                old_row = current_rows[r]
                old_max = old_row[-1]
                
                # Collect used diffs from other rows
                used_diffs = set()
                for r2 in range(n_rows):
                    if r2 == r:
                        continue
                    used_diffs |= row_diffs(current_rows[r2])
                
                # Try to rebuild with smaller max
                # Start from just below current max and go down
                found = False
                for target_max in range(old_max - 1, max(lower_bound // n_rows, 1), -1):
                    if time.time() > deadline:
                        break
                    
                    new_row = backtrack_row(used_diffs.copy(), target_max)
                    if new_row is not None:
                        current_rows[r] = new_row
                        improved = True
                        found = True
                        break
                
                if not found:
                    # Try with same max but different structure
                    new_row = backtrack_row(used_diffs.copy(), old_max)
                    if new_row is not None and new_row != old_row:
                        # Check if it allows other rows to improve
                        current_rows[r] = new_row
                        improved = True
        
        return current_rows

    # Main algorithm
    initial_rows = pure_greedy()
    best_rows = [row[:] for row in initial_rows]
    best_max = get_max_entry(best_rows)
    
    current_rows = [row[:] for row in initial_rows]
    current_max = best_max
    
    # Phase 1: Initial polish
    polished = local_search_row_rebuild(current_rows)
    polished_max = get_max_entry(polished)
    if polished_max < best_max:
        best_rows = polished
        best_max = polished_max
        current_rows = polished
        current_max = polished_max
    
    # Phase 2: Iterated local search with random restarts
    ils_iter = 0
    max_ils_iters = 30
    
    while ils_iter < max_ils_iters and time.time() < deadline:
        ils_iter += 1
        
        # Perturb: rebuild a random subset of rows
        n_perturb = max(1, n_rows // 3)
        perturb_rows = random.sample(range(n_rows), min(n_perturb, n_rows))
        
        perturbed = [row[:] for row in current_rows]
        
        # Rebuild perturbed rows with randomization
        for r in perturb_rows:
            if time.time() > deadline:
                break
            
            old_row = perturbed[r]
            old_max = old_row[-1]
            
            used_diffs = set()
            for r2 in range(n_rows):
                if r2 == r:
                    continue
                used_diffs |= row_diffs(perturbed[r2])
            
            # Try several random target maxes
            found = False
            attempts = 0
            while attempts < 10 and time.time() < deadline:
                attempts += 1
                target_max = random.randint(max(lower_bound // n_rows, 1), old_max)
                new_row = backtrack_row(used_diffs.copy(), target_max)
                if new_row is not None:
                    perturbed[r] = new_row
                    found = True
                    break
            
            if not found:
                # Restore original if rebuild failed
                perturbed[r] = old_row
        
        perturbed_max = get_max_entry(perturbed)
        
        # Polish the perturbed state
        polished_perturbed = local_search_row_rebuild(perturbed)
        polished_p_max = get_max_entry(polished_perturbed)
        
        # Accept if better
        if polished_p_max < current_max:
            current_rows = polished_perturbed
            current_max = polished_p_max
            if current_max < best_max:
                best_rows = [row[:] for row in current_rows]
                best_max = current_max
        elif perturbed_max < current_max:
            current_rows = perturbed
            current_max = perturbed_max
            if current_max < best_max:
                best_rows = [row[:] for row in current_rows]
                best_max = current_max
    
    # Final validation
    if not is_valid(best_rows):
        best_rows = pure_greedy()

    return best_rows
# EVOLVE-BLOCK-END
