"""Difference triangle set (7,5) - randomized search with local improvement."""
import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s kill
    total_diffs = k * (k + 1) // 2  # number of positive diffs per row (k=5 -> 15)
    
    def row_diffs(row):
        """Return set of all positive differences in a row."""
        return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
    
    def is_valid(rows):
        """Check all positive differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for d in row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    def greedy_row(used, k, randomize=False, max_val=None):
        """Build one row greedily, optionally with randomization."""
        row = [0]
        if max_val is None:
            max_val = 200
        while len(row) < k + 1:
            candidates = []
            for x in range(row[-1] + 1, max_val + 1):
                new = {x - a for a in row}
                if not (new & used):
                    candidates.append(x)
            if not candidates:
                return None
            if randomize and candidates:
                # Pick from a random subset of the best candidates
                pick_from = candidates[:max(1, len(candidates) // 3)]
                x = random.choice(pick_from)
            else:
                x = candidates[0]
            row.append(x)
            used |= {x - a for a in row[:-1]}
        return row
    
    def construct_randomized(k, used, randomize=True, max_val=150):
        """Construct all rows with randomization."""
        rows = []
        local_used = set(used)
        for _ in range(n_rows):
            row = greedy_row(local_used, k, randomize=randomize, max_val=max_val)
            if row is None:
                return None
            rows.append(row)
            local_used |= row_diffs(row)
        return rows
    
    def local_improve(rows, used, deadline):
        """Try to improve by removing one entry and re-optimizing."""
        best_rows = rows
        best_max = max_entry(rows)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            # Pick a random row and a random non-zero entry to remove
            ri = random.randrange(n_rows)
            # Can't remove index 0 (must start at 0)
            ei = random.randrange(1, len(best_rows[ri]))
            old_val = best_rows[ri][ei]
            
            # Remove the entry and its differences
            new_row = best_rows[ri][:ei] + best_rows[ri][ei+1:]
            old_diffs = {new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row))}
            
            # Recompute used without this row's diffs
            new_used = set()
            for r in best_rows:
                if r is not best_rows[ri]:
                    new_used |= row_diffs(r)
            
            # Try to rebuild this row
            rebuilt = greedy_row(new_used, k, randomize=True, max_val=best_max + 20)
            if rebuilt is not None:
                candidate_rows = list(best_rows)
                candidate_rows[ri] = rebuilt
                if is_valid(candidate_rows):
                    cand_max = max_entry(candidate_rows)
                    if cand_max < best_max:
                        best_rows = candidate_rows
                        best_max = cand_max
        
        return best_rows, best_max
    
    def construct_with_backtracking(k, deadline, max_val=130):
        """Construct rows with limited backtracking."""
        rows = []
        used = set()
        
        def backtrack(row_idx):
            if time.time() > deadline:
                return False
            if row_idx == n_rows:
                return True
            
            row = [0]
            local_used = set(used)
            
            def build_row(pos):
                if time.time() > deadline:
                    return None
                if pos == k:
                    return row[:]
                for x in range(row[-1] + 1, max_val + 1):
                    new = {x - a for a in row}
                    if not (new & local_used):
                        row.append(x)
                        local_used |= new
                        result = build_row(pos + 1)
                        if result is not None:
                            return result
                        row.pop()
                        local_used -= new
                    if len(row) + (k - pos) <= k and x - row[-1] > 30:
                        break  # pruning
                return None
            
            result = build_row(0)
            if result is not None:
                rows.append(result)
                used |= row_diffs(result)
                if backtrack(row_idx + 1):
                    return True
                # Backtrack
                used -= row_diffs(result)
                rows.pop()
            return False
        
        if backtrack(0):
            return rows
        return None
    
    # Strategy 1: Try pure greedy first (fast baseline)
    best_rows = None
    best_max = float('inf')
    
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used, k, randomize=False, max_val=200)
        if row is None:
            break
        rows.append(row)
        used |= row_diffs(row)
    if len(rows) == n_rows and is_valid(rows):
        best_rows = rows
        best_max = max_entry(rows)
    
    # Strategy 2: Randomized greedy with multiple restarts
    for restart in range(2000):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, randomize=True, max_val=best_max + 30 if best_rows else 150)
            if row is None:
                break
            rows.append(row)
            used |= row_diffs(row)
        if len(rows) == n_rows and is_valid(rows):
            m = max_entry(rows)
            if m < best_max:
                best_max = m
                best_rows = rows
                # Try local improvement
                improved, imax = local_improve(best_rows, set(), deadline)
                if imax < best_max:
                    best_max = imax
                    best_rows = improved
    
    # Strategy 3: Backtracking for tighter solutions
    if time.time() < deadline - 30:
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 7919)
            bt_rows = construct_with_backtracking(k, deadline - 10, max_val=best_max + 10 if best_rows else 120)
            if bt_rows is not None:
                m = max_entry(bt_rows)
                if m < best_max:
                    best_max = m
                    best_rows = bt_rows
    
    if best_rows is None:
        # Fallback: pure greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, randomize=False, max_val=200)
            if row is not None:
                best_rows.append(row)
                used |= row_diffs(row)
    
    return best_rows
