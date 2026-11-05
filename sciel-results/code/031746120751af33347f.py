import time
import random
import sys

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0, minimizing the largest entry,
    such that all positive differences within rows are distinct across all rows.
    
    Uses a local search / simulated annealing style approach with rebuilds.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe within 150s limit
    
    # Total number of differences per row: k*(k+1)/2
    # Total differences across all rows: n_rows * k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound on max value: we need at least total_diffs distinct positive integers
    # The smallest possible set of total_diffs distinct positive integers is 1..total_diffs
    # But differences from a row starting at 0 with max value M can only produce values up to M
    # So we need M >= total_diffs (roughly)
    # Actually, the counting bound: if max entry is M, then all differences are in [1, M]
    # We need total_diffs distinct differences, so M >= total_diffs
    # But also, each row has k+1 entries, so the max difference in a row is at most M
    # A tighter bound: for a single row with max entry M, the number of distinct differences
    # is at most k*(k+1)/2, and they must be in [1, M].
    
    def greedy_construct(n_rows, k):
        """Simple greedy: each row takes the smallest next mark whose differences are unused."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def get_score(rows):
        """Return the largest entry (the objective to minimize)."""
        return max(max(row) for row in rows)
    
    def is_valid(rows, n_rows, k):
        """Check if all positive differences across all rows are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def rebuild_row(rows, row_idx, used_diffs, k):
        """
        Rebuild a single row while keeping other rows fixed.
        used_diffs is a set of differences used by OTHER rows.
        Returns (new_row, success) where success is True if a valid row was found.
        """
        # We need to find a row of k+1 increasing integers starting at 0
        # such that all its internal differences are not in used_diffs
        # and all internal differences are distinct (within the row)
        
        # Greedy approach: try to build the row with smallest possible max
        # Start with a random offset or just greedy from 0
        row = [0]
        x = 0
        local_used = set()
        
        # Try multiple random starting strategies
        best_row = None
        best_max = float('inf')
        
        for attempt in range(50):
            row = [0]
            x = 0
            local_used = set()
            success = True
            while len(row) < k + 1:
                # Try next x, but sometimes jump ahead randomly
                found = False
                for _ in range(200):  # limit search per position
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs) and not (new_diffs & local_used):
                        row.append(x)
                        local_used |= new_diffs
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success and row[-1] < best_max:
                best_max = row[-1]
                best_row = list(row)
                if best_max <= k * (k + 1) // 2:  # can't do much better than counting bound per row
                    break
        
        if best_row is None:
            return None, False
        return best_row, True
    
    def compute_used_diffs_except(rows, exclude_idx):
        """Compute the set of differences used by all rows except the one at exclude_idx."""
        used = set()
        for i, row in enumerate(rows):
            if i == exclude_idx:
                continue
            for j in range(len(row)):
                for l in range(j + 1, len(row)):
                    used.add(row[l] - row[j])
        return used
    
    def local_search(rows, n_rows, k, time_limit):
        """
        Local search: repeatedly pick a row, rebuild it, keep if it reduces the global max.
        Also do random restarts of individual rows to escape local minima.
        """
        best_rows = [list(r) for r in rows]
        best_score = get_score(best_rows)
        
        while time.time() < time_limit:
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            
            # Compute used diffs from other rows
            used_diffs = compute_used_diffs_except(best_rows, row_idx)
            
            # Try to rebuild this row
            new_row, success = rebuild_row(best_rows, row_idx, used_diffs, k)
            
            if not success:
                continue
            
            # Check if the new row is valid with the other rows
            # (it should be by construction, but let's verify)
            temp_rows = [list(r) for r in best_rows]
            temp_rows[row_idx] = new_row
            if is_valid(temp_rows, n_rows, k):
                new_score = get_score(temp_rows)
                if new_score < best_score:
                    best_rows = temp_rows
                    best_score = new_score
                elif new_score == best_score and random.random() < 0.1:
                    # Accept equal-score moves with some probability to escape local minima
                    best_rows = temp_rows
            else:
                # Shouldn't happen, but skip
                continue
            
            # Occasionally do a more aggressive restart of a row
            if random.random() < 0.3:
                # Try to build the row with a larger initial gap to explore different structures
                # This is a heuristic to find different packing strategies
                pass
        
        return best_rows, best_score
    
    # Initial solution: greedy
    rows = greedy_construct(n_rows, k)
    
    # Verify validity
    if not is_valid(rows, n_rows, k):
        # Fallback: return the greedy result anyway (it should be valid)
        return rows
    
    # Run local search
    best_rows, best_score = local_search(rows, n_rows, k, deadline)
    
    # Final check
    if not is_valid(best_rows, n_rows, k):
        # Return the original greedy if something went wrong
        return rows
    
    return best_rows
