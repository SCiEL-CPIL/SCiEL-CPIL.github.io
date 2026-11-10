"""Construct difference triangle set: n_rows rows of k+1 increasing ints from 0."""
import time

def construct(n_rows, k):
    """Return n_rows rows, each a list of k+1 strictly increasing ints starting at 0,
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry."""
    deadline = time.time() + 120.0
    safety = 10.0  # safety margin in seconds

    # Greedy construction (baseline)
    def greedy():
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

    def max_entry(rows):
        return max(max(r) for r in rows)

    # Try greedy first
    best_rows = greedy()
    best_score = max_entry(best_rows)

    # Local improvement: try to reduce the last (largest) entry of each row
    # by backtracking within a row
    def try_improve_row(rows, row_idx, used, time_limit):
        """Try to find a shorter last entry for row row_idx by trying alternative
        sequences. Returns (new_row, new_used) or None."""
        row = rows[row_idx]
        if len(row) != k + 1:
            return None
        
        # Try to reduce the last element
        last = row[-1]
        # We need to find an alternative sequence ending at a smaller value
        # This is complex, so instead try swapping some elements
        
        # Alternative approach: try to shift the entire row down if possible
        # or find a different set of marks
        
        # Simple approach: try replacing the last mark with a smaller unused one
        # that doesn't conflict with existing diffs
        prefix = row[:-1]
        # Find all diffs already used in this row (excluding those involving last)
        row_diffs_prefix = set()
        for i in range(len(prefix)):
            for j in range(i + 1, len(prefix)):
                row_diffs_prefix.add(prefix[j] - prefix[i])
        
        # Diffs involving last
        for a in prefix:
            row_diffs_prefix.add(last - a)
        
        # We want to find x < last such that all (x - a) for a in prefix
        # are not in used (globally) and not in row_diffs_prefix (except we remove old ones)
        
        # Actually, let's try a different strategy: local search on the whole set
        
    # Backtracking search for a single row given used differences
    def search_row(used, k, max_val, time_start, time_limit):
        """Find a row of k+1 increasing ints starting at 0, max <= max_val,
        with no diff conflicts with used. Returns row or None."""
        row = [0]
        
        def backtrack(pos, current_max):
            if time.time() > time_limit:
                return None
            if pos == k + 1:
                return list(row)
            
            # Try candidates from current_max+1 upward
            for x in range(current_max + 1, max_val + 1):
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    result = backtrack(pos + 1, x)
                    if result is not None:
                        return result
                    used -= new_diffs
                    row.pop()
                if time.time() > time_limit:
                    return None
            return None
        
        return backtrack(1, 0)

    # Iterative improvement: try to reduce the max entry
    # Strategy: rebuild rows one at a time with a tighter bound
    for iteration in range(50):
        if time.time() > deadline - safety:
            break
        
        current_max = best_score
        # Try to find a configuration with max < current_max
        target_max = current_max - 1
        
        # Rebuild with constraint
        used = set()
        new_rows = []
        success = True
        time_start = time.time()
        
        for r in range(n_rows):
            if time.time() > time_start + 15:
                success = False
                break
            row = search_row(used, k, target_max, time_start, time_start + 15)
            if row is None:
                success = False
                break
            new_rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        if success and new_rows:
            new_score = max_entry(new_rows)
            if new_score < best_score:
                best_rows = new_rows
                best_score = new_score
                print(f"Iteration {iteration}: improved to {best_score}")
        else:
            # Can't improve with target_max, stop
            break

    return best_rows
