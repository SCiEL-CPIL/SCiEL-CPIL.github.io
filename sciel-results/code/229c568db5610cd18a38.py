import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0

    def greedy_init(n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_diffs = set()
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs) and not (new_diffs & local_diffs):
                    row.append(x)
                    local_diffs.update(new_diffs)
                x += 1
            rows.append(row)
            used_diffs.update(local_diffs)
        return rows, used_diffs

    def find_minimal_row(used_diffs, k, max_limit, time_limit=0.05):
        """
        Finds a row of length k+1 starting with 0, strictly increasing,
        with differences not in used_diffs and not repeating within the row.
        Minimizes the last element.
        Returns None if no such row exists with last element <= max_limit.
        """
        row = [0]
        local_diffs = set()
        start_t = time.time()
        
        def backtrack(current_row, current_local_diffs):
            if time.time() - start_t > time_limit:
                return None
            
            if len(current_row) == k + 1:
                return current_row[:]
            
            last_val = current_row[-1]
            
            # Heuristic: Try smaller x first to minimize max
            for x in range(last_val + 1, max_limit + 1):
                # Check conflicts with global used_diffs
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs:
                    continue
                # Check conflicts with local diffs
                if new_diffs & current_local_diffs:
                    continue
                
                # Tentatively add x
                current_row.append(x)
                current_local_diffs.update(new_diffs)
                
                # Recursive step
                result = backtrack(current_row, current_local_diffs)
                if result is not None:
                    return result
                
                # Backtrack
                current_row.pop()
                for d in new_diffs:
                    current_local_diffs.discard(d)
            
            return None

        return backtrack(row, local_diffs)

    # 1. Initialize
    rows, used_diffs = greedy_init(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    current_scope = best_scope

    # 2. Local Search with Backtracking Row Rebuild
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        indices = list(range(n_rows))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_row = rows[i]
            old_max = max(old_row)
            
            # Calculate diffs to remove
            old_diffs = set()
            for a in range(len(old_row)):
                for b in range(a + 1, len(old_row)):
                    old_diffs.add(old_row[b] - old_row[a])
            
            for d in old_diffs:
                used_diffs.discard(d)
                
            new_row = None
            
            # Try to find a row with max < old_max
            if old_max > k:
                # We search for a row with max <= old_max - 1
                new_row = find_minimal_row(used_diffs, k, old_max - 1, time_limit=0.05)
                
                if new_row is not None:
                    new_diffs = set()
                    for a in range(len(new_row)):
                        for b in range(a + 1, len(new_row)):
                            new_diffs.add(new_row[b] - new_row[a])
                            
                    if not (new_diffs & used_diffs):
                        rows[i] = new_row
                        used_diffs.update(new_diffs)
                        improved = True
                        current_scope = max(max(r) for r in rows)
                        
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [r[:] for r in rows]
                        no_improve_count = 0
                        continue # Next row
                
            # If no improvement, restore old row
            used_diffs.update(old_diffs)
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
        else:
            no_improve_count = 0

    return best_rows
