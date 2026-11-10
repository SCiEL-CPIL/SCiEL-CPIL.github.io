"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_rows = None
    best_max = float('inf')
    
    def validate(rows):
        """Check that all positive differences within rows are distinct across all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True
    
    def max_entry(rows):
        return max(row[-1] for row in rows)
    
    def greedy_row(used, k, rng, backtracks=0):
        """Build one row greedily with optional randomization and backtracking."""
        row = [0]
        
        def dfs(depth, current_row, used_local, bt_remaining):
            if depth == k + 1:
                return True, current_row
            
            # Try candidates in randomized order, biased toward smaller
            candidates = list(range(current_row[-1] + 1, current_row[-1] + 50))
            rng.shuffle(candidates)
            # Sort by a score: prefer smaller, but with some randomness
            candidates.sort(key=lambda x: x + rng.random() * 5)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_local:
                    continue
                new_row = current_row + [x]
                new_used = used_local | new_diffs
                success, result = dfs(depth + 1, new_row, new_used, bt_remaining - 1)
                if success:
                    return True, result
                if bt_remaining <= 0:
                    break
            return False, None
        
        success, result = dfs(1, row, used, backtracks)
        if success:
            return result
        return None
    
    def randomized_greedy(n_rows, k, rng, max_backtracks=2):
        """Build all rows using randomized greedy."""
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            bt = max_backtracks if r < n_rows // 2 else 0
            
            def dfs_row(depth, current_row, used_local, bt_rem):
                if depth == k + 1:
                    return True, current_row
                
                # Generate candidates: start from last+1, go up
                last = current_row[-1]
                # Try a window of candidates
                max_candidate = last + 80
                candidates = list(range(last + 1, max_candidate + 1))
                
                # Randomize with bias toward smaller
                rng.shuffle(candidates)
                candidates.sort(key=lambda x: x * 0.7 + rng.random() * 20)
                
                for x in candidates:
                    new_diffs = {x - a for a in current_row}
                    if new_diffs & used_local:
                        continue
                    new_row = current_row + [x]
                    new_used = used_local | new_diffs
                    
                    if depth == k:  # Last position, no need to backtrack further
                        return True, new_row
                    
                    success, result = dfs_row(depth + 1, new_row, new_used, bt_rem - 1)
                    if success:
                        return True, result
                    if bt_rem <= 0:
                        break
                
                return False, None
            
            success, result = dfs_row(1, row, used, bt)
            if not success:
                # Fallback to pure greedy
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
                        x += 1
                result = row
            
            rows.append(result)
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    used.add(result[j] - result[i])
        
        return rows
    
    def try_compression(rows, used_diffs, k, rng, max_iters=100):
        """Try to reduce the maximum entry by locally modifying rows."""
        rows = [list(r) for r in rows]
        current_max = max_entry(rows)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Find the row with the largest entry
            max_row_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            row = rows[max_row_idx]
            
            if row[-1] <= 0:
                break
            
            # Try to reduce the last element
            old_last = row[-1]
            # Compute used diffs excluding this row's diffs
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    row_diffs.add(row[j] - row[i])
            
            other_diffs = used_diffs - row_diffs
            
            # Try smaller values for the last element
            found = False
            for new_last in range(row[-2] + 1, old_last):
                new_row = row[:-1] + [new_last]
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                
                if not (new_diffs & other_diffs):
                    # Valid replacement
                    rows[max_row_idx] = new_row
                    # Update used_diffs
                    used_diffs = other_diffs | new_diffs
                    current_max = max_entry(rows)
                    found = True
                    break
            
            if not found:
                # Try modifying other elements too
                # Try to reduce second-to-last
                if len(row) >= 3:
                    old_second_last = row[-2]
                    for new_sl in range(row[-3] + 1, old_second_last):
                        # Check if this creates valid diffs
                        # This is complex; skip for now
                        pass
        
        return rows, used_diffs
    
    # Try multiple strategies
    strategies = [
        (0, 0),    # pure greedy
        (1, 1),    # light randomization
        (2, 3),    # more backtracking
        (3, 5),    # even more
    ]
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Pure greedy baseline
            used = set()
            rows = []
            for r in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    x = row[-1] + 1
                    while True:
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            break
                        x += 1
                rows.append(row)
            
            if validate(rows):
                m = max_entry(rows)
                if m < best_max:
                    best_max = m
                    best_rows = [list(r) for r in rows]
        else:
            # Randomized approach
            rng = random.Random(attempt * 12345 + 7)
            
            # Choose strategy based on attempt
            if attempt < 10:
                max_bt = 1
            elif attempt < 25:
                max_bt = 3
            else:
                max_bt = 5
            
            rows = randomized_greedy(n_rows, k, rng, max_backtracks=max_bt)
            
            if rows and validate(rows):
                m = max_entry(rows)
                if m < best_max:
                    best_max = m
                    best_rows = [list(r) for r in rows]
                    
                    # Try compression
                    used_diffs = set()
                    for row in rows:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                used_diffs.add(row[j] - row[i])
                    
                    compressed_rows, _ = try_compression(rows, used_diffs, k, rng)
                    if validate(compressed_rows):
                        cm = max_entry(compressed_rows)
                        if cm < best_max:
                            best_max = cm
                            best_rows = [list(r) for r in compressed_rows]
    
    # Final fallback: pure greedy if nothing else worked
    if best_rows is None:
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
    
    return best_rows
# EVOLVE-BLOCK-END
