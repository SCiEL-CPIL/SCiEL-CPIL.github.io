"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    def greedy_row(used, k, max_scan=200):
        """Build one row greedily, trying to keep entries small."""
        row = [0]
        x = 0
        for _ in range(k):
            found = False
            # Try candidates starting from x+1
            for cand in range(x + 1, x + 1 + max_scan):
                new_diffs = {cand - a for a in row}
                if not new_diffs & used:
                    row.append(cand)
                    used |= new_diffs
                    x = cand
                    found = True
                    break
            if not found:
                return None
        return row
    
    def greedy_solution(n_rows, k, max_scan=200):
        """Build a full solution greedily."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used, k, max_scan)
            if row is None:
                return None
            rows.append(row)
        return rows
    
    # Phase 1: Get a baseline greedy solution
    best_rows = greedy_solution(n_rows, k)
    if best_rows is None:
        # Fallback: very simple greedy with larger scan
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) <= k:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
    
    best_scope = max(max(r) for r in best_rows)
    
    # Phase 2: Randomized restarts with different strategies
    # Strategy variations:
    # - Different scan ranges
    # - Shuffled candidate order
    # - Different row ordering (randomize which row we build first)
    
    def randomized_greedy(n_rows, k, seed=None):
        if seed is not None:
            random.seed(seed)
        
        used = set()
        rows = []
        
        # Randomize the order in which we build rows
        row_order = list(range(n_rows))
        random.shuffle(row_order)
        
        for row_idx in row_order:
            row = [0]
            x = 0
            success = True
            for _ in range(k):
                found = False
                # Try candidates in a slightly randomized order
                base = x + 1
                # First try a window of candidates
                window_size = random.choice([20, 50, 100, 150])
                candidates = list(range(base, base + window_size))
                random.shuffle(candidates)
                
                for cand in candidates:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        used |= new_diffs
                        x = cand
                        found = True
                        break
                
                if not found:
                    # Try larger values
                    for cand in range(base + window_size, base + window_size + 300):
                        new_diffs = {cand - a for a in row}
                        if not new_diffs & used:
                            row.append(cand)
                            used |= new_diffs
                            x = cand
                            found = True
                            break
                
                if not found:
                    success = False
                    break
            
            if not success:
                return None
            
            rows.append(row)
        
        return rows
    
    # Phase 3: Local search / improvement
    def try_improve(rows, used, k, time_limit=5.0):
        """Try to improve the scope by moving entries down."""
        start = time.time()
        best_scope = max(max(r) for r in rows)
        
        # Build a fresh used set from current rows
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try to reduce the largest entry
            max_val = max(max(r) for r in rows)
            for row_idx, row in enumerate(rows):
                if row[-1] < max_val:
                    continue
                # Try to move the last element down
                current_last = row[-1]
                for new_val in range(current_last - 1, row[-2] if len(row) > 1 else 0, -1):
                    if new_val <= row[-2]:
                        break
                    # Check if moving is valid
                    new_diffs = {new_val - a for a in row[:-1]}
                    old_diffs = {current_last - a for a in row[:-1]}
                    
                    # Remove old diffs, add new diffs
                    # But we need to check against all other rows' diffs
                    all_other_diffs = set()
                    for other_idx, other_row in enumerate(rows):
                        if other_idx == row_idx:
                            continue
                        for i in range(len(other_row)):
                            for j in range(i + 1, len(other_row)):
                                all_other_diffs.add(other_row[j] - other_row[i])
                    
                    # Check new diffs don't conflict
                    if not (new_diffs & all_other_diffs):
                        # Also check new diffs don't conflict with each other (they can't since row is increasing)
                        row[-1] = new_val
                        improved = True
                        break
                if improved:
                    break
        
        # Recompute used
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        return max(max(r) for r in rows)
    
    # Run randomized restarts
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 500:
        attempts += 1
        
        rows = randomized_greedy(n_rows, k, seed=attempts)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                # Try to improve further
                if time.time() < deadline - 15.0:
                    improved_scope = try_improve(best_rows, set(), k, time_limit=3.0)
                    if improved_scope < best_scope:
                        best_scope = improved_scope
                        best_rows = best_rows  # Already modified in place
                if best_scope <= 112:
                    break
    
    return best_rows
# EVOLVE-BLOCK-END
