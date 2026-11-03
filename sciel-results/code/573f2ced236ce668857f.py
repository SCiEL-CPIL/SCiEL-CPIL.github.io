"""Difference triangle set (7,5) - randomized local search with simulated annealing."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Phase 1: Generate initial valid configuration using greedy
    def greedy_initial(n_rows, k):
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
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def get_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.append(d)
        return True
    
    # Start with greedy solution
    rows = greedy_initial(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(best_rows)
    
    # Simulated annealing
    def random_neighbor(rows, k):
        """Generate a random valid neighbor by perturbing one row."""
        new_rows = [r[:] for r in rows]
        ri = random.randint(0, n_rows - 1)
        
        # Try to modify row ri
        row = new_rows[ri]
        # Pick a position to modify (not position 0)
        pos = random.randint(1, len(row) - 1)
        
        # Try several random values
        current_val = row[pos]
        for attempt in range(50):
            delta = random.randint(-10, 10)
            new_val = current_val + delta
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Check if this creates duplicate differences within the row
            new_row = row[:]
            new_row[pos] = new_val
            
            # Check internal differences of this row
            internal_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d in internal_diffs:
                        break
                    internal_diffs.add(d)
            else:
                # Check against all other rows
                conflict = False
                for other_i in range(n_rows):
                    if other_i == ri:
                        continue
                    other_row = new_rows[other_i]
                    for i in range(len(other_row)):
                        for j in range(i + 1, len(other_row)):
                            if other_row[j] - other_row[i] in internal_diffs:
                                conflict = True
                                break
                        if conflict:
                            break
                    if conflict:
                        break
                
                if not conflict:
                    new_rows[ri] = new_row
                    return new_rows
        
        return None
    
    def try_reduce_scope(rows, k, target_scope):
        """Try to find a configuration with scope <= target_scope."""
        # For each row, try to construct it with max entry <= target_scope
        used = set()
        result_rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                for candidate in range(x + 1, target_scope + 1):
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        row.append(candidate)
                        used |= new_diffs
                        x = candidate
                        found = True
                        break
                if not found:
                    return None
            result_rows.append(row)
        
        return result_rows
    
    # Iteratively try to reduce the scope
    current_scope = best_scope
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try to find a solution with scope = current_scope - 1
        target = current_scope - 1
        if target < 0:
            break
        
        found = False
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Try greedy with shuffled order or randomized choices
            used = set()
            trial_rows = []
            success = True
            
            # Randomize which row to build first
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            
            # Build rows in random order
            row_results = [None] * n_rows
            for idx in row_order:
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Try candidates in slightly randomized order near the minimum
                    candidates = list(range(x + 1, target + 1))
                    # Shuffle a window around the minimum
                    if len(candidates) > 10:
                        window = min(10, len(candidates))
                        rng.shuffle(candidates[:window])
                    
                    found_candidate = False
                    for candidate in candidates:
                        new_diffs = {candidate - a for a in row}
                        if not new_diffs & used:
                            row.append(candidate)
                            used |= new_diffs
                            x = candidate
                            found_candidate = True
                            break
                    
                    if not found_candidate:
                        success = False
                        break
                
                if not success:
                    break
                row_results[idx] = row
            
            if success:
                # Verify validity
                all_diffs = []
                valid = True
                for r in row_results:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.append(d)
                    if not valid:
                        break
                
                if valid:
                    trial_scope = get_scope(row_results)
                    if trial_scope < best_scope:
                        best_rows = [r[:] for r in row_results]
                        best_scope = trial_scope
                        current_scope = trial_scope
                    found = True
                    break
        
        if not found:
            # Try local search from current best
            if time.time() > deadline:
                break
            
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                neighbor = random_neighbor(best_rows, k)
                if neighbor is None:
                    continue
                
                neighbor_scope = get_scope(neighbor)
                if neighbor_scope < best_scope:
                    best_rows = neighbor
                    best_scope = neighbor_scope
                    current_scope = best_scope
                    improved = True
                    break
            
            if not improved:
                # Try more aggressive: rebuild one row entirely
                for trial in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Remove one row and try to rebuild it with smaller scope
                    ri = rng.randint(0, n_rows - 1)
                    other_rows = [best_rows[i] for i in range(n_rows) if i != ri]
                    
                    # Collect all diffs from other rows
                    used = set()
                    for r in other_rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                used.add(r[j] - r[i])
                    
                    # Try to build row ri with scope < current_scope
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        found_candidate = False
                        for candidate in range(x + 1, current_scope):
                            new_diffs = {candidate - a for a in row}
                            if not new_diffs & used:
                                row.append(candidate)
                                used |= new_diffs
                                x = candidate
                                found_candidate = True
                                break
                        
                        if not found_candidate:
                            break
                    
                    if len(row) == k + 1:
                        new_rows = [best_rows[i] for i in range(n_rows) if i != ri]
                        new_rows.insert(ri, row)
                        new_scope = get_scope(new_rows)
                        if new_scope < best_scope:
                            best_rows = new_rows
                            best_scope = new_scope
                            current_scope = best_scope
                            improved = True
                            break
    
    return best_rows
