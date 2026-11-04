import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2  # C(k+1, 2)
    total_diffs = n_rows * diffs_per_row
    
    # Counting bound: need total_diffs distinct values in [1, M]
    # So M >= total_diffs
    counting_bound = total_diffs
    
    best_result = None
    best_scope = float('inf')
    
    def scope_of(rows):
        return max(max(row) for row in rows)
    
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
    
    # Pure greedy baseline
    def pure_greedy():
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
    
    # Get initial solution
    initial = pure_greedy()
    best_result = initial
    best_scope = scope_of(initial)
    
    # Randomized greedy with different orderings
    def randomized_greedy(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        # Randomize the order in which we build rows
        # But we must build all rows
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # With some probability, accept a slightly suboptimal choice
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Better: use a scoring function to guide choices
    def scored_greedy(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # Score: prefer smaller x, but also consider how "spread out" the new diffs are
                    # to leave room for future rows
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Local search: try to improve by modifying individual elements
    def local_search(rows, seed, max_iters=5000):
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        current_scope = scope_of(rows)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random row and a random position to try to reduce
            row_idx = rng.randrange(n_rows)
            # Don't change position 0 (must be 0)
            pos = rng.randrange(1, k + 1)
            
            original_val = rows[row_idx][pos]
            
            # Try to reduce this value
            # Find the smallest valid value that keeps the row increasing and all diffs unique
            min_possible = rows[row_idx][pos - 1] + 1
            max_possible = original_val - 1 if pos < k else original_val  # If last, can try same or less
            
            # Actually, we want to try values from min_possible up to original_val
            # But we need to maintain strict increase and all diffs unique
            found = False
            for new_val in range(min_possible, original_val + 1):
                # Check if this maintains strict increase
                if new_val <= rows[row_idx][pos - 1]:
                    continue
                if pos < k and new_val >= rows[row_idx][pos + 1]:
                    continue
                
                # Temporarily set and check validity
                rows[row_idx][pos] = new_val
                
                # Check all diffs in this row are still unique within the row
                row = rows[row_idx]
                row_diffs = set()
                valid = True
                for i in range(k + 1):
                    for j in range(i + 1, k + 1):
                        d = row[j] - row[i]
                        if d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check no diff conflicts with other rows
                all_diffs = set()
                for r_i in range(n_rows):
                    if r_i == row_idx:
                        continue
                    r = rows[r_i]
                    for i in range(k + 1):
                        for j in range(i + 1, k + 1):
                            d = r[j] - r[i]
                            if d in row_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    break
            
            if found:
                new_scope = scope_of(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_result = [list(r) for r in rows]
            else:
                # Restore original
                rows[row_idx][pos] = original_val
        
        return rows, current_scope
    
    # Try multiple randomized approaches
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Try local search from greedy solution
        rows, scope = local_search(initial, seed=attempt, max_iters=2000)
        if scope < best_scope:
            best_scope = scope
            best_result = [list(r) for r in rows]
    
    # If we found something better, return it; otherwise return the greedy
    if best_result is not None:
        return best_result
    return initial
