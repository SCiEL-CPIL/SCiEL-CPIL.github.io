import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Lower bound on scope: M(M+1)/2 >= total_diffs
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
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
    
    def compute_diffs(rows):
        """Return set of all differences in rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Randomized greedy with restarts
    num_restarts = 0
    while time.time() < deadline - 15:  # Leave 15s for local search
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates in a window
                candidates = []
                x = current_max + 1
                max_x = current_max + max(50, k * 5)  # Search window
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Pick randomly from candidates, but bias towards smaller values
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Weight inversely proportional to value
                    weights = [1.0 / (c - current_max) for c in candidates]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    cumulative = 0
                    next_x = candidates[-1]
                    for c, w in zip(candidates, weights):
                        cumulative += w
                        if r <= cumulative:
                            next_x = c
                            break
                
                row.append(next_x)
                for a in row[:-1]:
                    used.add(next_x - a)
                current_max = next_x
            
            if not success:
                break
            rows.append(row)
        
        if success and is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        
        num_restarts += 1
    
    # Phase 3: Local search with row-replacement moves
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        temperature = max(1.0, (current_scope - lower_bound) / 10.0)
        cooling_rate = 0.995
        
        # Precompute all differences for current solution
        current_diffs = compute_diffs(current_rows)
        
        def try_modify(rows, diffs):
            """Try a random modification and return (modified_rows, modified_diffs) if valid, else (None, None)."""
            row_idx = random.randint(0, len(rows) - 1)
            row = rows[row_idx]
            
            # Try shifting one element in the row
            if len(row) > 1:
                idx = random.randint(1, len(row) - 1)
                # Try various deltas
                deltas = list(range(-15, 16))
                random.shuffle(deltas)
                for delta in deltas:
                    new_val = row[idx] + delta
                    # Check ordering
                    if new_val <= row[idx - 1]:
                        continue
                    if idx < len(row) - 1 and new_val >= row[idx + 1]:
                        continue
                    
                    # Compute old and new differences for this element
                    old_diffs = set()
                    new_diffs = set()
                    for j in range(idx):
                        old_diffs.add(row[idx] - row[j])
                        new_diffs.add(new_val - row[j])
                    for j in range(idx + 1, len(row)):
                        old_diffs.add(row[j] - row[idx])
                        new_diffs.add(row[j] - new_val)
                    
                    # Check if new_diffs conflicts with existing diffs (excluding old_diffs)
                    other_diffs = diffs - old_diffs
                    if new_diffs & other_diffs:
                        continue
                    
                    # Valid! Apply the change
                    new_rows = [r[:] for r in rows]
                    new_rows[row_idx][idx] = new_val
                    new_diffs_set = diffs - old_diffs | new_diffs
                    
                    return new_rows, new_diffs_set
            
            # Try replacing an entire row with a fresh greedy row
            other_rows = [r for i, r in enumerate(rows) if i != row_idx]
            other_diffs = set()
            for r in other_rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        other_diffs.add(r[j] - r[i])
            
            # Try to build a new row that avoids other_diffs
            new_row = [0]
            x = 0
            success = True
            while len(new_row) < k + 1:
                x += 1
                new_diffs = {x - a for a in new_row}
                if not (new_diffs & other_diffs):
                    new_row.append(x)
                else:
                    # If stuck, try a random offset
                    x = random.randint(1, 50)
                    new_row = [0]
                    success = True
                    break
            
            if len(new_row) < k + 1:
                # Retry with randomization
                for attempt in range(5):
                    new_row = [0]
                    current_max = 0
                    while len(new_row) < k + 1:
                        candidates = []
                        xx = current_max + 1
                        max_xx = current_max + max(30, k * 3)
                        while xx <= max_xx:
                            new_diffs = {xx - a for a in new_row}
                            if not (new_diffs & other_diffs):
                                candidates.append(xx)
                            xx += 1
                        if not candidates:
                            break
                        # Random pick with bias to smaller
                        if len(candidates) == 1:
                            next_x = candidates[0]
                        else:
                            weights = [1.0 / (c - current_max) for c in candidates]
                            total_w = sum(weights)
                            r = random.random() * total_w
                            cumulative = 0
                            next_x = candidates[-1]
                            for c, w in zip(candidates, weights):
                                cumulative += w
                                if r <= cumulative:
                                    next_x = c
                                    break
                        
                        new_row.append(next_x)
                        for a in new_row[:-1]:
                            other_diffs.add(next_x - a)
                        current_max = next_x
                    
                    if len(new_row) == k + 1:
                        break
            
            if len(new_row) == k + 1:
                # Verify the new row doesn't conflict with other rows
                new_row_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_row_diffs.add(new_row[j] - new_row[i])
                
                if not (new_row_diffs & other_diffs):
                    new_rows = [r[:] for r in rows]
                    new_rows[row_idx] = new_row
                    new_diffs_set = (diffs - compute_diffs([rows[row_idx]])) | new_row_diffs
                    return new_rows, new_diffs_set
            
            # Try swapping two elements in the same row
            if len(row) > 2:
                i, j = sorted(random.sample(range(1, len(row)), 2))
                if row[i] == row[j]:
                    return None, None
                
                # Check if swap maintains ordering
                if i > 0 and row[j] <= row[i-1]:
                    return None, None
                if j < len(row) - 1 and row[i] >= row[j+1]:
                    return None, None
                if i > 0 and j > i and row[i] <= row[j-1]:
                    return None, None
                if i > 0 and j > i and row[j] >= row[i+1]:
                    return None, None
                
                # Compute differences affected by swap
                old_diffs = set()
                new_diffs = set()
                
                # Element i changes from row[i] to row[j]
                for t in range(len(row)):
                    if t == i or t == j:
                        continue
                    if t < i:
                        old_diffs.add(row[i] - row[t])
                        new_diffs.add(row[j] - row[t])
                    elif t > i:
                        old_diffs.add(row[t] - row[i])
                        new_diffs.add(row[t] - row[j])
                
                # Element j changes from row[j] to row[i]
                for t in range(len(row)):
                    if t == i or t == j:
                        continue
                    if t < j:
                        old_diffs.add(row[j] - row[t])
                        new_diffs.add(row[i] - row[t])
                    elif t > j:
                        old_diffs.add(row[t] - row[j])
                        new_diffs.add(row[t] - row[i])
                
                # Difference between i and j stays the same
                old_diffs.add(row[j] - row[i])
                new_diffs.add(row[i] - row[j])
                
                other_diffs = diffs - old_diffs
                if new_diffs & other_diffs:
                    return None, None
                
                new_rows = [r[:] for r in rows]
                new_rows[row_idx][i], new_rows[row_idx][j] = new_rows[row_idx][j], new_rows[row_idx][i]
                new_diffs_set = diffs - old_diffs | new_diffs
                
                return new_rows, new_diffs_set
            
            return None, None
        
        steps = 0
        while time.time() < deadline - 2:
            modified_rows, modified_diffs = try_modify(current_rows, current_diffs)
            if modified_rows is not None:
                new_scope = get_scope(modified_rows)
                if new_scope < current_scope:
                    current_rows = modified_rows
                    current_diffs = modified_diffs
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                elif random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_rows = modified_rows
                    current_diffs = modified_diffs
                    current_scope = new_scope
            
            temperature *= cooling_rate
            steps += 1
            
            # Occasionally restart local search from a fresh greedy solution
            if steps % 200 == 0 and random.random() < 0.1:
                fresh = greedy_construct()
                if is_valid(fresh):
                    fresh_scope = get_scope(fresh)
                    if fresh_scope < current_scope:
                        current_rows = fresh
                        current_diffs = compute_diffs(fresh)
                        current_scope = fresh_scope
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]
    
    return best_rows
