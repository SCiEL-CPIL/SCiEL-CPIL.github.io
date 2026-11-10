import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    num_elements = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * total_diffs_per_row
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    def elapsed():
        return time.time() - start_time
    
    def deadline_check():
        return elapsed() > deadline - 3
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def compute_diffs_mask(rows):
        mask = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    mask |= (1 << (d - 1))
        return mask
    
    def is_valid_mask(rows, mask):
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if mask & (1 << (d - 1)):
                        return False
                    mask |= (1 << (d - 1))
        return True
    
    def solve_greedy(used_mask=0):
        """Simple greedy: pick smallest valid next element for each row."""
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << (d - 1)
                    if (used_mask & bit) or (new & bit):
                        valid = False
                        break
                    new |= bit
                if valid:
                    row.append(x)
                    used_mask |= new
            rows.append(row)
        return rows, used_mask
    
    # Phase 1: Greedy baseline
    greedy_rows, greedy_mask = solve_greedy(0)
    best_rows = greedy_rows
    best_scope = get_scope(greedy_rows)
    
    # Phase 2: Randomized backtracking with bitmask
    def build_row_backtrack(used_mask, row, pos, current_max, rng, max_window, deadline_fn):
        """Backtracking to build one row given current used_mask."""
        if deadline_fn():
            return None
        if pos == num_elements:
            return row[:]
        
        start_val = current_max + 1
        # Generate candidates with some randomness
        candidates = list(range(start_val, start_val + max_window))
        # Weight towards smaller values but allow exploration
        if rng.random() < 0.3:
            rng.shuffle(candidates)
        else:
            # Partial shuffle: keep first few in order, shuffle the rest
            split = min(len(candidates), 10)
            if len(candidates) > split:
                rng.shuffle(candidates[split:])
        
        for x in candidates:
            if deadline_fn():
                return None
            
            new_diffs = 0
            valid = True
            for a in row:
                d = x - a
                bit = 1 << (d - 1)
                if (used_mask & bit) or (new_diffs & bit):
                    valid = False
                    break
                new_diffs |= bit
            
            if not valid:
                continue
            
            row.append(x)
            result = build_row_backtrack(used_mask | new_diffs, row, pos + 1, x, rng, max_window, deadline_fn)
            row.pop()
            
            if result is not None:
                return result
        
        return None
    
    # Try multiple randomized constructions
    max_restarts = 200
    for attempt in range(max_restarts):
        if deadline_check():
            break
        
        rng = random.Random(attempt * 7919 + 42)
        used_mask = 0
        rows = []
        success = True
        
        # Build rows in order of increasing difficulty (heuristic: later rows are harder)
        # Try different orderings
        order = list(range(n_rows))
        if attempt % 3 == 1:
            rng.shuffle(order)
        elif attempt % 3 == 2:
            order = list(reversed(order))
        
        for row_idx in order:
            if deadline_check():
                success = False
                break
            
            row = [0]
            max_window = 100 + (attempt % 50)
            
            result = build_row_backtrack(used_mask, row, 1, 0, rng, max_window, deadline_check)
            
            if result is None:
                success = False
                break
            
            # Update used_mask with this row's differences
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    d = result[j] - result[i]
                    used_mask |= (1 << (d - 1))
            
            rows.append(result)
        
        if success and len(rows) == n_rows:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
    
    # Phase 3: Local search refinement
    if best_rows is not None and not deadline_check():
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_mask = compute_diffs_mask(current_rows)
        
        temperature = max(1.0, (current_scope - lower_bound) / 5.0)
        cooling_rate = 0.997
        no_improve_count = 0
        max_no_improve = 3000
        
        while not deadline_check() and no_improve_count < max_no_improve:
            row_idx = random.randint(0, len(current_rows) - 1)
            row = current_rows[row_idx]
            
            if random.random() < 0.75 and len(row) > 1:
                # Move type 1: Shift one element
                idx = random.randint(1, len(row) - 1)
                delta = random.randint(-30, 30)
                new_val = row[idx] + delta
                
                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                
                # Compute old and new differences involving this element
                old_diffs = set()
                new_diffs = set()
                for j in range(idx):
                    old_diffs.add(row[idx] - row[j])
                    new_diffs.add(new_val - row[j])
                for j in range(idx + 1, len(row)):
                    old_diffs.add(row[j] - row[idx])
                    new_diffs.add(row[j] - new_val)
                
                # Check if new_diffs conflict with other diffs
                other_mask = current_mask
                for d in old_diffs:
                    other_mask &= ~(1 << (d - 1))
                
                valid = True
                for d in new_diffs:
                    if other_mask & (1 << (d - 1)):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Valid move - compute new scope
                new_rows = [r[:] for r in current_rows]
                new_rows[row_idx][idx] = new_val
                new_scope = get_scope(new_rows)
                
                # Accept or reject based on simulated annealing
                if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                    current_rows = new_rows
                    current_scope = new_scope
                    current_mask = compute_diffs_mask(current_rows)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Move type 2: Replace entire row
                other_mask = 0
                for i, r in enumerate(current_rows):
                    if i == row_idx:
                        continue
                    for a in range(len(r)):
                        for b in range(a + 1, len(r)):
                            d = r[b] - r[a]
                            other_mask |= (1 << (d - 1))
                
                # Try to build a new row avoiding other_mask
                new_row = [0]
                current_max = 0
                success = True
                while len(new_row) < num_elements:
                    candidates = []
                    x = current_max + 1
                    max_x = current_max + max(80, k * 8)
                    while x <= max_x:
                        new_diffs = 0
                        valid = True
                        for a in new_row:
                            d = x - a
                            bit = 1 << (d - 1)
                            if (other_mask & bit) or (new_diffs & bit):
                                valid = False
                                break
                            new_diffs |= bit
                        if valid:
                            candidates.append((x, new_diffs))
                        x += 1
                    
                    if not candidates:
                        success = False
                        break
                    
                    # Weight towards smaller values
                    if len(candidates) == 1:
                        next_x, next_diffs = candidates[0]
                    else:
                        weights = [1.0 / (c[0] - current_max + 1) ** 0.5 for c in candidates]
                        total_w = sum(weights)
                        r = random.random() * total_w
                        cumulative = 0
                        next_x, next_diffs = candidates[-1]
                        for c, w in zip(candidates, weights):
                            cumulative += w
                            if r <= cumulative:
                                next_x, next_diffs = c
                                break
                    
                    new_row.append(next_x)
                    other_mask |= next_diffs
                    current_max = next_x
                
                if success:
                    new_rows = [r[:] for r in current_rows]
                    new_rows[row_idx] = new_row
                    new_scope = get_scope(new_rows)
                    
                    if new_scope <= current_scope or random.random() < math.exp(-(new_scope - current_scope) / temperature):
                        current_rows = new_rows
                        current_scope = new_scope
                        current_mask = compute_diffs_mask(current_rows)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            temperature *= cooling_rate
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = current_rows
    
    return best_rows
