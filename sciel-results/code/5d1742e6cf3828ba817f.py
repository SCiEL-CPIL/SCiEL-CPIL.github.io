import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
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
    
    def compute_diffs_set(rows):
        """Return the set of all differences, or None if duplicate found."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return None
                    used.add(d)
        return used
    
    def greedy_construct(bias=1.0):
        """Greedy construction with optional randomization bias."""
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
    
    def greedy_random_construct(window_size=30, bias_power=2.0):
        """
        Randomized greedy: for each position, pick from valid candidates
        in a window, with bias towards smaller values (1/x^bias_power).
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Find valid candidates in a window
                candidates = []
                x = current_max + 1
                max_x = current_max + window_size
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Extend window
                    while x <= max_x + 100:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            candidates.append(x)
                        x += 1
                
                if not candidates:
                    return None  # Failed
                
                # Pick with bias towards smaller values
                if len(candidates) == 1:
                    next_x = candidates[0]
                else:
                    # Weight: 1/(x - current_max)^bias_power
                    weights = [(1.0 / (c - current_max) ** bias_power) for c in candidates]
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
            
            rows.append(row)
        
        if is_valid(rows):
            return rows
        return None
    
    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope
    
    # Phase 2: Randomized greedy with multiple restarts and diversification
    num_restarts = 0
    # Try different window sizes and bias powers
    configs = [
        (20, 1.0), (30, 1.5), (40, 2.0), (50, 2.5), (60, 3.0),
        (25, 1.2), (35, 1.8), (45, 2.2), (55, 2.8), (70, 3.5),
    ]
    random.shuffle(configs)
    
    while time.time() < deadline - 15:
        window_size, bias_power = configs[num_restarts % len(configs)]
        rows = greedy_random_construct(window_size, bias_power)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        num_restarts += 1
    
    # Phase 3: Local search / simulated annealing on the best solution
    if best_rows is not None and time.time() < deadline - 3:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_diffs = compute_diffs_set(current_rows)
        
        # Temperature schedule
        initial_temp = max(1.0, (current_scope - lower_bound) / 5.0)
        temperature = initial_temp
        cooling_rate = 0.999
        
        def try_modify(rows, diffs_set):
            """Try a random modification; return (new_rows, new_diffs) if valid, else (None, None)."""
            new_rows = [r[:] for r in rows]
            mod_type = random.choice(['shift', 'replace', 'swap', 'insert_remove', 'shift_multi'])
            
            row_idx = random.randint(0, len(new_rows) - 1)
            row = new_rows[row_idx]
            
            if mod_type == 'shift':
                if len(row) > 1:
                    idx = random.randint(1, len(row) - 1)
                    delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                    new_val = row[idx] + delta
                    if new_val > row[idx - 1] and (idx == len(row) - 1 or new_val < row[idx + 1]):
                        # Check validity efficiently
                        old_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                old_diffs.add(row[j] - row[i])
                        
                        new_row = row[:]
                        new_row[idx] = new_val
                        new_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                new_diffs.add(new_row[j] - new_row[i])
                        
                        # Check if any new diff collides with other rows' diffs
                        other_diffs = diffs_set - old_diffs
                        if new_diffs & other_diffs:
                            return None, None
                        
                        new_rows[row_idx] = new_row
                        new_all_diffs = other_diffs | new_diffs
                        if len(new_all_diffs) == total_diffs:
                            return new_rows, new_all_diffs
                        return None, None
            
            elif mod_type == 'replace':
                if len(row) > 1:
                    idx = random.randint(1, len(row) - 1)
                    # Try values in a range around current value
                    for new_val in range(row[idx] - 15, row[idx] + 16):
                        if new_val <= row[idx - 1] or (idx < len(row) - 1 and new_val >= row[idx + 1]):
                            continue
                        old_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                old_diffs.add(row[j] - row[i])
                        
                        new_row = row[:]
                        new_row[idx] = new_val
                        new_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                new_diffs.add(new_row[j] - new_row[i])
                        
                        other_diffs = diffs_set - old_diffs
                        if new_diffs & other_diffs:
                            continue
                        
                        new_all_diffs = other_diffs | new_diffs
                        if len(new_all_diffs) == total_diffs:
                            new_rows[row_idx] = new_row
                            return new_rows, new_all_diffs
                    return None, None
            
            elif mod_type == 'swap':
                if len(row) > 2:
                    i, j = sorted(random.sample(range(1, len(row)), 2))
                    row[i], row[j] = row[j], row[i]
                    if all(row[t] < row[t + 1] for t in range(len(row) - 1)):
                        old_diffs = set()
                        for r in new_rows:
                            for a in range(len(r)):
                                for b in range(a + 1, len(r)):
                                    old_diffs.add(r[b] - r[a])
                        
                        # Recompute after swap
                        new_diffs_all = set()
                        valid = True
                        for r in new_rows:
                            for a in range(len(r)):
                                for b in range(a + 1, len(r)):
                                    d = r[b] - r[a]
                                    if d in new_diffs_all:
                                        valid = False
                                        break
                                    new_diffs_all.add(d)
                            if not valid:
                                break
                        
                        if valid and len(new_diffs_all) == total_diffs:
                            return new_rows, new_diffs_all
                    row[i], row[j] = row[j], row[i]  # Revert
                    return None, None
            
            elif mod_type == 'insert_remove':
                if len(row) > 2:
                    remove_idx = random.randint(1, len(row) - 2)
                    saved = row.pop(remove_idx)
                    for add_val in range(row[-1] + 1, row[-1] + 20):
                        row.append(add_val)
                        # Check validity
                        valid = True
                        new_diffs_all = set()
                        for r in new_rows:
                            for a in range(len(r)):
                                for b in range(a + 1, len(r)):
                                    d = r[b] - r[a]
                                    if d in new_diffs_all:
                                        valid = False
                                        break
                                    new_diffs_all.add(d)
                            if not valid:
                                break
                        
                        if valid and len(new_diffs_all) == total_diffs:
                            return new_rows, new_diffs_all
                        row.pop()
                    row.insert(remove_idx, saved)
                    return None, None
            
            elif mod_type == 'shift_multi':
                # Shift multiple elements in a row
                if len(row) > 2:
                    num_shifts = random.randint(1, min(3, len(row) - 1))
                    indices = random.sample(range(1, len(row)), num_shifts)
                    deltas = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(num_shifts)]
                    
                    new_row = row[:]
                    for idx, d in zip(indices, deltas):
                        new_row[idx] += d
                    
                    if all(new_row[t] < new_row[t + 1] for t in range(len(new_row) - 1)):
                        old_diffs = set()
                        for r in new_rows:
                            for a in range(len(r)):
                                for b in range(a + 1, len(r)):
                                    old_diffs.add(r[b] - r[a])
                        
                        new_rows[row_idx] = new_row
                        new_diffs_all = set()
                        valid = True
                        for r in new_rows:
                            for a in range(len(r)):
                                for b in range(a + 1, len(r)):
                                    d = r[b] - r[a]
                                    if d in new_diffs_all:
                                        valid = False
                                        break
                                    new_diffs_all.add(d)
                            if not valid:
                                break
                        
                        if valid and len(new_diffs_all) == total_diffs:
                            return new_rows, new_diffs_all
                    return None, None
            
            return None, None
        
        steps = 0
        while time.time() < deadline - 3:
            modified, new_diffs = try_modify(current_rows, current_diffs)
            if modified is not None:
                new_scope = get_scope(modified)
                delta_scope = new_scope - current_scope
                
                if delta_scope < 0:
                    # Better solution
                    current_rows = modified
                    current_scope = new_scope
                    current_diffs = new_diffs
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in current_rows]
                        if best_scope <= lower_bound:
                            return best_rows
                elif random.random() < math.exp(-delta_scope / max(temperature, 0.1)):
                    current_rows = modified
                    current_scope = new_scope
                    current_diffs = new_diffs
            
            temperature *= cooling_rate
            steps += 1
            
            # Occasionally restart from a fresh greedy solution
            if steps % 300 == 0 and random.random() < 0.15:
                fresh = greedy_construct()
                if is_valid(fresh):
                    fresh_scope = get_scope(fresh)
                    if fresh_scope < current_scope:
                        current_rows = fresh
                        current_scope = fresh_scope
                        current_diffs = compute_diffs_set(fresh)
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]
            
            # Also try a fresh randomized greedy
            if steps % 300 == 0 and random.random() < 0.15:
                ws, bp = configs[random.randint(0, len(configs) - 1)]
                fresh = greedy_random_construct(ws, bp)
                if fresh is not None:
                    fresh_scope = get_scope(fresh)
                    if fresh_scope < current_scope:
                        current_rows = fresh
                        current_scope = fresh_scope
                        current_diffs = compute_diffs_set(fresh)
                        if fresh_scope < best_scope:
                            best_scope = fresh_scope
                            best_rows = [r[:] for r in current_rows]
    
    return best_rows
