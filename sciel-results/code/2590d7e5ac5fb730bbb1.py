import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Lower bound: need n_rows * k*(k+1)/2 distinct differences in [1, M-1]
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def try_greedy(seed_val, strategy=0):
        """Randomized greedy with various strategies."""
        random.seed(seed_val)
        used = set()
        rows = []
        
        # Determine row construction order
        if strategy == 0:
            row_order = list(range(n_rows))
        elif strategy == 1:
            row_order = list(range(n_rows))
            random.shuffle(row_order)
        else:
            # Build rows in order of increasing difficulty (later rows first)
            row_order = list(range(n_rows))
            random.shuffle(row_order)
        
        rows = [None] * n_rows
        
        for row_idx in row_order:
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                
                # Strategy-dependent candidate generation
                if strategy == 0:
                    # Mostly sequential with small random perturbation
                    search_start = last + 1
                    search_end = min(last + 60, lb + 100)
                    candidates = list(range(search_start, search_end))
                    # Bias toward smaller values
                    for i in range(len(candidates)):
                        if random.random() < 0.3:
                            candidates[i] += random.randint(0, 3)
                    candidates.sort()
                elif strategy == 1:
                    # More aggressive randomization
                    search_start = last + 1
                    search_end = min(last + 40, lb + 80)
                    candidates = list(range(search_start, search_end))
                    random.shuffle(candidates)
                    # Sort with noise
                    candidates = sorted(candidates, key=lambda x: x + random.random() * 10)
                else:
                    # Jump-based: try larger gaps sometimes
                    search_start = last + 1
                    search_end = min(last + 80, lb + 120)
                    candidates = list(range(search_start, search_end))
                    random.shuffle(candidates)
                
                for x in candidates:
                    if x <= last:
                        continue
                    valid = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                
                if not found:
                    # Wider search
                    for x in range(last + 1, lb + 300):
                        valid = True
                        new_diffs = set()
                        for a in row:
                            d = x - a
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                    if not found:
                        return None, float('inf')
            
            rows[row_idx] = row
        
        scope = get_scope(rows)
        return rows, scope
    
    def compact_rows(rows, max_iter=50):
        """Try to reduce scope by shifting elements left."""
        nonlocal best_scope, best_rows
        if rows is None:
            return None, float('inf')
        
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        scope = get_scope(rows)
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try to shift a random element in a random row to a smaller value
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            
            if len(row) <= 1:
                continue
            
            # Pick a random position to try to move (not position 0)
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to find a smaller valid value
            for new_val in range(row[pos - 1] + 1, old_val):
                # Check if moving this element keeps all differences valid
                # Remove old differences involving this element
                old_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        d = abs(row[pos] - row[i])
                        old_diffs.add(d)
                
                # Tentatively remove
                for d in old_diffs:
                    used.discard(d)
                
                # Check new differences
                valid = True
                new_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        d = abs(new_val - row[i])
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                
                if valid:
                    # Also check that new_val doesn't conflict with existing row structure
                    # (must maintain strictly increasing order)
                    if pos > 0 and new_val <= row[pos - 1]:
                        valid = False
                    if pos < len(row) - 1 and new_val >= row[pos + 1]:
                        valid = False
                
                if valid:
                    row[pos] = new_val
                    used.update(new_diffs)
                    improved = True
                    break
                else:
                    # Restore old differences
                    used.update(old_diffs)
            
            if not improved:
                # Try removing and re-adding a random element
                row_idx = random.randint(0, n_rows - 1)
                row = rows[row_idx]
                if len(row) <= 1:
                    continue
                
                pos = random.randint(1, len(row) - 1)
                removed_val = row[pos]
                
                # Remove differences involving this element
                old_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        d = abs(removed_val - row[i])
                        old_diffs.add(d)
                
                for d in old_diffs:
                    used.discard(d)
                
                # Remove the element
                row.pop(pos)
                
                # Re-insert at same position with a smaller value if possible
                inserted = False
                for new_val in range(row[pos - 1] + 1, removed_val):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
                        d = abs(new_val - row[i])
                        if d in used:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row.insert(pos, new_val)
                        used.update(new_diffs)
                        inserted = True
                        break
                
                if not inserted:
                    # Restore
                    row.insert(pos, removed_val)
                    used.update(old_diffs)
        
        scope = get_scope(rows)
        return rows, scope
    
    def perturb_and_repair(rows, num_perturbations=3):
        """Perturb rows and try to repair."""
        nonlocal best_scope, best_rows
        if rows is None:
            return None, float('inf')
        
        rows = [list(r) for r in rows]
        
        for _ in range(num_perturbations):
            if time.time() > deadline - 3:
                break
            
            # Remove a random element from a random row
            row_idx = random.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) <= 1:
                continue
            
            pos = random.randint(1, len(row) - 1)
            removed_val = row.pop(pos)
            
            # Recompute used set
            used = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used.add(r[j] - r[i])
            
            # Try to re-insert a valid value at or near the old position
            inserted = False
            min_val = row[pos - 1] + 1 if pos > 0 else 1
            max_val = row[pos] - 1 if pos < len(row) else removed_val + 50
            
            for new_val in range(min_val, min(max_val + 1, lb + 100)):
                valid = True
                new_diffs = set()
                for i in range(len(row)):
                    d = abs(new_val - row[i])
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row.insert(pos, new_val)
                    used.update(new_diffs)
                    inserted = True
                    break
            
            if not inserted:
                # Restore
                row.insert(pos, removed_val)
                used = set()
                for r in rows:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            used.add(r[j] - r[i])
        
        scope = get_scope(rows)
        return rows, scope
    
    # Phase 1: Run many diverse greedy attempts
    num_greedy_attempts = 200
    for i in range(num_greedy_attempts):
        if time.time() > deadline - 20:
            break
        
        strategy = i % 3
        rows, scope = try_greedy(i * 104729 + 7, strategy)
        
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
            if best_scope == lb:
                break
    
    # Phase 2: Local improvement on best solution
    if best_rows is not None:
        for i in range(30):
            if time.time() > deadline - 10:
                break
            
            rows, scope = compact_rows(best_rows, max_iter=30)
            if rows is not None and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
                if best_scope == lb:
                    break
    
    # Phase 3: Perturbation and repair
    if best_rows is not None:
        for i in range(20):
            if time.time() > deadline - 5:
                break
            
            rows, scope = perturb_and_repair(best_rows, num_perturbations=5)
            if rows is not None and scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
                # Follow up with compaction
                rows, scope = compact_rows(best_rows, max_iter=20)
                if rows is not None and scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                
                if best_scope == lb:
                    break
    
    # Final validation
    if best_rows is not None:
        # Verify all differences are distinct
        all_diffs = set()
        valid = True
        for row in best_rows:
            if len(row) != k + 1 or row[0] != 0:
                valid = False
                break
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i - 1]:
                    valid = False
                    break
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        valid = False
                        break
                    all_diffs.add(d)
            if not valid:
                break
        
        if not valid:
            # Fallback to a simple greedy
            used = set()
            best_rows = []
            for _ in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    valid_x = True
                    new_diffs = set()
                    for a in row:
                        d = x - a
                        if d in used:
                            valid_x = False
                            break
                        new_diffs.add(d)
                    if valid_x:
                        row.append(x)
                        used.update(new_diffs)
                best_rows.append(row)
    
    return best_rows
