import time
import random
import sys

def construct(n_rows, k):
    deadline = time.time() + 140
    total_diffs = n_rows * (k * (k + 1) // 2)  # e.g., 4*36=144
    
    # Bitset size: differences up to ~256 (we'll cap scope at 256)
    MAX_DIFF = 256
    mask_all = (1 << MAX_DIFF) - 1
    
    def get_diffs(row):
        """Get bitmask of all pairwise differences in a row."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d < MAX_DIFF:
                    mask |= (1 << d)
                else:
                    mask |= (1 << (MAX_DIFF - 1))  # overflow flag
        return mask
    
    def is_valid(rows):
        """Check if all differences across rows are distinct."""
        seen = 0
        for row in rows:
            m = get_diffs(row)
            if seen & m:
                return False
            seen |= m
        return True
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # --- Initial solution via greedy ---
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = 0
                for a in row:
                    d = x - a
                    if d < MAX_DIFF:
                        new |= (1 << d)
                    else:
                        new |= (1 << (MAX_DIFF - 1))
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    best_rows = greedy()
    best_scope = get_scope(best_rows)
    
    # --- Simulated Annealing ---
    def random_row(max_val):
        """Generate a random valid row (increasing, starting at 0, max <= max_val)."""
        if k + 1 > max_val + 1:
            return None
        # Choose k values from 1..max_val
        if k <= max_val:
            vals = sorted(random.sample(range(1, max_val + 1), k))
            return [0] + vals
        return None
    
    def random_solution(max_scope):
        """Generate a random valid solution with scope <= max_scope."""
        for _ in range(200):
            rows = []
            used = 0
            valid = True
            for _r in range(n_rows):
                row = None
                for _attempt in range(50):
                    candidate = random_row(max_scope)
                    if candidate is None:
                        break
                    m = get_diffs(candidate)
                    if not (m & used) and (m & (1 << (MAX_DIFF - 1))) == 0:
                        row = candidate
                        used |= m
                        break
                if row is None:
                    valid = False
                    break
                rows.append(row)
            if valid:
                return rows
        return None
    
    # Try to get a better initial solution
    for scope_try in range(best_scope - 1, best_scope - 20, -1):
        if time.time() > deadline - 30:
            break
        sol = random_solution(scope_try)
        if sol is not None:
            best_rows = sol
            best_scope = get_scope(sol)
    
    # Annealing
    rows = [list(r) for r in best_rows]
    current_scope = best_scope
    T = 5.0
    T_min = 0.01
    cooling = 0.9999
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        
        # Choose a move
        move_type = random.random()
        r_idx = random.randrange(n_rows)
        row = rows[r_idx]
        
        if move_type < 0.3:
            # Shift a mark
            if len(row) > 1:
                pos = random.randrange(1, len(row))
                delta = random.choice([-1, 1]) * random.randint(1, 3)
                new_val = row[pos] + delta
                # Check bounds and ordering
                if new_val > 0 and new_val < row[pos + 1] if pos + 1 < len(row) else new_val > 0:
                    if pos == len(row) - 1:
                        if new_val > row[pos - 1] and new_val > 0:
                            old_val = row[pos]
                            row[pos] = new_val
                            # Check validity
                            if is_valid(rows):
                                new_scope = get_scope(rows)
                                if new_scope <= current_scope:
                                    current_scope = new_scope
                                    if new_scope < best_scope:
                                        best_scope = new_scope
                                        best_rows = [list(r) for r in rows]
                                else:
                                    row[pos] = old_val
                            else:
                                row[pos] = old_val
                    else:
                        if new_val > row[pos - 1] and new_val < row[pos + 1]:
                            old_val = row[pos]
                            row[pos] = new_val
                            if is_valid(rows):
                                new_scope = get_scope(rows)
                                if new_scope <= current_scope:
                                    current_scope = new_scope
                                    if new_scope < best_scope:
                                        best_scope = new_scope
                                        best_rows = [list(r) for r in rows]
                                else:
                                    row[pos] = old_val
                            else:
                                row[pos] = old_val
        
        elif move_type < 0.5:
            # Remove a mark and re-add
            if len(row) > 2:
                pos = random.randrange(1, len(row) - 1)
                removed = row.pop(pos)
                # Try to re-insert somewhere
                for insert_pos in range(1, len(row) + 1):
                    for val in range(max(1, row[insert_pos-1] + 1), min(row[insert_pos] if insert_pos < len(row) else current_scope + 1, current_scope + 1)):
                        row.insert(insert_pos, val)
                        if is_valid(rows):
                            new_scope = get_scope(rows)
                            if new_scope <= current_scope:
                                current_scope = new_scope
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                    best_rows = [list(r) for r in rows]
                            else:
                                row.pop(insert_pos)
                            break
                        else:
                            row.pop(insert_pos)
                    if len(row) == k + 1:
                        break
                # If we didn't successfully re-insert, put it back
                if len(row) < k + 1:
                    row.insert(pos, removed)
        
        elif move_type < 0.6:
            # Replace an entire row
            new_row = random_row(current_scope)
            if new_row is not None:
                m = get_diffs(new_row)
                # Check against other rows
                other_used = 0
                for i in range(n_rows):
                    if i != r_idx:
                        other_used |= get_diffs(rows[i])
                if not (m & other_used) and (m & (1 << (MAX_DIFF - 1))) == 0:
                    rows[r_idx] = new_row
                    new_scope = get_scope(rows)
                    if new_scope <= current_scope:
                        current_scope = new_scope
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                    else:
                        rows[r_idx] = [list(r) for r in best_rows][r_idx]
        
        else:
            # Perturbation: randomly modify 1-2 marks
            for _ in range(random.randint(1, 2)):
                if len(row) > 1:
                    pos = random.randrange(1, len(row))
                    if pos == len(row) - 1:
                        new_val = random.randint(row[pos - 1] + 1, current_scope)
                    else:
                        new_val = random.randint(row[pos - 1] + 1, row[pos + 1] - 1)
                    if new_val > 0:
                        row[pos] = new_val
            
            if is_valid(rows):
                new_scope = get_scope(rows)
                if new_scope <= current_scope:
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                else:
                    rows = [list(r) for r in best_rows]
                    current_scope = best_scope
            else:
                # Accept with probability exp(-delta/T)
                delta = get_scope(rows) - current_scope
                if delta < 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    current_scope = get_scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                else:
                    rows = [list(r) for r in best_rows]
        
        # Cool down
        if step % 100 == 0:
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_rows
