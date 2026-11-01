import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Greedy Initialization
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def count_conflicts(rows):
        all_diffs = []
        for r in rows:
            all_diffs.extend(get_diffs(r))
        c = Counter(all_diffs)
        conflicts = 0
        for d, cnt in c.items():
            if cnt > 1:
                conflicts += cnt - 1
        return conflicts

    def get_scope(rows):
        return max(max(r) for r in rows)

    # 2. Best-Fit Row Constructor
    # Given a set of used differences (from other rows), find the "best" row 
    # that minimizes its own max element while avoiding conflicts.
    def construct_best_row(used_diffs_from_others, target_max=None):
        if target_max is None:
            target_max = 200 
            
        best_row = None
        best_score = float('inf')
        
        # Try a few random seeds for the greedy process
        for _ in range(10):
            row = [0]
            current_max = 0
            valid = True
            
            while len(row) < k + 1:
                current_row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        current_row_diffs.add(row[j]-row[i])
                
                x = current_max + 1
                limit = target_max
                
                found = False
                while x <= limit:
                    conflict_ext = False
                    conflict_int = False
                    
                    for a in row:
                        d = x - a
                        if d in used_diffs_from_others:
                            conflict_ext = True
                            break
                        if d in current_row_diffs:
                            conflict_int = True
                            break
                    
                    if not conflict_ext and not conflict_int:
                        row.append(x)
                        current_row_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        found = True
                        break
                    x += 1
                
                if not found:
                    valid = False
                    break
            
            if valid:
                max_val = row[-1]
                if max_val < best_score:
                    best_score = max_val
                    best_row = row[:]
        
        if best_row:
            return best_row
        return None

    # 3. Simulated Annealing
    
    # Initial State
    current_rows = greedy_init()
    current_conflicts = count_conflicts(current_rows)
    current_scope = get_scope(current_rows)
    current_score = current_conflicts * 10000 + current_scope
    
    best_rows = [r[:] for r in current_rows]
    best_score = current_score
    best_conflicts = current_conflicts
    best_scope = current_scope

    # Annealing Parameters
    T = 50.0
    T_min = 0.01
    alpha = 0.995
    
    step = 0
    while time.time() < deadline and T > T_min:
        step += 1
        
        # Generate Neighbor
        new_rows = [r[:] for r in current_rows]
        move_type = random.random()
        
        if move_type < 0.4:
            # Move Type 1: Reconstruct a random row using best-fit against others
            ri = random.randint(0, n_rows - 1)
            used_diffs_from_others = set()
            for j in range(n_rows):
                if j == ri:
                    continue
                used_diffs_from_others.update(get_diffs(new_rows[j]))
            
            # Target max slightly smaller than current scope to encourage reduction
            # Or just current scope
            target_max = current_scope + 5
            new_row = construct_best_row(used_diffs_from_others, target_max=target_max)
            
            if new_row:
                new_rows[ri] = new_row
            else:
                # Fallback: keep old row
                new_rows[ri] = current_rows[ri][:]
                
        elif move_type < 0.7:
            # Move Type 2: Local element adjustment
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            row = new_rows[ri]
            
            # Try to move element ei to a nearby value
            # Range: [row[ei-1]+1, row[ei+1]-1] or [row[ei-1]+1, row[ei]+5] if last
            lower = row[ei-1] + 1
            upper = row[ei+1] - 1 if ei < k else row[ei] + 5
            
            if lower < upper:
                # Random pick in range
                new_val = random.randint(lower, upper)
                old_val = row[ei]
                row[ei] = new_val
                
                # Fix structure: ensure increasing
                valid_struct = True
                for i in range(1, len(row)):
                    if row[i] <= row[i-1]:
                        valid_struct = False
                        break
                
                if not valid_struct:
                    # Revert if structure broken
                    row[ei] = old_val
                    # Try a small shift instead
                    if ei < k:
                        row[ei] = min(row[ei] + 1, row[ei+1]-1)
                        if row[ei] >= row[ei+1]:
                             row[ei] = old_val
                    else:
                        row[ei] = old_val
            else:
                 # No space, try expanding last element if possible
                 if ei == k:
                     new_val = row[ei] + random.randint(1, 3)
                     row[ei] = new_val
                     if row[ei] > current_scope + 10:
                         row[ei] = current_rows[ri][ei]
        else:
            # Move Type 3: Random restart of a row (more aggressive)
            ri = random.randint(0, n_rows - 1)
            # Just randomize the last few elements?
            # For now, treat as similar to Move Type 1 but with higher target
            used_diffs_from_others = set()
            for j in range(n_rows):
                if j == ri:
                    continue
                used_diffs_from_others.update(get_diffs(new_rows[j]))
            
            target_max = current_scope + 20
            new_row = construct_best_row(used_diffs_from_others, target_max=target_max)
            if new_row:
                new_rows[ri] = new_row
            else:
                new_rows[ri] = current_rows[ri][:]

        # Evaluate New State
        new_conflicts = count_conflicts(new_rows)
        new_scope = get_scope(new_rows)
        new_score = new_conflicts * 10000 + new_scope
        
        # Acceptance Criterion
        delta = new_score - current_score
        if delta < 0:
            # Always accept improvement
            current_rows = new_rows
            current_conflicts = new_conflicts
            current_scope = new_scope
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_conflicts = current_conflicts
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Accept with probability exp(-delta / T)
            # Note: delta is in "score units". Conflicts are weighted by 10000.
            # So a single conflict increase is huge.
            # We might want to normalize T or delta.
            # Let's scale T to be relevant to the score difference.
            # If delta is small (scope change), T=50 is fine.
            # If delta is large (conflict change), T=50 is too small.
            # Let's use a dynamic T or just accept if random < exp(-delta/T)
            # But exp(-10000/50) is 0. So we never accept conflict increases.
            # This is good for validity, but might get stuck.
            # To allow escaping local minima in conflicts, we could occasionally accept conflict increases.
            # Let's modify: if new_conflicts > current_conflicts, accept with very low prob
            # unless we are stuck.
            
            # Simple approach:
            # If conflict count is same, use T for scope.
            # If conflict count increases, use a very small T (e.g., 0.1) to make it rare.
            if new_conflicts > current_conflicts:
                prob = math.exp(-delta / 1.0) # Very small
            else:
                prob = math.exp(-delta / T)
            
            if random.random() < prob:
                current_rows = new_rows
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_score = new_score
        
        # Cool down
        T *= alpha
        
        # Check time
        if time.time() > deadline:
            break

    return best_rows
