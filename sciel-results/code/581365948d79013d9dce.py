"""
Difference triangle set (7,5) solver.

Strategy:
1. Greedy construction to get a valid baseline.
2. Local search (simulated annealing style) to minimize the largest entry.
   - State: list of rows, each row is a list of k+1 integers starting at 0.
   - Move: pick a random row, pick a random position j (1..k), change that entry
     to a new value (either +/- small random step, or random value in a range).
     Then re-sort the row and check validity.
   - Score: if valid, score = max entry. If invalid, score = max entry + penalty * (number of conflicts).
   - Accept worse solutions with probability exp(-delta/T).

Key insight: We want to minimize the maximum entry across all rows. The greedy
approach tends to spread things out, but local search can find tighter packings.
"""

import time
import random
import math

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # --- Phase 1: Greedy construction for initial valid solution ---
    rows = greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # --- Phase 2: Local search / Simulated Annealing ---
    # We'll try to reduce the maximum entry.
    # State representation: list of rows (each sorted, starting with 0).
    # Move: modify one element in one row, then re-sort and validate.
    
    current_rows = [r[:] for r in rows]
    current_max = best_score
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.1
    n_steps = 50000  # will be limited by time
    
    # Precompute: for each row, the set of differences
    def compute_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def all_diffs_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def count_conflicts(rows):
        used = set()
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        conflicts += 1
                    used.add(d)
        return conflicts
    
    # Simulated annealing
    T = T_start
    cooling = (T_start / T_end) ** (1.0 / max(1, n_steps))
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Choose a random row and position to modify
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)  # positions 1..k (position 0 is always 0)
        
        old_val = current_rows[row_idx][pos]
        
        # Propose new value
        # Try values in a neighborhood
        step_size = random.randint(1, 15)
        if random.random() < 0.5:
            new_val = old_val + step_size
        else:
            new_val = old_val - step_size
        
        # Ensure new_val > previous element and < next element (after sorting)
        # Actually, let's just set it and re-sort, then check validity
        new_row = current_rows[row_idx][:]
        new_row[pos] = new_val
        
        # Re-sort the row (keeping 0 at start)
        new_row = sorted(new_row)
        
        # Check if row is strictly increasing and starts at 0
        if new_row[0] != 0 or any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
            # Invalid row, reject
            continue
        
        # Compute new max
        new_max = max(max(r) for r in current_rows[:row_idx] + [new_row] + current_rows[row_idx+1:])
        
        # Check validity
        temp_rows = current_rows[:row_idx] + [new_row] + current_rows[row_idx+1:]
        conflicts = count_conflicts(temp_rows)
        
        if conflicts == 0:
            new_score = new_max
        else:
            new_score = new_max + conflicts * 10  # penalty
        
        old_score = current_max + count_conflicts(current_rows) * 10
        
        delta = new_score - old_score
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_rows[row_idx] = new_row
            current_max = new_max
            
            if conflicts == 0 and current_max < best_score:
                best_score = current_max
                best_rows = [r[:] for r in current_rows]
        
        # Cool down
        T *= cooling
        if T < T_end:
            T = T_end
    
    return best_rows


def greedy_construct(n_rows, k):
    """Greedy construction: each row takes the smallest next mark whose differences
    to earlier marks are all unused."""
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
