"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Uses force-directed layout: each row is a point in Z^k (coordinates a_1..a_k),
    and we iteratively push rows apart to minimize overlap in difference space.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    num_diffs_per_row = k * (k + 1) // 2
    
    def row_diffs(row):
        """Get the set of positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def max_entry(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct across rows."""
        used = set()
        for r in rows:
            diffs = row_diffs(r)
            if diffs & used:
                return False
            used |= diffs
        return True
    
    def greedy_row(used_diffs, max_val=None):
        """Build a single row greedily, avoiding used differences."""
        row = [0]
        used_local = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            if max_val is not None and x > max_val:
                return None, None
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs) and not (new_diffs & used_local):
                row.append(x)
                used_local |= new_diffs
        return row, used_local
    
    def diff_overlap_count(diffs_a, diffs_b):
        """Count how many differences are shared."""
        return len(diffs_a & diffs_b)
    
    # Force-directed layout approach:
    # 1. Initialize rows using greedy with randomization
    # 2. Iteratively adjust rows to reduce difference overlap and max entry
    # 3. Use simulated annealing to accept worse solutions temporarily
    
    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Generate initial solutions using randomized greedy
    for attempt in range(50):
        random.seed(attempt)
        used = set()
        rows = []
        valid = True
        
        for i in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    # With some probability, accept; otherwise skip to explore
                    if random.random() < 0.6:
                        row.append(x)
                        used |= new_diffs
                    else:
                        # Skip this x, try next
                        pass
                if x > 200:  # Safety bound
                    valid = False
                    break
            if not valid:
                break
            rows.append(row)
        
        if not valid or len(rows) != n_rows:
            continue
        
        if not is_valid(rows):
            continue
        
        cur_max = max_entry(rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]
    
    # Phase 2: Force-directed optimization
    # Represent each row as a vector of k coordinates (a_1, ..., a_k)
    # Force = repulsion from other rows based on difference overlap
    
    if best_rows is None:
        # Fallback: simple greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row, row_diffs_set = greedy_row(used)
            best_rows.append(row)
            used |= row_diffs_set
        best_max = max_entry(best_rows)
    
    # Force-directed local search
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Compute initial difference sets
    current_diffs = [row_diffs(r) for r in current_rows]
    
    # Simulated annealing parameters
    temperature = 10.0
    cooling_rate = 0.95
    max_iterations = 5000
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > 110:  # Leave time for finalization
            break
        
        # Pick a random row to move
        idx = random.randint(0, n_rows - 1)
        
        # Save old state
        old_row = current_rows[idx][:]
        old_diffs = current_diffs[idx]
        old_max = current_max
        
        # Generate a perturbation: try to reduce max entry
        # Strategy: rebuild this row with a tighter constraint
        used_other = set()
        for j in range(n_rows):
            if j != idx:
                used_other |= current_diffs[j]
        
        # Try to build a new row with smaller max
        new_row, new_diffs = greedy_row(used_other, max_val=current_max - 1)
        
        if new_row is None:
            # Try with current max
            new_row, new_diffs = greedy_row(used_other, max_val=current_max)
        
        if new_row is None:
            continue
        
        new_max = max(max(r) for r in current_rows[:idx] + [new_row] + current_rows[idx+1:])
        
        # Accept if better, or with probability based on temperature
        if new_max < current_max:
            current_rows[idx] = new_row
            current_diffs[idx] = new_diffs
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in current_rows]
        elif new_max == current_max:
            # Accept equal solutions with some probability to explore
            if random.random() < 0.3:
                current_rows[idx] = new_row
                current_diffs[idx] = new_diffs
        else:
            # Worse solution: accept with probability exp(-(new_max - current_max) / temperature)
            delta = new_max - current_max
            if delta < temperature and random.random() < math.exp(-delta / temperature):
                current_rows[idx] = new_row
                current_diffs[idx] = new_diffs
                current_max = new_max
        
        # Cool down
        temperature *= cooling_rate
    
    # Phase 3: Final local search - try swapping last elements
    for _ in range(30):
        if time.time() - start_time > 115:
            break
        
        rows = [r[:] for r in best_rows]
        used = set()
        for r in rows:
            used |= row_diffs(r)
        
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                if rows[i][-1] != rows[j][-1]:
                    a, b = rows[i][-1], rows[j][-1]
                    rows[i][-1], rows[j][-1] = b, a
                    
                    new_used = set()
                    valid = True
                    for r in rows:
                        diffs = row_diffs(r)
                        if diffs & new_used:
                            valid = False
                            break
                        new_used |= diffs
                    
                    if valid:
                        new_max = max_entry(rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [r[:] for r in rows]
                    else:
                        rows[i][-1], rows[j][-1] = a, b
    
    return best_rows
# EVOLVE-BLOCK-END
