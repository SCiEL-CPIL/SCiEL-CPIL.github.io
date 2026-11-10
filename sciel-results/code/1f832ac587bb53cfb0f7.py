import time
import random
import numpy as np

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    """
    deadline = time.time() + 130  # safety margin before 150s kill
    
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
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Precompute difference sets for fast checking
    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def greedy_construct(n_rows, k, rng, max_val_limit=300, perturb=0.0):
        """
        Greedy construction with optional random perturbation.
        """
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Decide step size
                if perturb > 0 and rng.random() < perturb:
                    step = rng.randint(1, 3)
                else:
                    step = 1
                x += step
                if x > max_val_limit:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def local_search(rows, n_rows, k, rng, max_iters=200):
        """
        Local search: try to reduce the maximum entry by moving elements down.
        """
        rows = [row[:] for row in rows]
        current_max = get_max_rows_max(rows)
        
        # Precompute all differences for each row
        row_diffs = [compute_row_diffs(row) for row in rows]
        all_diffs = set()
        for rd in row_diffs:
            all_diffs |= rd
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try to reduce each element in each row (from last to first)
            for r_idx in range(n_rows):
                for pos in range(k, 0, -1):
                    old_val = rows[r_idx][pos]
                    if pos == 0:
                        continue
                    
                    # Try to reduce this element
                    for new_val in range(old_val - 1, rows[r_idx][pos - 1], -1):
                        # Check if new_val creates a valid configuration
                        # Remove old differences involving position pos
                        old_row = rows[r_idx]
                        old_diffs_involved = set()
                        for i in range(pos):
                            old_diffs_involved.add(old_row[pos] - old_row[i])
                        for i in range(pos + 1, len(old_row)):
                            old_diffs_involved.add(old_row[i] - old_row[pos])
                        
                        # Add new differences
                        new_row = old_row[:]
                        new_row[pos] = new_val
                        new_diffs_involved = set()
                        for i in range(pos):
                            new_diffs_involved.add(new_row[pos] - new_row[i])
                        for i in range(pos + 1, len(new_row)):
                            new_diffs_involved.add(new_row[i] - new_row[pos])
                        
                        # Check for conflicts
                        # The new differences must not conflict with:
                        # 1. Other differences in the same row (not involving pos)
                        # 2. Differences in other rows
                        same_row_other_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                if i != pos and j != pos:
                                    same_row_other_diffs.add(new_row[j] - new_row[i])
                        
                        if new_diffs_involved & same_row_other_diffs:
                            continue
                        if new_diffs_involved & (all_diffs - old_diffs_involved):
                            continue
                        
                        # Valid move! Apply it
                        rows[r_idx][pos] = new_val
                        row_diffs[r_idx] = compute_row_diffs(rows[r_idx])
                        all_diffs = set()
                        for rd in row_diffs:
                            all_diffs |= rd
                        
                        new_max = get_max_rows_max(rows)
                        if new_max < current_max:
                            current_max = new_max
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return rows, current_max
    
    def get_max_rows_max(rows):
        return max(max(row) for row in rows)
    
    best_rows = None
    best_score = float('inf')
    
    # Strategy 1: Pure greedy
    rng = random.Random(42)
    rows = greedy_construct(n_rows, k, rng, max_val_limit=200)
    if rows is not None and is_valid(rows):
        s = get_max_entry(rows)
        if s < best_score:
            best_score = s
            best_rows = [r[:] for r in rows]
    
    # Strategy 2: Multiple randomized greedy attempts
    n_attempts = 0
    while time.time() < deadline and n_attempts < 50:
        seed = n_attempts * 7919 + 13
        rng = random.Random(seed)
        perturb = 0.02 + 0.08 * (n_attempts % 10) / 10.0
        rows = greedy_construct(n_rows, k, rng, max_val_limit=200, perturb=perturb)
        if rows is not None and is_valid(rows):
            s = get_max_entry(rows)
            if s < best_score:
                best_score = s
                best_rows = [r[:] for r in rows]
        n_attempts += 1
    
    # Strategy 3: Local search on best solution
    if best_rows is not None and time.time() < deadline:
        rng = random.Random(12345)
        improved_rows, improved_score = local_search(best_rows, n_rows, k, rng, max_iters=100)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = [r[:] for r in improved_rows]
    
    # Strategy 4: More randomized greedy with different parameters
    n_attempts = 0
    while time.time() < deadline and n_attempts < 100:
        seed = 100000 + n_attempts * 31
        rng = random.Random(seed)
        perturb = 0.01 + 0.15 * (n_attempts % 20) / 20.0
        rows = greedy_construct(n_rows, k, rng, max_val_limit=150, perturb=perturb)
        if rows is not None and is_valid(rows):
            s = get_max_entry(rows)
            if s < best_score:
                best_score = s
                best_rows = [r[:] for r in rows]
        n_attempts += 1
    
    # Strategy 5: Simulated annealing-like approach
    if best_rows is not None and time.time() < deadline:
        rng = random.Random(99999)
        rows = [r[:] for r in best_rows]
        current_score = best_score
        T = 10.0
        for step in range(500):
            if time.time() > deadline:
                break
            T *= 0.995
            if T < 0.1:
                break
            
            # Random move: pick a row and position, try a random nearby value
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)
            old_val = rows[r_idx][pos]
            
            # Try a few random values
            for _ in range(5):
                delta = rng.randint(-5, 5)
                new_val = old_val + delta
                if new_val <= rows[r_idx][pos - 1] or (pos < k and new_val >= rows[r_idx][pos + 1]):
                    continue
                
                # Check validity
                new_row = rows[r_idx][:]
                new_row[pos] = new_val
                new_diffs = compute_row_diffs(new_row)
                
                # Check against all other differences
                other_diffs = set()
                for i in range(n_rows):
                    if i != r_idx:
                        other_diffs |= compute_row_diffs(rows[i])
                    else:
                        # Same row, other differences
                        for a in range(len(new_row)):
                            for b in range(a + 1, len(new_row)):
                                if a != pos and b != pos:
                                    other_diffs.add(new_row[b] - new_row[a])
                
                if not (new_diffs & other_diffs):
                    new_score = get_max_entry(rows)
                    # Actually need to compute new max
                    rows[r_idx] = new_row
                    new_max = get_max_entry(rows)
                    
                    if new_max <= current_score or rng.random() < np.exp(-(new_max - current_score) / T):
                        current_score = new_max
                        if new_max < best_score:
                            best_score = new_max
                            best_rows = [r[:] for r in rows]
                    else:
                        rows[r_idx][pos] = old_val
    
    if best_rows is not None:
        return best_rows
    
    # Fallback: simple greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
