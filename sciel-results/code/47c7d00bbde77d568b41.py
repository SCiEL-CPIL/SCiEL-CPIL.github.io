"""Difference triangle set (4,7) - simulated annealing with element-level moves."""
import random
import time

def _all_diffs_and_valid(rows):
    """Compute all differences and check validity. Returns (all_diffs_set, is_valid)."""
    all_diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return all_diffs, False
                all_diffs.add(d)
    return all_diffs, True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid_row_structure(row):
    """Check row is increasing and starts at 0."""
    if not row or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i-1]:
            return False
    return True

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    timeout = 115
    
    def _greedy_initial():
        """Generate initial solution using greedy."""
        used_diffs = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used_diffs):
                    row.append(x)
                    used_diffs |= new_diffs
                x += 1
            rows.append(row)
        return rows
    
    best_rows = None
    best_score = float('inf')
    
    # Multiple restarts with simulated annealing
    for restart in range(15):
        if time.time() - start_time > timeout:
            break
        
        # Generate initial solution
        current_rows = _greedy_initial()
        current_score = _max_entry(current_rows)
        
        # Simulated annealing
        T = 30.0
        cooling = 0.9995
        min_T = 0.1
        max_iters = 30000
        
        for iteration in range(max_iters):
            if time.time() - start_time > timeout:
                break
            if T < min_T:
                break
            
            # Pick a random row and element to perturb
            row_idx = random.randint(0, n_rows - 1)
            elem_idx = random.randint(1, k)
            
            old_val = current_rows[row_idx][elem_idx]
            
            # Generate new value
            step = random.randint(-15, 15)
            if step == 0:
                continue
            new_val = old_val + step
            
            # Check row structure validity
            if new_val <= current_rows[row_idx][elem_idx - 1]:
                continue
            if elem_idx < k and new_val >= current_rows[row_idx][elem_idx + 1]:
                continue
            
            # Apply move
            current_rows[row_idx][elem_idx] = new_val
            
            # Check full validity
            _, valid = _all_diffs_and_valid(current_rows)
            
            if valid:
                new_score = _max_entry(current_rows)
                delta = new_score - current_score
                
                # Accept if better or with Boltzmann probability
                if delta < 0 or random.random() < (2.718281828459045 ** (-delta / T)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_rows = [row[:] for row in current_rows]
                else:
                    current_rows[row_idx][elem_idx] = old_val
            else:
                # Reject invalid
                current_rows[row_idx][elem_idx] = old_val
            
            T *= cooling
        
        if best_score == float('inf'):
            best_score = current_score
            best_rows = [row[:] for row in current_rows]
    
    if best_rows is None:
        best_rows = _greedy_initial()
    
    return best_rows
