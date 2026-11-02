import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry (scope).
    """
    deadline = time.time() + 140  # safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    num_diffs_per_row = k * (k + 1) // 2
    
    def greedy_build(seed_rng, row_order=None, bias_factor=1.0):
        """Build rows greedily with randomization."""
        used = set()
        rows = []
        
        if row_order is None:
            row_order = list(range(n_rows))
        
        for r in row_order:
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # Accept with some probability based on bias
                    # For the first few elements, be more deterministic
                    if bias_factor >= 1.0 or random.random() < bias_factor:
                        row.append(x)
                        used |= new_diffs
                    else:
                        # Skip this candidate, try next
                        pass
                # If we've gone too far, this path is bad
                if x > best_scope + 50:
                    return None, float('inf')
            
            scope = row[-1]
            if scope >= best_scope:
                return None, float('inf')
            rows.append(row)
        
        # Reorder rows to original index order
        final_rows = [None] * n_rows
        for i, r in enumerate(row_order):
            final_rows[r] = rows[i]
        
        max_scope = max(row[-1] for row in final_rows)
        return final_rows, max_scope
    
    # First, try deterministic greedy to get a baseline
    rows_det, scope_det = greedy_build(random.Random(0), bias_factor=1.0)
    if rows_det is not None:
        best_rows = rows_det
        best_scope = scope_det
    
    # Now try many randomized restarts
    rng = random.Random()
    attempts = 0
    
    while time.time() < deadline:
        attempts += 1
        
        # Vary the random seed, row order, and bias
        seed_val = rng.randint(0, 2**31 - 1)
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        # Vary bias: sometimes prefer smaller gaps, sometimes more random
        if rng.random() < 0.5:
            bias_factor = 1.0  # deterministic within the randomized order
        else:
            bias_factor = rng.uniform(0.3, 1.0)
        
        rows, scope = greedy_build(random.Random(seed_val), row_order, bias_factor)
        
        if rows is not None and scope < best_scope:
            best_rows = rows
            best_scope = scope
            # If we hit the counting bound, we're done
            if best_scope <= int((2 * n_rows * num_diffs_per_row) ** 0.5) + 1:
                break
    
    # If we never found anything valid (shouldn't happen), fall back to simple greedy
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
