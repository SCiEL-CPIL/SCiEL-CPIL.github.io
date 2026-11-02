"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    def greedy_row(used, k, randomize=False, num_candidates=1):
        """Build one row of k+1 increasing integers starting at 0."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if randomize and num_candidates > 1:
                # Try a few candidates and pick randomly among valid ones
                candidates = []
                for _ in range(num_candidates):
                    c = x
                    new_diffs = {c - a for a in row}
                    if not new_diffs & used:
                        candidates.append(c)
                    x += 1
                if candidates:
                    x = random.choice(candidates)
                    row.append(x)
                    used.update({x - a for a in row[:-1]})
                else:
                    # Fallback: keep incrementing
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used.update({x - a for a in row[:-1]})
                            break
            else:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used.update({x - a for a in row[:-1]})
        return row
    
    def compute_scope(rows):
        """Return the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    def build_initial(n_rows, k, seed=None):
        """Build initial solution using greedy."""
        rng = random.Random(seed)
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
                    used.update(new_diffs)
            rows.append(row)
        return rows, used
    
    # Build initial greedy solution
    best_rows, best_used = build_initial(n_rows, k, seed=42)
    best_scope = compute_scope(best_rows)
    
    # Simulated annealing
    rng = random.Random(12345)
    T = 1.0  # temperature
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Pick a random row to replace
        idx = rng.randrange(n_rows)
        old_row = best_rows[idx]
        old_scope = best_scope
        
        # Reconstruct this row with some randomization
        # Remove old row's differences from used
        old_diffs = set()
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                old_diffs.add(old_row[j] - old_row[i])
        new_used = best_used - old_diffs
        
        # Try to build a new row with randomization
        new_row = [0]
        x = 0
        success = False
        max_x = old_scope + 20  # allow some expansion
        while len(new_row) < k + 1 and x < max_x:
            x += 1
            # With some probability, try a few candidates
            if rng.random() < 0.3:
                # Try up to 5 candidates
                found = False
                for _ in range(5):
                    c = x
                    new_diffs = {c - a for a in new_row}
                    if not new_diffs & new_used:
                        new_row.append(c)
                        new_used.update(new_diffs)
                        x = c
                        found = True
                        break
                    x += 1
                if not found:
                    success = False
                    break
            else:
                new_diffs = {x - a for a in new_row}
                if not new_diffs & new_used:
                    new_row.append(x)
                    new_used.update(new_diffs)
        
        if not success or len(new_row) != k + 1:
            T *= cooling
            continue
        
        # Check if new row's differences don't conflict
        new_diffs = set()
        for i in range(len(new_row)):
            for j in range(i+1, len(new_row)):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Verify no conflicts with other rows
        if new_diffs & (best_used - old_diffs):
            T *= cooling
            continue
        
        new_rows = best_rows.copy()
        new_rows[idx] = new_row
        new_scope = compute_scope(new_rows)
        
        # Accept if better, or with probability based on temperature
        if new_scope <= old_scope:
            best_rows = new_rows
            best_used = new_used
            best_scope = new_scope
        else:
            delta = new_scope - old_scope
            if rng.random() < (1.0 / (1.0 + delta)) * T:
                best_rows = new_rows
                best_used = new_used
                best_scope = new_scope
        
        T *= cooling
    
    return best_rows
# EVOLVE-BLOCK-END
