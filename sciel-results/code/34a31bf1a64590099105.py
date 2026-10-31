import time
import math
import random
import numpy as np
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Greedy initialization
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
    
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in rows)
    
    # State for Local Search
    current_rows = [r[:] for r in rows]
    
    # Precompute difference sets per row for fast conflict checking
    # row_diffs[r] = set of all positive differences in row r
    row_diffs = []
    for r in current_rows:
        s = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                s.add(r[j] - r[i])
        row_diffs.append(s)
    
    # Track global difference counts
    all_diffs = {}
    for s in row_diffs:
        for d in s:
            all_diffs[d] = all_diffs.get(d, 0) + 1
    
    def num_conflicts():
        """Count of differences that appear more than once."""
        return sum(1 for c in all_diffs.values() if c > 1)
    
    current_conflicts = num_conflicts()
    
    def recalc_row_diffs(r):
        s = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                s.add(r[j] - r[i])
        return s
    
    def apply_move(ri, ei, new_val):
        """Move element ei in row ri to new_val, update all state."""
        nonlocal current_conflicts
        
        row = current_rows[ri]
        old_val = row[ei]
        if old_val == new_val:
            return
        
        old_diffs = row_diffs[ri]
        
        # Remove old differences involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - old_val)
            if d in all_diffs:
                all_diffs[d] -= 1
                if all_diffs[d] == 0:
                    del all_diffs[d]
        
        # Set new value
        row[ei] = new_val
        
        # Add new differences involving ei
        for j in range(len(row)):
            if j == ei:
                continue
            d = abs(row[j] - new_val)
            all_diffs[d] = all_diffs.get(d, 0) + 1
        
        # Recalculate row_diffs for this row
        row_diffs[ri] = recalc_row_diffs(row)
        
        # Recalculate conflicts
        current_conflicts = num_conflicts()
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    def local_search(max_steps=200000):
        """Simulated annealing to first eliminate conflicts, then minimize scope."""
        nonlocal best_scope, best_rows, current_conflicts
        
        steps = 0
        T = 5.0
        alpha = 0.995
        T_min = 0.01
        
        while steps < max_steps and time.time() < deadline:
            ri = random.randint(0, n_rows - 1)
            ei = random.randint(1, k)
            
            row = current_rows[ri]
            old_val = row[ei]
            
            lower = row[ei - 1] + 1
            upper = row[ei + 1] - 1 if ei < k else best_scope + 5
            
            if lower > upper:
                continue
            
            # Sample new value, biased toward smaller values when conflicts are 0
            if current_conflicts == 0:
                # Focus on reducing scope
                if old_val > lower:
                    new_val = random.randint(lower, old_val)
                else:
                    new_val = old_val
            else:
                # Try to fix conflicts, allow some expansion
                new_val = random.randint(lower, min(upper, old_val + 10))
            
            if new_val == old_val:
                continue
            
            # Calculate delta conflicts
            old_conflicts = current_conflicts
            
            # Simulate move
            row[ei] = new_val
            new_diffs = recalc_row_diffs(row)
            
            # Temporarily update all_diffs
            for j in range(len(row)):
                if j == ei:
                    continue
                d = abs(row[j] - new_val)
                all_diffs[d] = all_diffs.get(d, 0) + 1
            
            # Remove old diffs
            for j in range(len(row)):
                if j == ei:
                    continue
                d = abs(row[j] - old_val)
                if d in all_diffs:
                    all_diffs[d] -= 1
                    if all_diffs[d] == 0:
                        del all_diffs[d]
            
            new_conflicts = num_conflicts()
            delta_c = new_conflicts - old_conflicts
            
            # Undo simulation
            for j in range(len(row)):
                if j == ei:
                    continue
                d = abs(row[j] - new_val)
                if d in all_diffs:
                    all_diffs[d] -= 1
                    if all_diffs[d] == 0:
                        del all_diffs[d]
            
            for j in range(len(row)):
                if j == ei:
                    continue
                d = abs(row[j] - old_val)
                all_diffs[d] = all_diffs.get(d, 0) + 1
            
            row[ei] = old_val
            
            # Acceptance criterion
            accepted = False
            if current_conflicts > 0:
                if delta_c < 0:
                    accepted = True
                elif delta_c == 0:
                    if random.random() < 0.1:
                        accepted = True
            else:
                # Conflicts are 0, minimize scope
                if new_val < old_val:
                    accepted = True
                elif random.random() < math.exp(-5.0 / max(T, T_min)):
                    accepted = True
            
            if accepted:
                apply_move(ri, ei, new_val)
                
                if current_conflicts == 0:
                    cur_scope = get_scope()
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [r[:] for r in current_rows]
            
            T *= alpha
            if T < T_min:
                T = T_min
                # Restart with higher temperature occasionally
                if random.random() < 0.01:
                    T = 2.0
                    current_conflicts = num_conflicts()
            
            steps += 1
    
    # Multi-start: run local search from different initializations
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            # Use the greedy initialization
            pass
        else:
            # Perturb current best
            current_rows = [r[:] for r in best_rows]
            row_diffs = [recalc_row_diffs(r) for r in current_rows]
            all_diffs = {}
            for s in row_diffs:
                for d in s:
                    all_diffs[d] = all_diffs.get(d, 0) + 1
            current_conflicts = num_conflicts()
            
            # Random kicks
            for _ in range(random.randint(3, 8)):
                ri = random.randint(0, n_rows - 1)
                ei = random.randint(1, k)
                row = current_rows[ri]
                lower = row[ei - 1] + 1
                upper = row[ei + 1] - 1 if ei < k else best_scope + 10
                if lower <= upper:
                    new_val = random.randint(lower, upper)
                    if new_val != row[ei]:
                        apply_move(ri, ei, new_val)
        
        # Run local search
        local_search(max_steps=100000)
        
        # If we've hit the counting bound, stop early
        if best_scope <= 112:
            break
    
    return best_rows
