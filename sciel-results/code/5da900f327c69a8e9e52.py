"""
Combined solution: Element-wise Simulated Annealing with Incremental Conflict Tracking.
Strategy:
1. Greedy initialization to get a valid (or near-valid) starting point.
2. Use a global Counter to track difference frequencies across all rows.
3. Perform element-wise moves (change value of row[r][j]).
4. Two types of moves:
   - 'Shrink': Try to reduce a value by 1. Accept if valid (conflicts==0). This directly minimizes scope.
   - 'Perturb': Randomly change a value within a range. Accept via SA probability based on energy (conflicts + scope).
5. Efficiently update the difference counter by only removing/adding differences involving the changed element.
"""
import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: return by 145s
    deadline = start_time + 145.0
    
    # --- Helper Functions ---
    
    def get_row_diffs(row):
        """Compute all positive differences for a row."""
        diffs = []
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs

    def calculate_scope(rows):
        """Calculate the maximum entry in the entire set of rows."""
        return max(max(row) for row in rows)

    def count_conflicts(counter):
        """Count total conflicts (duplicates) in the difference frequency counter."""
        return sum(c - 1 for c in counter.values() if c > 1)

    # --- Initialization ---
    
    # Greedy construction:
    # Pick smallest possible next integer that doesn't create a difference conflict.
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            # We need k+1 elements, so k more to pick
            while len(row) < k + 1:
                candidate = current_max + 1
                while True:
                    new_diffs = {candidate - x for x in row}
                    if not (new_diffs & used_diffs):
                        row.append(candidate)
                        used_diffs |= new_diffs
                        current_max = candidate
                        break
                    candidate += 1
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_scope = calculate_scope(best_rows)
    
    # Initialize state
    current_rows = [list(r) for r in best_rows]
    
    # Build initial difference counter
    diff_counter = Counter()
    for row in current_rows:
        for d in get_row_diffs(row):
            diff_counter[d] += 1
            
    current_conflicts = count_conflicts(diff_counter)
    current_scope = calculate_scope(current_rows)
    
    # If greedy already found 0 conflicts, we just need to minimize scope.
    # If not, we need to resolve conflicts first.
    
    # --- Main Loop: Simulated Annealing ---
    
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    step = 0
    no_improve_steps = 0
    max_no_improve = 5000 # Reset T if stuck
    
    while time.time() < deadline:
        step += 1
        
        # Determine move type
        # Bias towards shrinking if we are valid (conflicts == 0)
        # Bias towards perturbation if we have conflicts
        if current_conflicts == 0:
            # 80% chance to try shrinking, 20% to perturb (to escape local minima in scope)
            move_type = 'shrink' if random.random() < 0.8 else 'perturb'
        else:
            # If conflicts exist, we must perturb to resolve them
            move_type = 'perturb'

        # Select row and index
        r = random.randint(0, n_rows - 1)
        row = current_rows[r]
        
        # We can't change index 0 (it's always 0)
        j = random.randint(1, k)
        
        old_val = row[j]
        
        if move_type == 'shrink':
            new_val = old_val - 1
            # Check bounds: must be strictly increasing
            if new_val <= row[j-1]:
                continue
        else:
            # Perturb: choose a new value in a range around old_val
            # Range: [row[j-1] + 1, min(row[j+1] - 1, old_val + 20)]
            min_val = row[j-1] + 1
            max_val = old_val + 20
            if j < k:
                max_val = min(max_val, row[j+1] - 1)
            
            if min_val > max_val:
                continue
                
            new_val = random.randint(min_val, max_val)
            if new_val == old_val:
                continue

        # --- Apply Move and Update State ---
        
        # 1. Remove old differences involving (r, j)
        # Differences are: row[j] - row[i] for i < j
        #                  row[i] - row[j] for i > j
        removed_diffs = []
        
        # Left side: row[j] - row[i]
        for i in range(j):
            d = old_val - row[i]
            removed_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        # Right side: row[i] - row[j]
        for i in range(j + 1, len(row)):
            d = row[i] - old_val
            removed_diffs.append(d)
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        # 2. Set new value
        row[j] = new_val
        
        # 3. Add new differences
        # Left side: new_val - row[i]
        for i in range(j):
            d = new_val - row[i]
            diff_counter[d] += 1
            
        # Right side: row[i] - new_val
        for i in range(j + 1, len(row)):
            d = row[i] - new_val
            diff_counter[d] += 1
            
        # 4. Evaluate new state
        new_conflicts = count_conflicts(diff_counter)
        new_scope = calculate_scope(current_rows)
        
        delta_conflicts = new_conflicts - current_conflicts
        delta_scope = new_scope - current_scope
        
        # Energy function: We prioritize zero conflicts, then minimize scope.
        # Weight conflicts heavily.
        energy_weight = 100
        delta_energy = energy_weight * delta_conflicts + delta_scope
        
        accept = False
        
        if delta_energy < 0:
            accept = True
        else:
            # SA acceptance
            if T > min_T:
                # Avoid overflow in exp
                if delta_energy / T > 50:
                    prob = 0.0
                else:
                    prob = math.exp(-delta_energy / T)
                accept = random.random() < prob
        
        if accept:
            # Commit move
            current_conflicts = new_conflicts
            current_scope = new_scope
            
            # Update best if valid
            if current_conflicts == 0:
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
                    no_improve_steps = 0
                    # Reheat slightly to explore more?
                    # T = max(T, 10.0) 
            else:
                no_improve_steps += 1
        else:
            # Revert move
            row[j] = old_val
            
            # Re-add old differences
            for d in removed_diffs:
                diff_counter[d] += 1
                
            # Remove new differences
            for i in range(j):
                d = new_val - row[i]
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
                    
            for i in range(j + 1, len(row)):
                d = row[i] - new_val
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            no_improve_steps += 1
            
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        # If stuck for a long time, restart temperature
        if no_improve_steps > max_no_improve:
            T = 50.0
            no_improve_steps = 0

    # Final check: ensure we return a valid set.
    # If best_rows has conflicts, we might need to return the current state if it's better?
    # The task says "Valid when...". If invalid, score 0.
    # We must return a valid set.
    # best_rows is only updated when conflicts == 0.
    # If we never found a valid set (unlikely with greedy start), we should try to fix it or return best effort.
    # However, greedy usually starts valid or near valid.
    
    # Safety: If best_rows is not valid (shouldn't happen), try to use current if valid?
    # Actually, best_rows is initialized with greedy. If greedy is valid, best_rows is valid.
    # If greedy was invalid (conflicts > 0), best_rows might still be the greedy one (invalid).
    # We need to ensure we return a VALID set.
    
    # Check validity of best_rows
    final_counter = Counter()
    for row in best_rows:
        for d in get_row_diffs(row):
            final_counter[d] += 1
    if count_conflicts(final_counter) > 0:
        # If best is invalid, check current
        if count_conflicts(diff_counter) == 0:
            return current_rows
        else:
            # Return the one with least conflicts? Or just best_rows?
            # The prompt says "Valid when...". If invalid, score 0.
            # We should try to return the most valid one.
            # But typically greedy is valid for small n_rows, k.
            pass
            
    return best_rows

# EVOLVE-BLOCK-END
