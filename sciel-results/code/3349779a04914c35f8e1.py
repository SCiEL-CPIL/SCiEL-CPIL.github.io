"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # First, get a baseline from simple greedy
    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(row) for row in best_rows)

    # Strategy: combine global greedy construction with local search refinement.
    # 1. Use stochastic global greedy to get diverse initial solutions.
    # 2. Apply local search (element shifting, row replacement) to refine.
    
    def find_next_valid(row, used, start_x):
        """Find the smallest x >= start_x such that adding x to row doesn't conflict."""
        x = start_x
        while x < 2000:
            valid = True
            diffs = []
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                diffs.append(d)
            if valid:
                return x, diffs
            x += 1
        return None, None

    def stochastic_global_greedy(random_seed):
        """Global greedy with random tie-breaking."""
        random.seed(random_seed)
        used = set()
        rows = [[] for _ in range(n_rows)]
        for i in range(n_rows):
            rows[i].append(0)
        
        while True:
            min_x = float('inf')
            best_row = -1
            tied_rows = []
            
            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                x, diffs = find_next_valid(rows[i], used, last + 1)
                if x is None:
                    continue
                if x < min_x:
                    min_x = x
                    best_row = i
                    tied_rows = [i]
                elif x == min_x:
                    tied_rows.append(i)
            
            if best_row == -1:
                break
            
            if len(tied_rows) > 1:
                chosen_row = random.choice(tied_rows)
            else:
                chosen_row = best_row
            
            last = rows[chosen_row][-1]
            x, diffs = find_next_valid(rows[chosen_row], used, last + 1)
            rows[chosen_row].append(x)
            for d in diffs:
                used.add(d)
            
            if time.time() > deadline:
                break
        
        if all(len(r) == k + 1 for r in rows):
            return rows
        return None

    def local_search(rows, max_iterations=5000):
        """Local search to reduce the maximum entry."""
        rows = [list(r) for r in rows]
        current_max = max(max(r) for r in rows)
        best_max_local = current_max
        best_rows_local = [list(r) for r in rows]
        
        def get_used_set():
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            return used
        
        def is_valid_row(row, used_except_row):
            """Check if row is valid given differences used by other rows."""
            row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_except_row:
                        return False
                    if d in row_diffs:
                        return False
                    row_diffs.add(d)
            return True
        
        # Precompute used sets per row
        all_used = get_used_set()
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to reduce the largest element in each row
            for i in range(n_rows):
                if len(rows[i]) < k + 1:
                    continue
                
                # Find the largest element
                last_idx = len(rows[i]) - 1
                old_val = rows[i][last_idx]
                
                if old_val <= 0:
                    continue
                
                # Try to replace with a smaller value
                # We need to try values from old_val - 1 down to some minimum
                # But we need to maintain strictly increasing order
                min_possible = rows[i][last_idx - 1] + 1 if last_idx > 0 else 1
                
                # Compute used set without this row's differences
                used_other = set()
                for j in range(n_rows):
                    if j == i:
                        continue
                    row = rows[j]
                    for a in range(len(row)):
                        for b in range(a + 1, len(row)):
                            used_other.add(row[b] - row[a])
                
                # Also include differences from this row except those involving last element
                for a in range(last_idx):
                    for b in range(a + 1, last_idx):
                        used_other.add(rows[i][b] - rows[i][a])
                
                # Try to find a smaller value for the last element
                found = False
                for new_val in range(old_val - 1, min_possible - 1, -1):
                    if new_val <= rows[i][last_idx - 1]:
                        break
                    # Check if new_val is valid
                    valid = True
                    for a in range(last_idx):
                        d = new_val - rows[i][a]
                        if d in used_other:
                            valid = False
                            break
                    if valid:
                        # Accept the move
                        rows[i][last_idx] = new_val
                        current_max = max(max(r) for r in rows)
                        if current_max < best_max_local:
                            best_max_local = current_max
                            best_rows_local = [list(r) for r in rows]
                        found = True
                        break
                # If we found a smaller value, continue; otherwise try other rows
        
        return best_rows_local, best_max_local

    # Run stochastic global greedy multiple times
    num_restarts = 30
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        rows = stochastic_global_greedy(seed)
        if rows:
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows
        
        # Apply local search to improve
        if time.time() < deadline - 5:
            improved_rows, improved_max = local_search(rows, max_iterations=2000)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows

    # Final local search on the best solution
    if time.time() < deadline - 2:
        final_rows, final_max = local_search(best_rows, max_iterations=10000)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows

    return best_rows

def _greedy_construct(n_rows, k):
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
