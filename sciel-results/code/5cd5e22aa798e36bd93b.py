import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    def is_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows

    # Initial solution
    best_solution = greedy_solve()
    best_M = max(max(row) for row in best_solution)
    best_diffs = [get_row_diffs(row) for row in best_solution]
    
    # Simulated Annealing
    # State: rows, row_diffs, used_diffs
    # Cost: max(max(row) for row in rows) + penalty for invalidity? 
    # Actually, we want to maintain validity. 
    # If we enforce validity in moves, it's a random walk in the valid space.
    # If we allow invalid states, we need a penalty.
    # Given the tight constraints, maintaining validity is safer but might get stuck.
    # Let's try to maintain validity by only accepting valid moves.
    # Moves:
    # 1. Replace a row with a random valid row (diffs disjoint from others).
    # 2. Shrink an entry in a row (reduce value if possible while maintaining validity and strict increase).
    
    # To generate a random valid row given other rows:
    # We need a row of size k+1 starting with 0, such that its diffs are disjoint from the union of diffs of other rows.
    
    def generate_random_row(other_used_diffs, max_val):
        # Backtracking to generate a valid row
        # To speed up, we can try to pick random candidates
        for _ in range(100): # Limit attempts per generation
            row = [0]
            current_diffs = set()
            success = True
            for _ in range(k):
                # Pick random x
                # Heuristic: pick from a range
                # We need x > row[-1]
                # And x - y not in other_used_diffs or current_diffs
                found = False
                # Try a few random spots
                # Range: row[-1] + 1 to max_val
                # To keep M small, we might bias towards smaller x?
                # Or just uniform?
                # Let's try uniform in [row[-1]+1, max_val]
                # But max_val might be large.
                # Let's try to pick from a set of candidates.
                
                candidates = []
                # To avoid iterating up to max_val (which might be 100+), 
                # we can just pick random x and check.
                # But we need to ensure we don't loop forever.
                
                # Heuristic: try x = row[-1] + random.randint(1, 10) first?
                # No, differences can be large.
                
                # Let's just pick random x in [row[-1]+1, max_val]
                # If max_val is small (e.g. 110), this is fine.
                
                for _ in range(20):
                    x = random.randint(row[-1] + 1, max_val)
                    new_diffs = {x - y for y in row}
                    if new_diffs & other_used_diffs or new_diffs & current_diffs:
                        continue
                    # Check if we can complete the rest? 
                    # Hard to know without lookahead.
                    # Just accept and hope backtracking or retry works.
                    row.append(x)
                    current_diffs.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    success = False
                    break
            
            if success:
                # Final check
                if current_diffs & other_used_diffs:
                    return None
                return row
        return None

    # SA Loop
    current_solution = [row[:] for row in best_solution]
    current_diffs = [get_row_diffs(row) for row in current_solution]
    current_M = best_M
    
    T = 50.0 # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    while T > min_T and time.time() < deadline - 2.0:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Shrink move
            # Pick a random row and a random entry (not 0)
            r_idx = random.randint(0, n_rows - 1)
            # Entries are at indices 1..k
            e_idx = random.randint(1, k)
            old_val = current_solution[r_idx][e_idx]
            
            # Try to reduce old_val
            # New val must be > current_solution[r_idx][e_idx-1]
            # And < current_solution[r_idx][e_idx+1] (if exists)
            min_val = current_solution[r_idx][e_idx-1] + 1
            max_val = (current_solution[r_idx][e_idx+1] - 1) if e_idx < k else old_val - 1
            
            if min_val < max_val:
                # Try random values in [min_val, max_val-1]
                # Or just try min_val?
                # Let's try a few random values
                candidates = list(range(min_val, max_val))
                random.shuffle(candidates)
                found_new_val = None
                for new_val in candidates:
                    # Check if changing entry to new_val keeps row valid and diffs disjoint
                    # We need to recalculate diffs for this row
                    new_row = current_solution[r_idx][:]
                    new_row[e_idx] = new_val
                    # Check strict increase
                    if not all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                        continue
                    
                    new_diffs = get_row_diffs(new_row)
                    # Check disjointness with other rows
                    # other_diffs = union of current_diffs[j] for j != r_idx
                    # We can precompute or just check
                    conflict = False
                    for j in range(n_rows):
                        if j == r_idx:
                            continue
                        if new_diffs & current_diffs[j]:
                            conflict = True
                            break
                    
                    if not conflict:
                        found_new_val = new_val
                        break
                
                if found_new_val is not None:
                    # Accept move? 
                    # This move reduces M or keeps it same.
                    # It maintains validity.
                    # Always accept? Or probabilistically?
                    # Since it improves or maintains validity and might reduce M, 
                    # and doesn't increase "complexity" significantly, let's accept it.
                    # Actually, it might block future moves if it's a "bad" shrink?
                    # But usually shrinking is good.
                    # Let's accept with probability 1.0 to encourage shrinking.
                    
                    # Update state
                    current_solution[r_idx][e_idx] = found_new_val
                    current_diffs[r_idx] = get_row_diffs(current_solution[r_idx])
                    
                    new_M = max(max(row) for row in current_solution)
                    if new_M < current_M:
                        current_M = new_M
                        if current_M < best_M:
                            best_M = current_M
                            best_solution = [row[:] for row in current_solution]
                            best_diffs = [get_row_diffs(row) for row in best_solution]
        
        else:
            # Row replacement move
            r_idx = random.randint(0, n_rows - 1)
            # Generate a new row that is disjoint from others
            other_diffs = set()
            for j in range(n_rows):
                if j != r_idx:
                    other_diffs.update(current_diffs[j])
            
            # Max val to try?
            # We want to keep M small.
            # Let's try up to current_M
            new_row = generate_random_row(other_diffs, current_M)
            
            if new_row is not None:
                # Check if it improves M
                new_M = max(max(row) if i != r_idx else max(new_row) for i, row in enumerate(current_solution))
                
                # Acceptance criterion
                # If new_M < current_M, accept.
                # If new_M == current_M, accept with prob exp(-delta/T)? 
                # Here delta is 0. So accept with prob 1?
                # This allows random walking in the space of solutions with same M.
                # This is crucial for escaping local minima.
                
                # Let's always accept if valid, to allow exploration.
                # But if it increases M, we reject.
                
                if new_M <= current_M:
                    current_solution[r_idx] = new_row
                    current_diffs[r_idx] = get_row_diffs(new_row)
                    if new_M < current_M:
                        current_M = new_M
                        if current_M < best_M:
                            best_M = current_M
                            best_solution = [row[:] for row in current_solution]
                            best_diffs = [get_row_diffs(row) for row in best_solution]
        
        T *= cooling_rate
        
        # Restart if stuck?
        # If no improvement for a long time, we might restart with a different initial solution?
        # Or just keep going.
        
    # Final check
    if not is_valid(best_solution):
        best_solution = greedy_solve()
        
    return best_solution

# Test locally
if __name__ == "__main__":
    # (7, 5)
    rows = construct(7, 5)
    max_val = max(max(r) for r in rows)
    print(f"Max value: {max_val}")
    # Print rows
    for r in rows:
        print(r)

