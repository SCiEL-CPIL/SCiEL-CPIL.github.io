"""Difference triangle set (4,7): construct 4 rows of 8 increasing integers starting at 0
with all pairwise differences distinct across all rows, minimizing the largest entry."""
import time
import random
import itertools

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin under 120s wall-clock
    
    # For this task: n_rows=4, k=7, so each row has 8 entries.
    # Total differences per row: C(8,2) = 28. Total across 4 rows: 112 differences.
    # The differences must all be distinct positive integers.
    # The largest entry M must be such that we can place 112 distinct differences
    # within [1, M] (since max difference in a row is at most M).
    # Counting bound: 112 distinct values in [1, M] requires M >= 112.
    
    # Strategy: use a randomized local search / simulated annealing approach.
    # Represent each row as a list of k+1 values starting at 0.
    # Objective: minimize the maximum entry across all rows.
    # Constraint: all C(k+1, 2) differences per row, summed over all rows, must be distinct.
    
    # First, generate a valid initial solution using greedy + repair.
    # Then iteratively try to reduce the max entry.
    
    def compute_differences(rows):
        """Return set of all differences and check validity."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return None, d, i, j, row
                    diffs.add(d)
        return diffs, None, None, None, None
    
    def is_valid(rows):
        diffs, dup, *_ = compute_differences(rows)
        return dup is None
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Greedy construction: build rows one by one, picking smallest valid next element
    def greedy_build(n_rows, k, seed=0):
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            # Shuffle the order in which we try to extend? No, greedy picks smallest.
            # But we can randomize the starting pattern slightly.
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Local search: try to move elements to reduce max entry
    def local_search(rows, n_rows, k, time_limit):
        """Iterative local search to reduce the maximum entry."""
        rng = random.Random()
        best_rows = [list(r) for r in rows]
        best_max = max_entry(best_rows)
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Try random perturbations
            rows_copy = [list(r) for r in best_rows]
            improved = False
            
            # Try moving one element in a random row
            for _ in range(50):
                r = rng.randint(0, n_rows - 1)
                # Pick a random position (not 0, which is fixed)
                pos = rng.randint(1, k)
                old_val = rows_copy[r][pos]
                
                # Try to move it to a smaller value
                # Find a new value that keeps the row strictly increasing and valid
                lo = rows_copy[r][pos - 1] + 1
                hi = old_val - 1 if pos < k else old_val
                
                if lo > hi:
                    continue
                    
                # Try values from old_val-1 down to lo
                found = False
                for new_val in range(old_val - 1, lo - 1, -1):
                    # Check if row remains strictly increasing
                    if pos < k and new_val >= rows_copy[r][pos + 1]:
                        continue
                    if pos > 1 and new_val <= rows_copy[r][pos - 1]:
                        continue
                    
                    # Try the move
                    rows_copy[r][pos] = new_val
                    if is_valid(rows_copy):
                        cur_max = max_entry(rows_copy)
                        if cur_max < best_max:
                            best_max = cur_max
                            best_rows = [list(rr) for rr in rows_copy]
                            improved = True
                        found = True
                        break
                    rows_copy[r][pos] = old_val
                
                if improved:
                    break
            
            # If no improvement, try a more aggressive perturbation
            if not improved:
                # Try rebuilding one row entirely with a different random strategy
                r = rng.randint(0, n_rows - 1)
                other_rows = [best_rows[i] for i in range(n_rows) if i != r]
                
                # Collect used differences from other rows
                used = set()
                for row in other_rows:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            used.add(row[j] - row[i])
                
                # Try to build row r with a smaller max
                for attempt in range(20):
                    row = [0]
                    x = 0
                    # Add some randomness
                    skip = 0
                    while len(row) < k + 1:
                        x += 1
                        if skip > 0:
                            skip -= 1
                            continue
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            # Occasionally skip ahead
                            if rng.random() < 0.3:
                                skip = rng.randint(1, 3)
                    
                    if max(row) < best_max:
                        new_rows = [best_rows[i] for i in range(n_rows)]
                        new_rows[r] = row
                        if is_valid(new_rows):
                            best_max = max_entry(new_rows)
                            best_rows = new_rows
                            improved = True
                            break
            
            # Check deadline
            if time.time() > deadline - 5:
                break
        
        return best_rows, best_max
    
    # Phase 1: Generate multiple initial solutions with different seeds
    initial_rows = greedy_build(n_rows, k, seed=42)
    best_rows = initial_rows
    best_max = max_entry(best_rows)
    
    # Phase 2: Local search to improve
    remaining_time = deadline - time.time() - 5
    if remaining_time > 1:
        best_rows, best_max = local_search(best_rows, n_rows, k, remaining_time)
    
    # Phase 3: Try to further reduce by optimizing individual rows
    # Sort rows by their max entry, try to reduce the largest row
    if time.time() < deadline - 10:
        for _ in range(3):
            # Find the row with the largest max entry
            row_indices = sorted(range(n_rows), key=lambda i: max(best_rows[i]), reverse=True)
            
            for r in row_indices:
                if time.time() > deadline - 5:
                    break
                
                other_rows = [best_rows[i] for i in range(n_rows) if i != r]
                used = set()
                for row in other_rows:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            used.add(row[j] - row[i])
                
                # Try to rebuild row r with a tighter bound
                current_max = max(best_rows[r])
                target_max = current_max - 1
                
                # Use backtracking with pruning to find a row with max <= target_max
                def backtrack(row, used_diffs, target):
                    if len(row) == k + 1:
                        return row[:]
                    
                    last = row[-1]
                    # Try next values
                    for x in range(last + 1, target + 1):
                        new_diffs = {x - a for a in row}
                        if new_diffs & used_diffs:
                            continue
                        row.append(x)
                        result = backtrack(row, used_diffs | new_diffs, target)
                        if result is not None:
                            return result
                        row.pop()
                    return None
                
                new_row = backtrack([0], set(), target_max)
                if new_row is not None:
                    new_rows = [best_rows[i] for i in range(n_rows)]
                    new_rows[r] = new_row
                    if is_valid(new_rows):
                        best_rows = new_rows
                        best_max = max_entry(best_rows)
                        if time.time() > deadline - 5:
                            break
    
    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(4, 7)
    print("Rows:", rows)
    print("Max entry:", max(max(r) for r in rows))
    # Verify validity
    diffs = set()
    valid = True
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    print(f"Duplicate difference {d}")
                diffs.add(d)
    print("Valid:", valid)
    print("Num differences:", len(diffs))
