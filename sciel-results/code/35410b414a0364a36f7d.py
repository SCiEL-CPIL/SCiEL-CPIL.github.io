import time
import random
import sys

def construct(n_rows, k):
    # Deadline guard: must finish well within 120s (evaluator kills at 150s wall-clock).
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Parameters
    # We are searching for a set of n_rows rows, each of length k+1, starting at 0,
    # strictly increasing, such that all positive differences within rows are distinct
    # across all rows. We want to minimize the maximum entry (scope).

    # The best known for (4,7) is 134.
    # Simple counting bound: number of differences per row is k*(k+1)/2.
    # Total differences = n_rows * k*(k+1)/2.
    # These must be distinct and <= scope. So scope >= n_rows * k*(k+1)/2.
    # For (4,7): 4 * 7*8/2 = 4 * 28 = 112.
    # So the theoretical lower bound is 112.

    # Strategy:
    # 1. Try to find a valid configuration with scope as low as possible.
    # 2. Use a combination of:
    #    a. Randomized greedy construction to get initial candidates.
    #    b. Local search to reduce the scope.
    #    c. Conflict-directed value removal (CDVR) style moves.

    # Helper functions
    def get_all_diffs(rows):
        """Get all differences from all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in diffs:
                        return False
                    diffs.add(d)
        return True

    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(row) for row in rows)

    def greedy_construct(n_rows, k, max_scope, seed=None):
        """Greedy construction with a given max_scope constraint."""
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Try values from row[-1]+1 up to max_scope
                found = False
                for x in range(row[-1] + 1, max_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    return None  # Failed to complete
            rows.append(row)
        return rows

    def local_search(rows, n_rows, k, max_iterations=10000, seed=None):
        """Local search to reduce the scope of the rows."""
        if seed is not None:
            random.seed(seed)
        best_rows = [row[:] for row in rows]
        best_scope = get_scope(rows)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            # Try to find a conflict or a way to reduce scope
            # Strategy: pick a random row and position, try to decrease its value
            # while maintaining validity and strict increase.
            
            # First, get all differences
            all_diffs = get_all_diffs(rows)
            
            # Try to reduce the maximum element
            max_val = best_scope
            # Find which row has the max element
            max_row_idx = -1
            max_pos_idx = -1
            for r_idx, row in enumerate(rows):
                if row[-1] == max_val:
                    max_row_idx = r_idx
                    max_pos_idx = len(row) - 1
                    break
            
            if max_row_idx == -1:
                break  # No element at max_scope
            
            # Try to reduce the last element of the row with max value
            row = rows[max_row_idx]
            # The last element can be reduced to any value > row[-2] that doesn't create conflicts
            # But we need to check all differences involving this position
            
            # Try values from row[-2] + 1 up to row[-1] - 1
            found_better = False
            for new_val in range(row[-2] + 1, row[-1]):
                # Check if new_val creates any conflicts
                # Remove old differences involving position -1
                old_diffs = set()
                for i in range(len(row) - 1):
                    old_diffs.add(row[-1] - row[i])
                
                # Add new differences
                new_diffs = set()
                for i in range(len(row) - 1):
                    new_diffs.add(new_val - row[i])
                
                # Check if any new_diffs conflict with all_diffs minus old_diffs
                remaining_diffs = all_diffs - old_diffs
                if not new_diffs & remaining_diffs:
                    # Valid move
                    row[-1] = new_val
                    best_rows[max_row_idx] = row[:]
                    best_scope = max(best_scope - 1, new_val)
                    found_better = True
                    break
            
            if not found_better:
                # Try random moves: pick a random row and a random position (not 0),
                # try to change its value
                r_idx = random.randint(0, n_rows - 1)
                p_idx = random.randint(1, k)  # Position 1 to k
                row = rows[r_idx]
                old_val = row[p_idx]
                
                # Determine range for new value
                min_val = row[p_idx - 1] + 1
                max_val = row[p_idx + 1] - 1 if p_idx < k else best_scope
                
                if min_val > max_val:
                    continue
                
                # Try a random new value in range
                new_val = random.randint(min_val, max_val)
                if new_val == old_val:
                    continue
                
                # Check validity
                # Remove old differences involving position p_idx
                old_diffs = set()
                for i in range(k + 1):
                    if i != p_idx:
                        if i < p_idx:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                
                # Add new differences
                new_diffs = set()
                for i in range(k + 1):
                    if i != p_idx:
                        if i < p_idx:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                
                # Check conflicts
                remaining_diffs = all_diffs - old_diffs
                if not new_diffs & remaining_diffs:
                    # Also check that the row remains strictly increasing
                    if (p_idx == 0 or new_val > row[p_idx - 1]) and \
                       (p_idx == k or new_val < row[p_idx + 1]):
                        row[p_idx] = new_val
                        # Update best if scope reduced
                        current_scope = get_scope(rows)
                        if current_scope < best_scope:
                            best_rows[r_idx] = row[:]
                            best_scope = current_scope
        
        return best_rows, best_scope

    # Main search
    best_rows = None
    best_scope = float('inf')
    
    # First, try greedy with increasing scope
    # Start from the theoretical lower bound
    lower_bound = n_rows * k * (k + 1) // 2
    
    # Try scopes from lower_bound upward
    for scope in range(lower_bound, lower_bound + 50):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds for greedy
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            rows = greedy_construct(n_rows, k, scope, seed)
            if rows is not None:
                if is_valid(rows):
                    current_scope = get_scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [row[:] for row in rows]
                        # Try local search to improve
                        improved_rows, improved_scope = local_search(rows, n_rows, k, max_iterations=5000, seed=seed)
                        if improved_scope < best_scope:
                            best_scope = improved_scope
                            best_rows = improved_rows
                        break  # Found a valid solution for this scope
    
    # If we found a solution, try to improve it further with more local search
    if best_rows is not None:
        # Do more local search iterations
        for iteration in range(5):
            if time.time() > deadline - 5:
                break
            improved_rows, improved_scope = local_search(best_rows, n_rows, k, max_iterations=10000, seed=iteration + 100)
            if improved_scope < best_scope:
                best_scope = improved_scope
                best_rows = improved_rows
    
    # If no solution found, fall back to a simple greedy without scope limit
    if best_rows is None:
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new = {x - a for a in row}
                    if not new & used:
                        row.append(x)
                        used |= new
                        break
                    x += 1
            rows.append(row)
        best_rows = rows
        best_scope = get_scope(rows)
    
    return best_rows

# Quick test
if __name__ == "__main__":
    # Test with the task parameters (4, 7)
    rows = construct(4, 7)
    scope = max(max(row) for row in rows)
    print(f"Scope: {scope}")
    print(f"Rows: {rows}")
    # Verify validity
    diffs = set()
    valid = True
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
    print(f"Valid: {valid}")
