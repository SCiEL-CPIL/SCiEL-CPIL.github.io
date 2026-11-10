"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0, minimizing the largest entry,
    such that all positive differences within rows are distinct across all rows.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    
    # For the specific task: n_rows=4, k=7 (8 entries per row)
    # We'll use a randomized local search approach with restarts
    
    def compute_diffs(row):
        """Compute all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            diffs = compute_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True
    
    def objective(rows):
        """Return the largest entry (scope)."""
        return max(max(row) for row in rows)
    
    def greedy_row(used_diffs, k, start_val=0):
        """Greedy construction of a single row avoiding used differences."""
        row = [0]
        x = start_val
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        return row
    
    def random_row(k, max_val):
        """Generate a random row of k+1 increasing integers starting at 0."""
        # Choose k values from 1..max_val
        if max_val < k:
            return None
        vals = sorted(random.sample(range(1, max_val + 1), k))
        return [0] + vals
    
    def local_search(rows, max_iterations=5000):
        """Local search to minimize the scope."""
        best_rows = [row[:] for row in rows]
        best_scope = objective(best_rows)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > deadline:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            current_row = best_rows[row_idx]
            
            # Try to improve this row by reducing its maximum
            current_max = current_row[-1]
            
            # Generate candidate rows with smaller max
            improved = False
            for _ in range(20):
                new_max = random.randint(k, current_max - 1)
                if new_max < k:
                    continue
                candidate = random_row(k, new_max)
                if candidate is None:
                    continue
                
                # Check if replacing this row keeps validity
                new_rows = best_rows[:]
                new_rows[row_idx] = candidate
                
                # Check difference conflicts
                all_diffs = set()
                valid = True
                for i, row in enumerate(new_rows):
                    diffs = compute_diffs(row)
                    if diffs & all_diffs:
                        valid = False
                        break
                    all_diffs |= diffs
                
                if valid:
                    new_scope = objective(new_rows)
                    if new_scope < best_scope:
                        best_rows = new_rows
                        best_scope = new_scope
                        improved = True
                        break
            
            if not improved:
                # Try a different move: perturb a random entry
                row_idx = random.randint(0, n_rows - 1)
                current_row = best_rows[row_idx]
                
                # Pick a random position to change (not the first or last)
                pos = random.randint(1, len(current_row) - 2)
                old_val = current_row[pos]
                
                # Try nearby values
                for delta in random.sample(range(1, 10), 5):
                    for sign in [-1, 1]:
                        new_val = old_val + sign * delta
                        if new_val <= 0 or new_val >= current_row[-1]:
                            continue
                        
                        # Check if it maintains strict increase
                        if pos > 0 and new_val <= current_row[pos - 1]:
                            continue
                        if pos < len(current_row) - 1 and new_val >= current_row[pos + 1]:
                            continue
                        
                        candidate = current_row[:]
                        candidate[pos] = new_val
                        
                        new_rows = best_rows[:]
                        new_rows[row_idx] = candidate
                        
                        # Check validity
                        all_diffs = set()
                        valid = True
                        for row in new_rows:
                            diffs = compute_diffs(row)
                            if diffs & all_diffs:
                                valid = False
                                break
                            all_diffs |= diffs
                        
                        if valid:
                            new_scope = objective(new_rows)
                            if new_scope <= best_scope:
                                best_rows = new_rows
                                best_scope = new_scope
                                break
        
        return best_rows, best_scope
    
    # Initial solution using greedy
    used = set()
    initial_rows = []
    for _ in range(n_rows):
        row = greedy_row(used, k)
        initial_rows.append(row)
        used |= compute_diffs(row)
    
    best_rows = [row[:] for row in initial_rows]
    best_scope = objective(best_rows)
    
    # Local search with restarts
    num_restarts = 0
    while time.time() - start_time < deadline:
        # Local search from current best
        improved_rows, improved_scope = local_search(best_rows, max_iterations=3000)
        if improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
        
        # Occasionally restart from a different initial solution
        num_restarts += 1
        if num_restarts % 5 == 0:
            # Try a different greedy order
            random.seed(int(time.time() * 1000) % 100000)
            used = set()
            alt_rows = []
            for _ in range(n_rows):
                row = greedy_row(used, k)
                alt_rows.append(row)
                used |= compute_diffs(row)
            
            alt_scope = objective(alt_rows)
            if alt_scope < best_scope:
                best_rows = alt_rows
                best_scope = alt_scope
            else:
                # Local search from the alternative
                improved_rows, improved_scope = local_search(alt_rows, max_iterations=2000)
                if improved_scope < best_scope:
                    best_rows = improved_rows
                    best_scope = improved_scope
    
    return best_rows
# EVOLVE-BLOCK-END
