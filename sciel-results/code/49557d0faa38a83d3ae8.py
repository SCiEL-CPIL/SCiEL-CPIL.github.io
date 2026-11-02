"""Improved solver for difference triangle sets using randomized local search with backtracking."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Total number of differences per row: k*(k+1)/2
    # Total differences across all rows: n_rows * k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Counting bound: need m such that m*(m+1)/2 >= total_diffs
    # Solve m^2 + m - 2*total_diffs >= 0
    import math
    m_lower = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    # Best known upper bound is around 134 for (4,7)
    # We'll search from a reasonable upper bound down
    
    # Start with greedy solution to get an initial upper bound
    best_scope = None
    best_rows = None
    
    def greedy_construct(n_rows, k, max_val=None):
        """Greedy construction: for each row, pick smallest available marks."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
                if max_val is not None and x > max_val:
                    return None
            rows.append(row)
        scope = max(max(r) for r in rows)
        return rows, scope
    
    # Get initial greedy solution
    greedy_result = greedy_construct(n_rows, k)
    if greedy_result:
        best_rows, best_scope = greedy_result
    
    # Try to improve with local search
    # Strategy: start with a feasible solution and try to reduce the scope
    # by swapping elements or using simulated annealing on the assignment
    
    def validate(rows, k):
        """Check if all differences are distinct across all rows."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    # Local search: try to reduce scope by modifying rows
    # Use a randomized approach: pick a random row, try to rebuild it with a smaller max
    
    def try_improve(rows, k, n_rows, max_scope, time_limit):
        """Try to find a solution with scope < max_scope."""
        target_scope = max_scope - 1
        if target_scope < m_lower:
            return None
        
        # Try randomized construction with constraint that max entry <= target_scope
        for attempt in range(1000):
            if time.time() > time_limit:
                break
            
            # Randomized greedy with backtracking
            used = set()
            new_rows = []
            success = True
            
            for row_idx in range(n_rows):
                row = [0]
                # Try to build the row with values up to target_scope
                # Use randomized order for trying values
                remaining = list(range(1, target_scope + 1))
                random.shuffle(remaining)
                
                row_complete = False
                for x in remaining:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        if len(row) == k + 1:
                            used |= new_diffs
                            row_complete = True
                            break
                
                if not row_complete:
                    success = False
                    break
            
            if success:
                scope = get_scope(new_rows)
                if scope < max_scope:
                    return new_rows, scope
        
        return None
    
    # Iteratively try to reduce the scope
    current_scope = best_scope
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        result = try_improve(best_rows, k, n_rows, current_scope, deadline)
        if result:
            best_rows, current_scope = result
            # Reset attempt counter when we make progress
            iterations = 0
        else:
            # No improvement found, try a different approach
            # Use simulated annealing-like perturbation
            if time.time() > deadline - 10:
                break
            
            # Perturb the current solution and try to repair
            perturbed = [list(r) for r in best_rows]
            # Pick a random row and a random element (not the first or last)
            row_idx = random.randint(0, n_rows - 1)
            pos_idx = random.randint(1, k)  # Not the first element (0)
            old_val = perturbed[row_idx][pos_idx]
            
            # Try replacing with a smaller value
            new_val = max(1, old_val - random.randint(1, 5))
            perturbed[row_idx][pos_idx] = new_val
            
            # Check if still valid
            if validate(perturbed, k):
                new_scope = get_scope(perturbed)
                if new_scope < current_scope:
                    best_rows = perturbed
                    current_scope = new_scope
                    iterations = 0
    
    # Final validation
    if best_rows is None or not validate(best_rows, k):
        # Fall back to greedy
        greedy_result = greedy_construct(n_rows, k)
        if greedy_result:
            best_rows, best_scope = greedy_result
        else:
            # Last resort: very conservative construction
            best_rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    x += 1
                best_rows.append(row)
    
    return best_rows
# EVOLVE-BLOCK-END
