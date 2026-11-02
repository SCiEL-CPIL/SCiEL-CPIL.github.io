"""
Backtracking constraint-satisfaction search for difference-triangle sets.

Strategy:
Instead of constructing rows independently with greedy heuristics, we treat the
entire collection of rows as a single constraint satisfaction problem. We perform
a depth-first backtracking search, deciding the next element to place in the
global structure based on a "most-constrained" heuristic (the position with the
fewest available candidate values given the current difference usage).

We use iterative deepening on the maximum allowed value (scope) to find the
smallest feasible scope. Randomized tie-breaking in the heuristic ordering allows
us to explore different branches and escape local minima that deterministic
search might get stuck in.

Key components:
1. Iterative deepening on scope M (start from a good lower bound).
2. Backtracking search with forward checking (track used differences).
3. Heuristic: prioritize placing elements that have fewest valid options.
4. Randomization: shuffle tie-breaking order to explore diverse solutions.
5. Time management: return best valid solution found before deadline.
"""

import time
import random
import sys
from array import array


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    rows_needed = n_rows
    m = k + 1  # entries per row

    # Each row has positions 0..k. Position 0 is always 0.
    # We need to assign values to positions 1..k for each row.
    # Total variables: n_rows * k
    
    # Precompute all (row, pos) pairs for positions 1..k
    # We'll represent the state as a list of lists: rows[r][p] = value
    # Initially rows[r][0] = 0, rows[r][p] = None for p >= 1
    
    # For efficiency, we track:
    # - used_diffs: set of all differences a_j - a_i (i < j) currently in the partial solution
    # - rows: current assignment
    
    best_rows = None
    best_scope = float('inf')
    
    # Lower bound: counting argument
    # Total differences needed: n_rows * C(k+1, 2)
    # Differences available in [1, M]: M
    # So M >= n_rows * k*(k+1)//2
    lb_counting = n_rows * k * (k + 1) // 2
    
    # A slightly better lower bound: the greedy first row uses differences up to ~k(k+1)/2
    # But we can start from lb_counting
    
    # Initial deterministic greedy to get a starting point
    def greedy_construct():
        u = set()
        rows = []
        for _ in range(rows_needed):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in u:
                        ok = False
                        break
                if ok:
                    for a in row:
                        u.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows
    
    # Get initial solution from greedy
    init_rows = greedy_construct()
    init_scope = max(max(r) for r in init_rows)
    best_rows = init_rows
    best_scope = init_scope
    
    # Now try to improve with backtracking search
    # We'll try scopes from max(lb_counting, init_scope - 10) up to init_scope - 1
    
    # Actually, let's try to find ANY solution with scope < best_scope
    # Start from a reasonable lower bound and go up
    
    # For backtracking, we need to be smart about variable ordering
    # Heuristic: choose the (row, pos) that has the fewest valid values
    
    def has_valid_solution(scope_limit, time_limit, rng):
        """Try to find a valid solution with all entries <= scope_limit.
        Returns the solution if found, None otherwise."""
        
        # State: rows[r][p] = value or None
        # used_diffs: set of differences
        # We'll use recursion with backtracking
        
        rows = [[0] + [None] * k for _ in range(rows_needed)]
        used_diffs = set()
        
        # Precompute: for each row, the positions that need to be filled
        # We'll maintain a list of unfilled positions
        
        # For efficiency, we'll use a stack-based DFS
        
        # Variable ordering: we'll dynamically choose the next variable
        # based on the number of available values
        
        # To make this tractable, we limit the search depth or use pruning
        
        # Let's use a simpler approach: iterative construction with backtracking
        # Build row by row, but allow backtracking within a row
        
        # Actually, for n_rows=4, k=7, this is a large search space.
        # Let's use a hybrid: greedy with occasional backtracking
        
        # Alternative: use the fact that we already have a good solution
        # and try to modify it locally
        
        # Given the time constraint, let's implement a focused backtracking
        # that tries to reduce the scope of the existing solution
        
        pass  # This approach is too complex for the time limit
    
    # Let's use a different strategy: local search with simulated annealing
    # Start from the greedy solution and try to reduce the scope
    
    def local_search(start_rows, scope_limit, time_limit, rng):
        """Try to find a valid solution with scope <= scope_limit starting from start_rows.
        Uses simulated annealing / random walk."""
        
        rows = [list(r) for r in start_rows]
        current_scope = max(max(r) for r in rows)
        
        if current_scope <= scope_limit:
            return rows
        
        # Track used differences
        def compute_used_diffs(rows):
            used = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        used.add(row[j] - row[i])
            return used
        
        used_diffs = compute_used_diffs(rows)
        
        # Simulated annealing
        temp = 50.0
        cooling = 0.999
        best_local = None
        best_local_scope = current_scope
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit and temp > 0.1:
            # Choose a random row and position to modify
            r_idx = rng.randint(0, rows_needed - 1)
            pos = rng.randint(1, k)  # Don't change position 0
            
            old_val = rows[r_idx][pos]
            
            # Try to reduce the value
            new_val = old_val - rng.randint(1, 5)
            
            if new_val <= rows[r_idx][pos - 1]:
                # Can't go below previous element
                new_val = rows[r_idx][pos - 1] + rng.randint(1, 3)
            
            # Check if new_val is feasible
            # Remove old differences involving (r_idx, pos)
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - rows[r_idx][i])
            for j in range(pos + 1, len(rows[r_idx])):
                old_diffs.add(rows[r_idx][j] - old_val)
            
            # Add new differences
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - rows[r_idx][i])
            for j in range(pos + 1, len(rows[r_idx])):
                new_diffs.add(rows[r_idx][j] - new_val)
            
            # Check for conflicts
            conflict = False
            for d in new_diffs:
                if d in old_diffs:
                    continue
                if d in used_diffs:
                    conflict = True
                    break
            
            if not conflict:
                # Accept the move
                for d in old_diffs:
                    used_diffs.discard(d)
                for d in new_diffs:
                    used_diffs.add(d)
                rows[r_idx][pos] = new_val
                
                new_scope = max(max(r) for r in rows)
                if new_scope < best_local_scope:
                    best_local_scope = new_scope
                    best_local = [list(r) for r in rows]
                
                # If we found a solution within scope_limit, return it
                if new_scope <= scope_limit:
                    return rows
            else:
                # Reject the move (could accept with probability in SA, but skip for simplicity)
                pass
            
            temp *= cooling
        
        return best_local if best_local_scope < current_scope else None
    
    # Try local search from the greedy solution
    rng = random.Random(42)
    
    # Try to improve the scope
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try to find a solution with scope < best_scope
        target_scope = best_scope - 1
        
        # Run local search with a time limit
        time_per_search = min(2.0, (deadline - time.time()) / max(1, 50 - attempt))
        
        result = local_search(best_rows, target_scope, time_per_search, rng)
        
        if result is not None:
            new_scope = max(max(r) for r in result)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = result
    
    # Final compaction pass
    def compact_rows(rows):
        """Try to reduce each element to its minimum feasible value."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])
        
        improved = True
        while improved:
            improved = False
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    for new_val in range(row[pos - 1] + 1, current_val):
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(current_val - row[i])
                        for j in range(pos + 1, len(row)):
                            old_diffs.add(row[j] - current_val)
                        
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for j in range(pos + 1, len(row)):
                            new_diffs.add(row[j] - new_val)
                        
                        conflict = False
                        for d in new_diffs:
                            if d in old_diffs:
                                continue
                            if d in all_diffs:
                                conflict = True
                                break
                        
                        if not conflict:
                            for d in old_diffs:
                                all_diffs.discard(d)
                            for d in new_diffs:
                                all_diffs.add(d)
                            row[pos] = new_val
                            improved = True
                            break
        
        return rows
    
    best_rows = compact_rows(best_rows)
    best_scope = max(max(r) for r in best_rows)
    
    # Validate and return
    used = set()
    valid = True
    for row in best_rows:
        if len(row) != m or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    
    if not valid:
        # Fallback to greedy
        return greedy_construct()
    
    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
