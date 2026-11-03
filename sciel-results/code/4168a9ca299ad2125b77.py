"""Difference triangle set (7,5) solver: minimize the largest entry (scope)."""
import time
import random
import sys

def construct(n_rows, k):
    # Deadline guard: evaluator kills at 150s; keep a safety margin.
    t0 = time.time()
    deadline = t0 + 130.0  # 20s safety margin

    # Total number of positive differences per row: k*(k+1)//2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # We need to assign total_diffs distinct positive integers.
    # Lower bound on scope: the largest difference is at least total_diffs.
    # Also, each row has k+1 marks starting at 0, so max mark >= k (trivial).
    # The counting bound: largest entry >= total_diffs (since all diffs distinct and positive).
    # For n_rows=7, k=5: total_diffs = 7*15 = 105. Best known: 112.

    # Strategy: Use a simulated annealing / local search approach.
    # Represent each row as a list of k+1 integers [0, a1, a2, ..., ak] with 0 < a1 < ... < ak.
    # The objective is to minimize the maximum entry across all rows, subject to all
    # positive differences within each row being distinct across all rows.

    # First, get a feasible starting solution via greedy.
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    def get_scope(rows):
        return max(max(row) for row in rows)

    def get_all_diffs(rows):
        """Return list of all positive differences and check validity."""
        all_diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.append(row[j] - row[i])
        return all_diffs

    def is_valid(rows):
        all_diffs = get_all_diffs(rows)
        return len(all_diffs) == len(set(all_diffs))

    def count_conflicts(rows):
        """Return number of duplicate differences (0 means valid)."""
        all_diffs = get_all_diffs(rows)
        return len(all_diffs) - len(set(all_diffs))

    # Start with greedy
    rows = greedy_solution()
    best_rows = [list(r) for r in rows]
    best_scope = get_scope(best_rows)
    
    # Ensure best is valid
    if not is_valid(best_rows):
        # Shouldn't happen with greedy, but be safe
        pass

    # Local search: try to reduce the scope by modifying rows.
    # We'll use a restart-based local search with simulated annealing.
    
    def local_search(initial_rows, max_time):
        """Perform local search to minimize scope while maintaining validity."""
        rows = [list(r) for r in initial_rows]
        best_local = [list(r) for r in rows]
        best_local_scope = get_scope(rows)
        
        # We'll do iterative improvement: try to reduce the max entry of each row.
        # For each row, try to find a smaller valid configuration.
        
        # Simulated annealing style:
        # State: the rows
        # Move: pick a row and a mark, try a new value
        # Cost: number of conflicts (want 0) + penalty for scope
        
        import copy
        
        current = [list(r) for r in rows]
        current_conflicts = count_conflicts(current)
        current_scope = get_scope(current)
        
        T = 10.0  # initial temperature
        cooling = 0.999
        min_T = 0.01
        
        t_start = time.time()
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < max_time and T > min_T:
            # Pick a random row and a random mark position (1..k, not position 0)
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # positions 1..k
            
            # Try changing current[r_idx][pos] to a nearby value
            old_val = current[r_idx][pos]
            # New value must maintain strict increase in the row
            lo = current[r_idx][pos - 1] + 1
            hi = current[r_idx][pos + 1] - 1 if pos < k else best_scope + 5
            
            if lo >= hi:
                T *= cooling
                continue
            
            # Try a few random candidates
            improved_this_iter = False
            for _ in range(5):
                new_val = random.randint(lo, hi)
                if new_val == old_val:
                    continue
                
                # Tentatively change
                old = current[r_idx][pos]
                current[r_idx][pos] = new_val
                
                new_conflicts = count_conflicts(current)
                new_scope = get_scope(current)
                
                # Accept if better or with probability based on temperature
                old_cost = current_conflicts * 1000 + current_scope
                new_cost = new_conflicts * 1000 + new_scope
                
                if new_cost < old_cost:
                    # Accept
                    current_conflicts = new_conflicts
                    current_scope = new_scope
                    if new_conflicts == 0 and new_scope < best_local_scope:
                        best_local_scope = new_scope
                        best_local = [list(r) for r in current]
                        no_improve = 0
                        improved_this_iter = True
                    elif new_conflicts == 0:
                        no_improve = 0
                        improved_this_iter = True
                    else:
                        no_improve += 1
                else:
                    # Reject, revert
                    current[r_idx][pos] = old
                    no_improve += 1
            
            if improved_this_iter:
                T *= 0.99
            else:
                T *= cooling
            
            if no_improve > max_no_improve:
                break
            
            if time.time() >= max_time:
                break
        
        return best_local, best_local_scope

    # Multi-start local search
    random.seed(42)
    
    # First, try to improve the greedy solution
    while time.time() < deadline - 10:
        # Restart: perturb the best solution slightly
        perturbed = [list(r) for r in best_rows]
        # Randomly change a few marks
        for _ in range(random.randint(1, 3)):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            lo = perturbed[r_idx][pos - 1] + 1
            hi = perturbed[r_idx][pos + 1] - 1 if pos < k else best_scope + 10
            if lo < hi:
                perturbed[r_idx][pos] = random.randint(lo, hi)
        
        # Run local search on the perturbed solution
        time_left = deadline - time.time() - 5
        if time_left < 1:
            break
        
        local_best, local_scope = local_search(perturbed, time_left)
        
        if local_scope < best_scope and is_valid(local_best):
            best_rows = local_best
            best_scope = local_scope
    
    # Final check: ensure validity
    if not is_valid(best_rows):
        # Fall back to greedy
        best_rows = greedy_solution()
        best_scope = get_scope(best_rows)
    
    return best_rows
