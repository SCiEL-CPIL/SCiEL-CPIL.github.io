import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    # Keep a safety margin: the evaluator kills after 150s, so use ~140s budget.
    deadline = start_time + 140.0

    lower_bound = n_rows * k * (k + 1) // 2

    # Get a fast, valid baseline via greedy.
    baseline_solution, baseline_scope = greedy_solve(n_rows, k)
    if not baseline_solution:
        return []

    best_solution = baseline_solution
    best_scope = baseline_scope

    if best_scope <= lower_bound:
        return best_solution

    # We will try to find a solution with a smaller scope.
    # Strategy:
    # 1. Iterate M from lower_bound up to best_scope - 1.
    # 2. For each M, run a randomized backtracking search with multiple restarts.
    # 3. If a solution is found for M, return it immediately (since we search in increasing M).
    # 4. If time runs out, return the best solution found so far (which is the baseline or a better one).

    # To maximize the chance of finding a solution for a given M, we use:
    # - Multiple random seeds.
    # - A smarter candidate ordering (not just random shuffle, but maybe weighted by "tightness" or just random).
    # - Early termination if time is low.

    for M in range(lower_bound, best_scope):
        if time.time() > deadline:
            break

        # How many attempts per M?
        # We don't know the difficulty. Start with a few, increase if M is close to lower_bound?
        # Actually, if M is small, the search space is tighter, so it might be harder or easier?
        # Usually, finding a solution at the lower bound is hardest.
        # Let's allocate a fixed time budget per M or a fixed number of attempts.
        
        # Let's try up to 10 attempts per M, or until 1 second has passed for this M (whichever comes first).
        # But if we have lots of time, we can do more.
        
        # Adaptive attempts:
        # If M is close to lower_bound, we might need more attempts? Or less because it's constrained?
        # Let's just do a fixed number of attempts, say 5-10, and see.
        
        found = False
        attempts = 0
        max_attempts = 20  # Increase attempts to improve chance
        
        while not found and attempts < max_attempts:
            if time.time() > deadline:
                break
            
            # Use a random seed for this attempt
            seed = random.randint(0, 2**31 - 1)
            
            sol = solve_dts_random(n_rows, k, M, deadline, seed)
            
            if sol is not None:
                best_solution = sol
                best_scope = M
                found = True
                # Since we are iterating M in increasing order, this is the optimal M found so far.
                # We can return immediately.
                return sol
            
            attempts += 1
        
        # If we didn't find a solution for this M, continue to M+1.
        # Note: If we fail for M, we might still find a solution for M+1.
        # It is possible that M is impossible, or we just didn't find it.
        # Given the small size (4x8), if M is feasible, random search should find it reasonably quickly.
        # If M is infeasible, we waste time. 
        # Heuristic: If we fail for M, and M is significantly above lower_bound, maybe skip some?
        # But we want the *smallest* M. So we must check each M.
        # However, if M is infeasible, we don't know that for sure. 
        # We can't skip M unless we are sure. 
        # For now, we check every M.

    return best_solution


def greedy_solve(n_rows, k):
    """
    Fast greedy construction to get a valid baseline solution.
    This is not optimized for minimal scope, just for validity.
    """
    used_diffs = 0  # Bitset of used differences
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        # We need k more numbers
        # Greedy: pick the smallest possible x that doesn't conflict
        for _ in range(k):
            x = row[-1] + 1
            while True:
                # Calculate conflicts with current row
                conflict_mask = 0
                for a in row:
                    diff = x - a
                    if diff <= 0:
                        continue
                    conflict_mask |= (1 << diff)
                
                if (used_diffs & conflict_mask) == 0:
                    # No conflict
                    row.append(x)
                    used_diffs |= conflict_mask
                    break
                x += 1
        rows.append(row)
        
    return rows, max(max(row) for row in rows)


def solve_dts_random(n_rows, k, max_scope, deadline, seed=0):
    """
    Solves DTS with max entry <= max_scope using randomized backtracking.
    """
    rng = random.Random(seed)
    used_diffs = 0
    rows = []
    
    # Pre-calculate some things?
    # The bitset operations are fast in Python.
    
    def backtrack_row(row_idx, pos, current_last, row_so_far, row_mask):
        if time.time() > deadline:
            return None
        
        if pos == k + 1:
            return row_so_far
        
        # Pruning: check if we can fit the remaining numbers
        min_possible_last = current_last + (k + 1 - pos)
        if min_possible_last > max_scope:
            return None
        
        max_x = max_scope - (k - pos)
        
        # Generate valid candidates
        candidates = []
        for x in range(current_last + 1, max_x + 1):
            conflict_mask = 0
            for a in row_so_far:
                d = x - a
                conflict_mask |= (1 << d)
            
            if ((used_diffs | row_mask) & conflict_mask) == 0:
                candidates.append(x)
        
        if not candidates:
            return None
        
        # Randomize the order of candidates
        rng.shuffle(candidates)
        
        # Try each candidate
        for x in candidates:
            conflict_mask = 0
            for a in row_so_far:
                d = x - a
                conflict_mask |= (1 << d)
            
            new_row_mask = row_mask | conflict_mask
            new_row = row_so_far + [x]
            
            res = backtrack_row(row_idx, pos + 1, x, new_row, new_row_mask)
            if res is not None:
                return res
        
        return None

    for i in range(n_rows):
        row = backtrack_row(i, 1, 0, [0], 0)
        if row is None:
            return None
        rows.append(row)
        # Add this row's differences to the global used_diffs
        for j in range(len(row)):
            for l in range(j+1, len(row)):
                d = row[l] - row[j]
                used_diffs |= (1 << d)
        
        # Check time
        if time.time() > deadline:
            return None

    return rows
