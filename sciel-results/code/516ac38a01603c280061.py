"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Helper to check validity of a full set of rows
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Helper to get max value
    def get_max_val(rows):
        return max(max(r) for r in rows)

    # 1. Get a baseline using a fast stochastic global greedy
    # This provides a good upper bound for the binary search.
    
    def find_next_valid_fast(row, used, start_x, limit):
        """Find the smallest x >= start_x such that adding x to row doesn't conflict."""
        x = start_x
        while x <= limit:
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

    def stochastic_global_greedy(random_seed, limit):
        """Global greedy with random tie-breaking, constrained by limit."""
        random.seed(random_seed)
        used = set()
        rows = [[] for _ in range(n_rows)]
        for i in range(n_rows):
            rows[i].append(0)
        
        steps = 0
        max_steps = n_rows * k * 10 # Safety
        
        while steps < max_steps:
            if time.time() > deadline:
                return None
                
            min_x = float('inf')
            best_row = -1
            tied_rows = []
            
            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                x, diffs = find_next_valid_fast(rows[i], used, last + 1, limit)
                if x is None:
                    continue
                if x < min_x:
                    min_x = x
                    best_row = i
                    tied_rows = [i]
                elif x == min_x:
                    tied_rows.append(i)
            
            if best_row == -1:
                # No row could be extended within limit
                return None
            
            if len(tied_rows) > 1:
                chosen_row = random.choice(tied_rows)
            else:
                chosen_row = best_row
            
            last = rows[chosen_row][-1]
            x, diffs = find_next_valid_fast(rows[chosen_row], used, last + 1, limit)
            if x is None:
                return None
            
            rows[chosen_row].append(x)
            for d in diffs:
                used.add(d)
            steps += 1
            
            if all(len(r) == k + 1 for r in rows):
                return rows
                
        return None

    # 2. Backtracking Solver for a specific limit M
    # This is the core structural change: exact search within a bound.
    
    def solve_backtrack(limit):
        """
        Attempt to construct n_rows of length k+1 with max value <= limit.
        Returns the rows if successful, None otherwise.
        """
        # State: rows (list of lists), used (set of diffs)
        # We use a heuristic to decide which row to extend next.
        # Heuristic: Pick the row that has the smallest next valid candidate.
        # This is similar to MRV (Minimum Remaining Values) but simpler.
        
        rows = [[0] for _ in range(n_rows)]
        used = set()
        
        # Pre-compute initial state? No, 0 is fixed.
        
        def get_next_options(row, used, limit):
            """Get list of valid next values for this row, sorted."""
            if len(row) == k + 1:
                return []
            last = row[-1]
            opts = []
            # We only need the smallest few, but for simplicity in small k, 
            # finding the smallest valid one is enough for the greedy step in backtracking.
            # However, for backtracking, we might need to try the 2nd, 3rd smallest if the 1st fails.
            # To keep it fast, we generate candidates on the fly in the recursive step.
            pass
            
        def backtrack(rows, used, depth):
            if time.time() > deadline:
                return False
            
            # Check if all rows are full
            if all(len(r) == k + 1 for r in rows):
                return True
            
            # Select row to extend
            # Heuristic: Find the row with the smallest next valid number.
            # This helps prune branches where a row is "stuck" early.
            best_row_idx = -1
            best_next_val = float('inf')
            best_diffs = None
            
            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                # Find smallest valid x
                x = last + 1
                while x <= limit:
                    valid = True
                    diffs = []
                    for a in rows[i]:
                        d = x - a
                        if d in used:
                            valid = False
                            break
                        diffs.append(d)
                    if valid:
                        if x < best_next_val:
                            best_next_val = x
                            best_row_idx = i
                            best_diffs = diffs
                        break
                    x += 1
                
                if x > limit:
                    # This row cannot be extended within the limit
                    # If this row is not full, we are stuck.
                    if len(rows[i]) < k + 1:
                        return False

            if best_row_idx == -1:
                # All rows are full? (Handled above) or no move found?
                # If no move found but not all full, it's a dead end.
                return False
            
            # Try the best move first
            i = best_row_idx
            x = best_next_val
            diffs = best_diffs
            
            # Apply move
            rows[i].append(x)
            for d in diffs:
                used.add(d)
            
            # Recurse
            if backtrack(rows, used, depth + 1):
                return True
            
            # Backtrack
            rows[i].pop()
            for d in diffs:
                used.discard(d)
            
            # Try next possible value for this row?
            # The heuristic picked the *smallest* valid x.
            # If that failed, we should try the *next* smallest valid x for this specific row.
            # We need to iterate through candidates for the chosen row.
            
            last = rows[i][-1]
            # We know x was valid. We need to find the next valid x' > x.
            # Note: rows[i] is currently WITHOUT x.
            
            x_curr = x + 1
            while x_curr <= limit:
                valid = True
                diffs_curr = []
                for a in rows[i]:
                    d = x_curr - a
                    if d in used:
                        valid = False
                        break
                    diffs_curr.append(d)
                
                if valid:
                    # Apply
                    rows[i].append(x_curr)
                    for d in diffs_curr:
                        used.add(d)
                    
                    if backtrack(rows, used, depth + 1):
                        return True
                    
                    # Backtrack
                    rows[i].pop()
                    for d in diffs_curr:
                        used.discard(d)
                
                x_curr += 1
            
            return False

        # Start backtracking
        if backtrack(rows, used, 0):
            return rows
        return None

    # 3. Strategy:
    # a. Run a few greedy runs to find a decent upper bound (M_hi).
    # b. Binary search for the smallest M in [M_lo, M_hi] that is solvable by backtracking.
    
    # Step a: Find M_hi
    best_rows_greedy = None
    best_max_greedy = 1000000
    
    # Try a few greedy runs with a loose limit to find a solution
    # We can start with a high limit and lower it, or just run greedy without strict limit 
    # but stop if it gets too big.
    
    # Simple Greedy to get a starting point
    def simple_greedy():
        used = set()
        rows = [[0] for _ in range(n_rows)]
        for _ in range(n_rows * k):
            if time.time() > deadline:
                return None
            min_x = float('inf')
            best_row = -1
            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                x = last + 1
                while True:
                    valid = True
                    for a in rows[i]:
                        if (x - a) in used:
                            valid = False
                            break
                    if valid:
                        break
                    x += 1
                    if x > 500: # Safety
                        return None
                if x < min_x:
                    min_x = x
                    best_row = i
            if best_row == -1:
                return None
            rows[best_row].append(min_x)
            for a in rows[best_row][:-1]:
                used.add(min_x - a)
        return rows

    baseline_rows = simple_greedy()
    if baseline_rows:
        best_max_greedy = get_max_val(baseline_rows)
        best_rows_greedy = baseline_rows
    else:
        # Fallback
        best_max_greedy = 200
        best_rows_greedy = None

    # Try to improve M_hi with stochastic greedy
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        # Use a limit slightly above current best to see if we can find a better one?
        # Actually, stochastic greedy with tie-breaking often finds the same or slightly different.
        # Let's just run it with a high limit to see if it beats the baseline.
        rows = stochastic_global_greedy(seed, best_max_greedy + 10)
        if rows:
            m = get_max_val(rows)
            if m < best_max_greedy:
                best_max_greedy = m
                best_rows_greedy = rows

    # Step b: Binary Search for optimal M
    # Lower bound: Theoretical lower bound is roughly (n_rows * k * (k+1) / 2) / n_rows ? 
    # Actually, total differences is n_rows * k*(k+1)/2.
    # Max diff is M. Number of distinct diffs is M.
    # So M >= n_rows * k*(k+1)/2.
    # For n=4, k=7: 4 * 7*8/2 = 112.
    # So M_lo = 112.
    
    M_lo = n_rows * k * (k + 1) // 2
    M_hi = best_max_greedy
    
    # If M_hi is very high, we might want to shrink it first with a few backtrack attempts?
    # But binary search handles that.
    
    # We want to find the smallest M such that solve_backtrack(M) is not None.
    # We iterate M from M_lo to M_hi.
    # Since the range is small (112 to ~160), we can just iterate upwards.
    # This is safer than binary search if the solver is fast for small M.
    
    found_solution = None
    found_M = best_max_greedy
    
    # If we have a greedy solution, we can try to verify if we can go lower.
    # Iterate M from M_lo to best_max_greedy - 1
    for M in range(M_lo, best_max_greedy):
        if time.time() > deadline - 5:
            break
            
        # Run backtracking solver for limit M
        # To make it faster, we can use the greedy solution to guide the search?
        # No, keep it pure.
        
        sol = solve_backtrack(M)
        if sol is not None:
            found_solution = sol
            found_M = M
            # We found a solution for M. Can we go lower?
            # Yes, continue loop to check M-1.
            # But wait, the loop is increasing M.
            # If we find a solution at M, the optimal is <= M.
            # We want the *smallest* M.
            # So we should search from M_lo upwards until we find the FIRST valid M.
            # If we find one, we can stop? No, we want the *smallest*.
            # If we iterate upwards, the first one we find IS the smallest.
            break

    # If we didn't find anything lower than best_max_greedy, use best_rows_greedy
    if found_solution is not None:
        return found_solution
    else:
        return best_rows_greedy

# EVOLVE-BLOCK-END
