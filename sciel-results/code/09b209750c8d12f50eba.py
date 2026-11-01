import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Problem Parameters
    # n_rows sets, each of size k+1, starting with 0.
    # Differences must be distinct across all rows.
    # Minimize max element (scope).
    
    # Total differences per row: C(k+1, 2) = k*(k+1)/2
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row
    
    # Lower bound for scope M:
    # We need total_diffs_needed distinct differences.
    # Differences are in range [1, M].
    # So M >= total_diffs_needed.
    # Also, each row has max element <= M.
    # Simple counting bound: M >= total_diffs_needed.
    
    # Strategy:
    # 1. Estimate a tight lower bound for M.
    # 2. Try to construct the solution using backtracking with MRV.
    # 3. If backtracking fails or times out, fall back to a greedy/heuristic approach 
    #    that minimizes conflicts, then try to repair.
    
    # For small k (like 7) and n_rows (like 4), the search space is huge but structured.
    # The "difference triangle" structure implies that if we fix the differences, 
    # the row is determined up to translation? No, the row starts at 0.
    # A row is defined by its set of differences? 
    # If we know all differences in a row, can we reconstruct the row?
    # Not uniquely, but the differences must form a "Sidon set" structure relative to 0?
    # Actually, a set of numbers {0, a1, ..., ak} generates differences.
    # The differences are not independent.
    
    # Alternative View:
    # Construct rows one by one.
    # When constructing a row, we must avoid differences already used by previous rows.
    # Also, differences within the row must be distinct.
    
    # Let's try a "Greedy with Backtracking" on the rows themselves.
    # Order of rows: The first row is the most constrained? Or the last?
    # Usually, filling the most constrained variable first is good.
    # Here, all rows have the same constraints except for the "used differences" set.
    # The set of used differences grows as we add rows.
    # So, the last row has the most used differences to avoid.
    # Therefore, it is better to construct the "hardest" rows first?
    # Or, we can view it as: Assign a set of differences to each row.
    
    # Let's stick to the directive: "Represent the problem as a constraint satisfaction puzzle where the global difference set is partitioned into n_rows disjoint Sidon sets... using MRV heuristics that select the next difference to place based on the tightest available gap constraints"
    
    # This suggests we should think in terms of differences directly.
    # However, a row is not just a set of differences; it's a set of points.
    # But if we know the points, we know the differences.
    # If we know the differences, can we find the points?
    # For a row starting at 0, the differences are d_{ij} = x_j - x_i.
    # This is a "turnpike" problem variant.
    
    # Let's try a hybrid approach:
    # 1. Greedy initialization to get a valid or near-valid set.
    # 2. Use Local Search (Simulated Annealing or Hill Climbing) to minimize scope.
    #    The previous program used GA. Let's try pure SA with a specific move set.
    #    Moves:
    #    - Change one element of a row (maintaining order).
    #    - Swap two rows? (No, rows are indistinguishable in score, but distinct in construction).
    #    - "Slide" a row to fit in gaps?
    
    # Actually, the directive asks for "deterministic backtracking ... MRV".
    # Let's implement a limited-depth backtracking solver that tries to find a valid configuration
    # with a specific scope M. We iterate M from lower bound upwards.
    
    # Lower bound estimation:
    # M_min = total_diffs_needed
    # But often M is significantly larger.
    # For n_rows=4, k=7: total_diffs = 4 * 28 = 112.
    # Best known is 134.
    
    # Backtracking State:
    # rows: list of lists.
    # used_diffs: set of all differences used so far.
    # row_idx: which row we are currently filling.
    # pos: which position in the current row we are filling.
    # current_row: the list of numbers in the current row so far.
    
    # MRV Heuristic:
    # When choosing the next number for the current row, we look at the candidates.
    # But the "variable" here is the number itself.
    # A better MRV might be: Which row is "tightest"?
    # Or, within a row, which position is most constrained?
    
    # Let's try a simpler deterministic backtracking:
    # 1. Sort rows by "difficulty" (maybe just index, or estimated max).
    # 2. For each row, try to build it greedily with backtracking.
    # 3. If a row fails, backtrack to the previous row and try a different configuration for it.
    
    # This is essentially DFS with row-level backtracking.
    
    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_row(row, used_diffs):
        # Check internal validity (strictly increasing, starts at 0)
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        
        # Check difference uniqueness
        row_diffs = get_diffs(row)
        if len(row_diffs) != len(row) * (len(row) - 1) // 2:
            return False
        
        # Check against used diffs
        for d in row_diffs:
            if d in used_diffs:
                return False
        return True

    # We will use a recursive backtracking that builds rows one by one.
    # To make it efficient, we use a "Greedy with Backtracking" for each row.
    
    best_solution = None
    best_scope = float('inf')
    
    # Estimate starting M
    # Counting bound: M >= total_diffs_needed
    # Let's start from a slightly higher value to allow for structure
    start_m = total_diffs_needed
    
    # We will try increasing M until we find a solution or time out.
    # For each M, we run a backtracking search.
    
    # To speed up, we can use a greedy initialization to find a feasible M quickly,
    # then use backtracking to optimize (reduce M) or find a valid one if greedy failed.
    
    # Let's first try a fast greedy to get an upper bound.
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                x = current_max + 1
                while True:
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                        # Also check internal diffs
                        for b in row[:-1]:
                            if (x - a) == (x - b): # This is always false if a!=b
                                pass
                            # Internal diff check: x - a must not equal any existing diff in row
                            # Existing diffs are row[j] - row[i]
                            # We need x - a not in {row[j]-row[i] for i<j}
                            # This is equivalent to: x - a not in row_diffs_so_far
                            
                    # Let's just compute row_diffs each time for clarity, it's small
                    row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i+1, len(row)):
                            row_diffs.add(row[j]-row[i])
                    
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs or d in row_diffs:
                            valid = False
                            break
                    
                    if valid:
                        row.append(x)
                        used_diffs.update(x - a for a in row[:-1])
                        current_max = x
                        break
                    x += 1
            rows.append(row)
        return rows

    initial_rows = greedy_construct()
    initial_scope = max(max(r) for r in initial_rows)
    best_solution = initial_rows
    best_scope = initial_scope
    
    # Now, try to improve using backtracking with a target scope.
    # We try scopes from best_scope down to start_m.
    # If we find a valid solution with scope M, we update best.
    # If we run out of time, we return best.
    
    # Backtracking function
    # state:
    #   rows: list of completed rows
    #   used_diffs: set
    #   target_max: M
    
    def backtrack(rows, used_diffs, target_max):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline:
            return False
            
        if len(rows) == n_rows:
            # Solution found
            scope = max(max(r) for r in rows)
            if scope <= target_max:
                # Check if this is better than current best
                if scope < best_scope:
                    best_scope = scope
                    best_solution = [r[:] for r in rows]
            return True
        
        # MRV Heuristic:
        # We are constructing the next row (index len(rows)).
        # To make it "MRV", we could try to construct the row that has the fewest options first.
        # But all rows are symmetric.
        # However, the "used_diffs" set grows.
        # So the later rows have fewer options.
        # It is often better to construct the "hardest" rows first.
        # Since all rows are identical in constraint, the order doesn't matter for symmetry,
        # but the "used_diffs" content matters.
        
        # Let's try to construct the next row.
        # We will use a recursive helper to build the row with backtracking.
        
        def build_row(current_row, used_diffs_local, target_max_local):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                # Row complete. Check if it fits in target_max
                if current_row[-1] <= target_max_local:
                    # Add to rows and recurse
                    rows.append(current_row)
                    # Update used_diffs
                    row_diffs = get_diffs(current_row)
                    used_diffs_local.update(row_diffs)
                    
                    success = backtrack(rows, used_diffs_local, target_max_local)
                    
                    rows.pop()
                    # Remove diffs from used_diffs_local? 
                    # Since we are passing a copy or modifying in place?
                    # In Python, sets are passed by reference. We need to be careful.
                    # Let's pass a copy to be safe, or undo changes.
                    # For performance, undo changes.
                    for d in row_diffs:
                        used_diffs_local.discard(d)
                    
                    return success
                else:
                    return False
            
            # Choose next element
            # Lower bound: current_row[-1] + 1
            # Upper bound: target_max_local (if last element) or something smaller
            # If this is not the last element, we need to leave room for remaining elements.
            # Min increment is 1. So if we are at pos p (0-indexed, so p+1 elements),
            # we need (k - p) more elements.
            # So current_row[-1] + (k - p) <= target_max_local
            # x <= target_max_local - (k - p)
            
            pos = len(current_row)
            min_x = current_row[-1] + 1
            # Max x such that we can still fit k - pos elements
            max_x = target_max_local - (k - pos)
            
            # MRV: Try values that are most constrained?
            # Usually, smaller values are more "crowded" with differences.
            # Let's iterate x from min_x to max_x.
            # To improve MRV, we could prioritize x that result in "rare" differences?
            # Or just stick to ascending order (greedy) with backtracking.
            
            # Optimization: Precompute row_diffs for current_row
            row_diffs = set()
            for i in range(len(current_row)):
                for j in range(i+1, len(current_row)):
                    row_diffs.add(current_row[j] - current_row[i])
            
            for x in range(min_x, max_x + 1):
                # Check validity
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs_local or d in row_diffs:
                        valid = False
                        break
                
                if valid:
                    # Recurse
                    new_row = current_row + [x]
                    if build_row(new_row, used_diffs_local, target_max_local):
                        return True
            
            return False
        
        # Start building the next row
        initial_row = [0]
        return build_row(initial_row, used_diffs, target_max)

    # Try to improve
    # We start from best_scope - 1 down to start_m
    # We limit the number of backtracking attempts or time per M.
    
    current_m = best_scope - 1
    while current_m >= start_m and time.time() < deadline:
        # Reset state
        rows = []
        used_diffs = set()
        
        # Time limit for this M: e.g., 5 seconds
        m_start = time.time()
        
        if backtrack(rows, used_diffs, current_m):
            # Found a solution with scope <= current_m
            # best_solution is updated in backtrack
            # We can try even smaller M
            current_m -= 1
        else:
            # Failed to find solution with this M
            # Try smaller M? No, if we can't fit in M, we can't fit in M-1.
            # Wait, backtrack returns True if found.
            # If it returns False, it means no solution exists with max <= current_m?
            # Not necessarily, it might have timed out or pruned incorrectly.
            # But assuming correct pruning, if False, no solution with max <= current_m.
            # So we should STOP decreasing M.
            # Because if no solution for M, no solution for M-1.
            break
        
    return best_solution
