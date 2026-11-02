"""Difference Triangle Set (4,7) solver.

Strategy:
1. Fast greedy baseline to establish an upper bound.
2. Binary search on the maximum entry M.
3. For each M, use a backtracking solver with symmetry breaking
   (rows are ordered lexicographically) and forward checking.
4. This structural change (optimization -> decision + binary search)
   allows for more rigorous pruning than pure local search.
"""
import time
import random
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction to get a quick upper bound."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


class BacktrackSolver:
    def __init__(self, n_rows, k, max_val, time_limit):
        self.n_rows = n_rows
        self.k = k
        self.L = k + 1
        self.max_val = max_val
        self.time_limit = time_limit
        self.start_time = time.time()
        self.used_diffs = set()
        self.rows = [[] for _ in range(n_rows)]
        self.node_count = 0
        self.node_limit = 1000000 # Safety valve
        self.found = False
        self.solution = None

    def _row_leq(self, row_a, row_b):
        """Check if row_a <= row_b lexicographically."""
        if len(row_a) != len(row_b):
            return len(row_a) < len(row_b)
        for i in range(len(row_a)):
            if row_a[i] < row_b[i]:
                return True
            if row_a[i] > row_b[i]:
                return False
        return True

    def _solve(self, r_idx, pos):
        """Recursive backtracking."""
        if self.found or self.node_count > self.node_limit:
            return
        
        if time.time() - self.start_time > self.time_limit:
            return

        self.node_count += 1

        if r_idx == self.n_rows:
            self.found = True
            self.solution = [list(row) for row in self.rows]
            return

        # If we are starting a new row, we must ensure it's >= previous row
        # But since we fill row by row, and rows are symmetric, we enforce
        # that row r is lexicographically >= row r-1.
        
        if pos == 0:
            # Start of a new row
            if r_idx > 0:
                # The new row must be lexicographically >= previous row
                # Since first element is always 0, we check the rest later
                # For now, just set 0
                pass
            
            self.rows[r_idx].append(0)
            self._solve(r_idx, 1)
            self.rows[r_idx].pop()
            return

        # Fill position pos in row r_idx
        prev_val = self.rows[r_idx][pos - 1]
        start_val = prev_val + 1
        
        # If this is the first element after 0, and r_idx > 0,
        # we need to be careful with symmetry.
        # The lexicographic constraint applies to the full row.
        # We can prune if the current prefix is already > previous row's prefix?
        # No, we need the full row to compare.
        # However, if we are at pos 1, and prev_row[1] is known,
        # we can restrict the range.
        
        # Optimization: If r_idx > 0, the new row must be >= rows[r_idx-1]
        # Let prev_row = self.rows[r_idx-1]
        # If pos == 1:
        #   If prev_row[1] > start_val, we can start from prev_row[1]?
        #   No, lexicographic order:
        #   (0, a, b...) >= (0, c, d...)
        #   If a < c, then (0, a...) < (0, c...) regardless of rest.
        #   So if we pick a < c, the row will be smaller.
        #   We WANT row r >= row r-1.
        #   So if we pick a < c, it's invalid?
        #   Wait. We want to avoid duplicates.
        #   If we generate (0, 2, ...) and later (0, 1, ...), they are distinct.
        #   To break symmetry, we impose an order.
        #   Let's impose: rows are sorted lexicographically.
        #   So row r must be >= row r-1.
        
        if r_idx > 0:
            prev_row = self.rows[r_idx - 1]
            # Compare current prefix of self.rows[r_idx] with prev_row
            # Current prefix is self.rows[r_idx][:pos]
            # We are choosing val for pos.
            
            # If current prefix < prev_row prefix, then no matter what we pick,
            # the row will be < prev_row. So we should skip?
            # No, if prefix is smaller, the row is smaller.
            # We want row >= prev_row.
            # So if current prefix < prev_row prefix, we MUST pick a value that makes it >=?
            # Actually, if prefix is already smaller, the row is smaller.
            # So we can prune this branch entirely if prefix < prev_row prefix.
            
            # Let's check prefix equality/inequality
            # prefix = self.rows[r_idx][:pos]
            # prev_prefix = prev_row[:pos]
            
            # If prefix < prev_prefix:
            #   The row will be < prev_row. Invalid. Prune.
            # If prefix > prev_prefix:
            #   The row will be > prev_row. Valid so far.
            # If prefix == prev_prefix:
            #   We need val >= prev_row[pos].
            
            prefix = self.rows[r_idx][:pos]
            prev_prefix = prev_row[:pos]
            
            if prefix < prev_prefix:
                return # Prune
            
            if prefix == prev_prefix:
                start_val = max(start_val, prev_row[pos])
        
        for val in range(start_val, self.max_val + 1):
            # Check differences
            new_diffs = set()
            valid = True
            for i in range(pos):
                d = val - self.rows[r_idx][i]
                if d in self.used_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Apply
            self.rows[r_idx].append(val)
            self.used_diffs.update(new_diffs)
            
            self._solve(r_idx, pos + 1)
            
            # Undo
            self.rows[r_idx].pop()
            self.used_diffs.difference_update(new_diffs)
            
            if self.found:
                return


def _backtrack_solve(M, n_rows, k, time_limit):
    """Try to find a solution with max entry <= M."""
    solver = BacktrackSolver(n_rows, k, M, time_limit)
    solver._solve(0, 0)
    if solver.found:
        return solver.solution
    return None


def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 145 s
    time_limit = start_time + 145

    # 1) Get a quick upper bound using greedy
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    high = _max_entry(best_rows)
    
    # 2) Binary search for the minimum M
    # Lower bound: simple counting bound or 0
    # Number of diffs per row: k*(k+1)/2
    # Total diffs: n_rows * k*(k+1)/2
    # Max diff is M. So M must be at least total_diffs? No, diffs are distinct 1..M.
    # So M >= n_rows * k * (k+1) / 2
    low = max(0, (n_rows * k * (k + 1)) // 2)
    
    # Ensure high is feasible (greedy should be)
    # If greedy fails, we might need to expand high, but greedy usually works.
    
    # We will search in [low, high]
    # To save time, we don't binary search the whole range if low is close to high.
    # We can just try to improve from high downwards?
    # Binary search is safer.
    
    best_found = None
    best_max = high
    
    # We allocate time for the binary search steps.
    # Each step calls backtrack.
    # We need to limit the time per backtrack call.
    
    # Estimate time per call.
    # Let's try a few steps.
    
    # Optimization: If high - low is small, linear search might be faster.
    # But binary search is O(log(range)).
    
    # We will perform binary search, but with a time limit for each solver call.
    # If a call times out, we assume infeasible for that M? 
    # No, if it times out, it might be feasible but hard.
    # We should be conservative. If it times out, we might skip.
    
    # Better: Use the time limit for the WHOLE binary search.
    # We can't easily pass a global deadline to the solver if we want to reuse it.
    # Let's create a wrapper that checks time.
    
    # We'll do a manual binary search loop.
    
    # To avoid getting stuck on infeasible M, we can set a node limit in the solver.
    # The solver already has a node_limit.
    
    # Let's iterate.
    # We want to find the smallest M.
    
    # We can try to narrow down quickly.
    
    # Let's just run the binary search.
    # If time runs out, return best_found.
    
    while low < high:
        if time.time() > time_limit - 5:
            break
            
        mid = (low + high) // 2
        # Give the solver a portion of the remaining time
        # Or just rely on node limit?
        # Node limit is better for consistency.
        
        # Let's try to solve for mid
        res = _backtrack_solve(mid, n_rows, k, time_limit - time.time())
        
        if res is not None:
            # Feasible
            high = mid
            best_found = res
            best_max = mid
            # Update best_rows if this is better
            # Actually, any solution with max <= mid is valid.
            # We want the one with smallest max.
            # So we keep updating best_found.
        else:
            # Infeasible (or timed out)
            # If it timed out, it might be feasible.
            # But we have to make a decision.
            # Usually, if it's infeasible, it fails fast.
            # If it's feasible but hard, it might timeout.
            # Let's assume infeasible if it returns None.
            low = mid + 1

    # After binary search, high is the best known max.
    # But best_found might be from a larger mid if the last successful one was higher?
    # No, we update high = mid when successful.
    # So best_found corresponds to the current high (or a previous one if we didn't update? 
    # We update best_found = res every time we find a solution.
    # Since we search for smaller M, the last successful res has the smallest M.
    
    if best_found is not None:
        # Verify validity just in case
        if _check_valid(best_found, k):
            return best_found
    
    # Fallback to greedy
    if _check_valid(best_rows, k):
        return best_rows
    
    # If greedy is invalid (shouldn't happen), try to fix or return empty?
    # The task requires a valid set.
    # Greedy is always valid by construction.
    
    return best_rows
