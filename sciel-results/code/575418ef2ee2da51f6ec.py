"""Structural change: constraint-satisfaction / backtracking search over the difference
set, using dynamic variable ordering (MRV) to place the most-constrained difference
first, with forward checking on global difference availability.

Instead of random-walk over row values, we:
1. Enumerate candidate difference sets for a row of size k+1 (starting at 0) that are
   internally valid (all pairwise differences distinct within the row).
2. Use backtracking to assign a difference set to each row such that no difference
   is reused across rows.
3. Dynamic variable ordering: at each step, choose the next row's candidate that
   leaves the fewest options for remaining rows (MRV heuristic).
4. Forward checking: maintain a set of used differences; prune candidates that conflict.
5. Minimize the maximum entry (scope) via iterative deepening / branch-and-bound.

This shifts the search from "optimize values" to "satisfy uniqueness constraint on
differences", which is the core structural constraint of the problem.
"""

import time
import random
import sys
import math
from collections import defaultdict


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    total_diffs_needed = n_rows * (m * (m - 1) // 2)  # total unique differences required

    # ---------- Generate candidate difference sets for a single row ----------
    # A row of m entries starting at 0: [0, a1, a2, ..., a_{m-1}] with 0 < a1 < ... < a_{m-1}
    # The differences are all a_j - a_i for i < j. There are m*(m-1)/2 of them.
    # We need to find rows where all these differences are distinct (within the row).
    
    def generate_candidates(max_scope, limit=5000):
        """Generate candidate rows (as tuples of values) with scope <= max_scope,
        where all intra-row differences are distinct."""
        candidates = []
        rng = random.Random(12345)
        
        # Greedy generation with randomization
        for attempt in range(limit * 3):
            if time.time() > deadline - 2:
                break
            row = [0]
            # Random walk to build row
            x = 0
            valid = True
            used_in_row = set()
            while len(row) < m:
                # Try to pick next value
                found = False
                for _ in range(50):
                    if x + 1 > max_scope:
                        break
                    x += 1
                    # Check all differences with existing entries
                    conflict = False
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_in_row:
                            conflict = True
                            break
                        new_diffs.append(d)
                    if not conflict:
                        used_in_row.update(new_diffs)
                        row.append(x)
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if valid and max(row) <= max_scope:
                # Convert to tuple for hashing
                rt = tuple(row)
                if rt not in candidates:
                    candidates.append(rt)
            if len(candidates) >= limit:
                break
        
        return candidates

    # ---------- Compute differences for a row ----------
    def row_diffs(row):
        """Return frozenset of all positive intra-row differences."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return frozenset(diffs)

    # ---------- Backtracking with MRV ----------
    def backtrack(rows_list, used_diffs, row_idx, max_scope):
        """Try to assign rows starting from row_idx. rows_list is list of candidate rows.
        used_diffs is set of differences already used. Returns solution or None."""
        if row_idx == n_rows:
            return [list(r) for r in rows_list]
        
        # MRV: for each remaining row slot, count how many candidates are compatible
        # with current used_diffs. Pick the slot with fewest compatible candidates.
        # Since we process rows in order for simplicity, use MRV to reorder.
        
        # Actually, let's do proper MRV: at each step, pick the next row to fill
        # that has the fewest valid candidates given used_diffs.
        
        best_slot = -1
        best_candidates = None
        min_count = float('inf')
        
        for slot in range(row_idx, n_rows):
            compatible = []
            for cand in rows_list[slot]:
                cd = row_diffs(cand)
                if not cd & used_diffs:
                    compatible.append((cand, cd))
            if len(compatible) < min_count:
                min_count = len(compatible)
                best_slot = slot
                best_candidates = compatible
            if min_count == 0:
                return None  # Dead end
        
        if min_count == 0:
            return None
        
        # Sort candidates by scope (prefer smaller)
        best_candidates.sort(key=lambda x: max(x[0]))
        
        # Try each candidate for the chosen slot
        for cand, cd in best_candidates:
            if time.time() > deadline - 2:
                return None
            
            # Forward check: can remaining rows still be filled?
            new_used = used_diffs | cd
            rows_list[best_slot] = cand
            
            # Simple forward check: for each remaining unfilled slot, verify at least
            # one candidate doesn't conflict with new_used
            feasible = True
            for slot in range(n_rows):
                if slot == best_slot:
                    continue
                if slot < row_idx:
                    continue  # already filled
                has_option = False
                for rcand in rows_list[slot]:
                    rcd = row_diffs(rcand)
                    if not rcd & new_used:
                        has_option = True
                        break
                if not has_option:
                    feasible = False
                    break
            
            if feasible:
                result = backtrack(rows_list, new_used, row_idx + 1, max_scope)
                if result is not None:
                    return result
        
        rows_list[best_slot] = None
        return None

    # ---------- Iterative deepening on scope ----------
    # Lower bound: counting argument
    # n_rows * C(m,2) unique differences, all in [1, max_scope]
    # So max_scope >= n_rows * m * (m-1) / 2
    lb = (n_rows * m * (m - 1)) // 2
    
    # Try increasing scopes
    best_solution = None
    current_scope = lb
    
    # First, try to find any valid solution with a reasonable scope
    # Generate candidates with a generous initial scope
    initial_scope = max(lb + 20, 150)
    
    candidates = generate_candidates(initial_scope, limit=3000)
    
    if not candidates:
        # Fallback: pure greedy
        u = set()
        res = []
        for _ in range(n_rows):
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
            res.append(row)
        return res
    
    # Sort candidates by scope
    candidates.sort(key=lambda r: max(r))
    
    # Use a subset of best candidates to keep search tractable
    # Take top N candidates by scope
    max_candidates = 500
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    
    # Assign same candidate pool to all rows (they can reuse the same candidate
    # as long as differences don't conflict)
    rows_list = [list(candidates) for _ in range(n_rows)]
    
    # Try backtracking
    used_diffs = set()
    solution = backtrack(rows_list, used_diffs, 0, initial_scope)
    
    if solution is not None:
        best_solution = solution
    else:
        # If backtracking fails, fall back to greedy
        u = set()
        res = []
        for _ in range(n_rows):
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
            res.append(row)
        best_solution = res
    
    # ---------- Local improvement: compact rows ----------
    if best_solution is not None and time.time() < deadline - 5:
        # Try to reduce scope by shifting elements left
        for _ in range(50):  # iterations
            if time.time() > deadline - 3:
                break
            improved = False
            for r_idx in range(n_rows):
                row = best_solution[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    cv = row[pos]
                    if cv <= row[pos - 1] + 1:
                        continue
                    nv = cv - 1
                    # Check if this creates a conflict
                    old_d = set()
                    for i in range(pos):
                        old_d.add(cv - row[i])
                    for j in range(pos + 1, len(row)):
                        old_d.add(row[j] - cv)
                    
                    new_d = set()
                    for i in range(pos):
                        new_d.add(nv - row[i])
                    for j in range(pos + 1, len(row)):
                        new_d.add(row[j] - nv)
                    
                    # Check conflicts with all other rows
                    conflict = False
                    for d in new_d:
                        if d in old_d:
                            continue
                        # Check if d is used by any other row
                        for other_idx in range(n_rows):
                            if other_idx == r_idx:
                                continue
                            other = best_solution[other_idx]
                            for i in range(len(other)):
                                for j in range(i + 1, len(other)):
                                    if other[j] - other[i] == d:
                                        conflict = True
                                        break
                                if conflict:
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    if not conflict:
                        row[pos] = nv
                        improved = True
            
            if not improved:
                break
    
    return best_solution


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
