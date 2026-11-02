"""Difference triangle set (4,7) – CSP backtracking with MRV + local search."""
import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _all_diffs_valid(rows):
    """Check all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True


def _scope(rows):
    """The largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k):
    """Simple greedy construction: build rows one by one picking smallest available."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _csp_backtrack(n_rows, k, max_scope, deadline, rng):
    """
    Backtracking search over difference slots.
    We assign differences to positions in a row, using MRV heuristic.
    The key insight: each row has C(k+1,2) differences. We need to partition
    the range [1, max_scope] into n_rows disjoint sets of size C(k+1,2),
    where each set corresponds to the differences of a valid row.
    
    Alternative approach: build rows by selecting which differences to use.
    For a row [0, a1, a2, ..., ak], the differences are:
    a1, a2, ..., ak, a2-a1, a3-a1, ..., ak-a1, a3-a2, ..., ak-a(k-1)
    
    We'll use a different CSP: assign values to row entries one at a time,
    but use MRV to decide which entry to place next based on how many
    available differences remain compatible.
    """
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    if total_diffs_needed > max_scope:
        return None
    
    # We'll build rows by placing entries. For each row, we need k+1 entries
    # starting at 0. The differences must all be distinct across all rows.
    
    # State: used_diffs (set), rows (list of partially built rows)
    # Variable: which row to extend next, and what value to add
    
    # For efficiency, we'll use a recursive approach with memoization-like pruning.
    
    best = [None]
    
    def backtrack(rows, used_diffs, row_idx, deadline):
        if time.time() > deadline:
            return False
        if row_idx == n_rows:
            best[0] = [list(r) for r in rows]
            return True
        
        # MRV: pick the row with the fewest available extensions
        # For simplicity, we process rows in order but try values in smart order
        
        row = rows[row_idx]
        current_max = row[-1] if row else 0
        
        # Try values from current_max+1 upward, but prioritize values that
        # use smaller differences (to keep scope small)
        # Actually, let's try to be more strategic: try values that create
        # differences not yet used, and prefer smaller differences
        
        # Generate candidate values
        candidates = []
        for x in range(current_max + 1, max_scope + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                candidates.append((x, new_diffs))
        
        # Sort by the maximum difference (prefer smaller scopes)
        # But also consider: we want to leave room for future rows
        # Heuristic: prefer values that use "common" small differences
        candidates.sort(key=lambda t: max(t[1]) if t[1] else 0)
        
        # Limit candidates to avoid explosion
        max_candidates = min(len(candidates), 50)
        
        for x, new_diffs in candidates[:max_candidates]:
            row.append(x)
            used_diffs |= new_diffs
            
            if backtrack(rows, used_diffs, row_idx + 1, deadline):
                return True
            
            row.pop()
            used_diffs -= new_diffs
        
        return False
    
    # Start with all rows containing just [0]
    initial_rows = [[0] for _ in range(n_rows)]
    if backtrack(initial_rows, set(), 0, deadline):
        return best[0]
    
    return None


def _csp_with_ordering(n_rows, k, max_scope, deadline, rng):
    """
    CSP backtracking with dynamic row ordering.
    Instead of fixed row order, we pick which row to extend next based on MRV.
    """
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    if total_diffs_needed > max_scope:
        return None
    
    best = [None]
    
    def count_available(row, used_diffs, max_scope):
        """Count how many valid extensions exist for this row."""
        current_max = row[-1] if row else 0
        count = 0
        for x in range(current_max + 1, max_scope + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                count += 1
                if count >= 10:  # Early exit
                    break
        return count
    
    def backtrack(rows, used_diffs, deadline):
        if time.time() > deadline:
            return False
        
        # Check if all rows are complete
        incomplete = [i for i in range(n_rows) if len(rows[i]) < k + 1]
        if not incomplete:
            best[0] = [list(r) for r in rows]
            return True
        
        # MRV: pick the incomplete row with fewest available extensions
        row_counts = []
        for i in incomplete:
            c = count_available(rows[i], used_diffs, max_scope)
            row_counts.append((c, i))
        
        row_counts.sort()
        
        # If any row has 0 available extensions, backtrack
        if row_counts[0][0] == 0:
            return False
        
        # Try the most constrained row first
        _, chosen_row = row_counts[0]
        row = rows[chosen_row]
        current_max = row[-1] if row else 0
        
        # Generate and sort candidates
        candidates = []
        for x in range(current_max + 1, max_scope + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                candidates.append((x, new_diffs))
        
        candidates.sort(key=lambda t: max(t[1]) if t[1] else 0)
        
        max_candidates = min(len(candidates), 30)
        
        for x, new_diffs in candidates[:max_candidates]:
            row.append(x)
            used_diffs |= new_diffs
            
            if backtrack(rows, used_diffs, deadline):
                return True
            
            row.pop()
            used_diffs -= new_diffs
        
        return False
    
    initial_rows = [[0] for _ in range(n_rows)]
    if backtrack(initial_rows, set(), deadline):
        return best[0]
    
    return None


def _local_search(rows, k, deadline):
    """Local search to reduce scope by moving entries down."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    n_rows = len(rows)
    rng = random.Random(42)
    
    def check_and_get_scope(candidate_rows):
        seen = set()
        scope = 0
        for row in candidate_rows:
            for i in range(len(row)):
                if row[i] > scope:
                    scope = row[i]
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in seen:
                        return False, scope
                    seen.add(d)
        return True, scope
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        
        ri = rng.randrange(n_rows)
        row = list(best_rows[ri])
        k1 = len(row)
        improved = False
        
        # Try to reduce each entry (from right to left)
        for pos in range(k1 - 1, 0, -1):
            orig = row[pos]
            min_val = row[pos - 1] + 1
            # Try values from min_val to orig-1
            for candidate in range(min_val, orig):
                row[pos] = candidate
                candidate_rows = [best_rows[i] if i != ri else row for i in range(n_rows)]
                valid, scope = check_and_get_scope(candidate_rows)
                if valid and scope < best_scope:
                    best_rows[ri] = row
                    best_scope = scope
                    improved = True
                    break
            if improved:
                break
            row[pos] = orig
        
        if not improved:
            # Try rebuilding a row
            if iters % 50 == 0:
                used = set()
                for i in range(n_rows):
                    if i != ri:
                        used |= _row_diffs(best_rows[i])
                
                # Try to build a new row with smaller scope
                new_row = [0]
                x = 0
                success = False
                while len(new_row) < k1:
                    x += 1
                    if x > best_scope:
                        break
                    new_diffs = {x - a for a in new_row}
                    if not new_diffs & used:
                        new_row.append(x)
                
                if len(new_row) == k1:
                    new_rows = [best_rows[i] if i != ri else new_row for i in range(n_rows)]
                    valid, scope = check_and_get_scope(new_rows)
                    if valid and scope < best_scope:
                        best_rows[ri] = new_row
                        best_scope = scope
    
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin under 150s limit
    
    # Phase 1: Greedy construction for initial solution
    rows = _greedy_construct(n_rows, k)
    best_scope = _scope(rows)
    
    # Phase 2: Try CSP backtracking with decreasing max_scope
    # Start from the greedy scope and try to find solutions with smaller scope
    rng = random.Random(42)
    
    # First, try to find any valid solution with scope < best_scope
    for target_scope in range(best_scope - 1, max(100, best_scope - 50), -1):
        if time.time() > deadline - 30:
            break
        
        # Try CSP with this max_scope
        result = _csp_with_ordering(n_rows, k, target_scope, deadline - 25, rng)
        if result is not None:
            new_scope = _scope(result)
            if new_scope < best_scope:
                rows = result
                best_scope = new_scope
            break
    
    # Phase 3: Local search to further reduce scope
    if time.time() < deadline - 10:
        rows, ls_scope = _local_search(rows, k, deadline - 5)
        if ls_scope < best_scope:
            best_scope = ls_scope
    
    return rows

# EVOLVE-BLOCK-END
