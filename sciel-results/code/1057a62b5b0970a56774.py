"""Rewritten program for difference triangle set (4,7)."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Best known solution: scope 134 for (4,7)
    # We'll try to find a solution with minimal scope

    def row_diffs(row):
        """Compute all positive differences in a row."""
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))

    def scope_of(rows):
        """Return the maximum entry across all rows."""
        return max(max(r) for r in rows)

    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            for d in row_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    def backtrack(rows, used_diffs, scope, n_rows, k):
        """
        Backtracking search to build rows one by one.
        rows: list of completed rows
        used_diffs: set of all differences used so far
        scope: maximum allowed entry
        Returns a valid list of rows or None.
        """
        if len(rows) == n_rows:
            return rows[:]
        
        # Build the next row
        row = [0]
        remaining_slots = k  # need k more entries
        
        def build_row(current_row, last_val, remaining, used, rows_so_far):
            if remaining == 0:
                # Check all diffs in this row against used
                rd = row_diffs(current_row)
                if rd & used:
                    return None
                return rows_so_far + [current_row], used | rd
            
            if time.time() > deadline:
                return None
            
            # Try next values from last_val+1 to scope
            for x in range(last_val + 1, scope + 1):
                new_row = current_row + [x]
                # Quick check: new diffs with existing elements
                new_diffs = set(x - a for a in current_row)
                if new_diffs & used:
                    continue
                # Also check if any diff within new_row conflicts (partial)
                # We'll do full check at the end, but prune early
                # Check partial: all pairs in new_row
                partial_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        d = new_row[j] - new_row[i]
                        if d in used or d in partial_diffs:
                            break
                        partial_diffs.add(d)
                    else:
                        continue
                    break
                else:
                    # All partial diffs are new
                    result = build_row(new_row, x, remaining - 1, used | partial_diffs, rows_so_far)
                    if result is not None:
                        return result
            return None
        
        result = build_row(row, 0, remaining_slots, used_diffs, rows)
        return result

    def greedy_construct(n_rows, k, scope):
        """Greedy construction with a fixed scope limit."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                for x in range(row[-1] + 1, scope + 1):
                    new_diffs = set(x - a for a in row)
                    if not (new_diffs & used):
                        # Check all pairwise diffs in the new row
                        full_diffs = set()
                        for i in range(len(row) + 1):
                            for j in range(i + 1, len(row) + 1):
                                if i == len(row) or j == len(row):
                                    d = (row + [x])[j] - (row + [x])[i]
                                else:
                                    d = row[j] - row[i]
                                if d in full_diffs:
                                    break
                                full_diffs.add(d)
                            else:
                                continue
                            break
                        else:
                            if not (full_diffs & used):
                                row.append(x)
                                used |= full_diffs
                                found = True
                                break
                if not found:
                    return None
            rows.append(row)
        return rows

    def local_search(rows, n_rows, k, max_iterations=5000):
        """Local search to reduce scope."""
        if rows is None:
            return None
        
        current_scope = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try to reduce one entry
            improved = False
            for ri in range(n_rows):
                for pos in range(k, -1, -1):  # from last to first
                    if pos == 0:
                        continue
                    old_val = best_rows[ri][pos]
                    # Try to reduce
                    for new_val in range(old_val - 1, best_rows[ri][pos - 1], -1):
                        candidate = best_rows[:]
                        candidate[ri] = best_rows[ri][:]
                        candidate[ri][pos] = new_val
                        # Check strictly increasing
                        if any(candidate[ri][j] >= candidate[ri][j + 1] for j in range(len(candidate[ri]) - 1)):
                            continue
                        if is_valid(candidate):
                            new_scope = scope_of(candidate)
                            if new_scope < best_scope:
                                best_rows = candidate
                                best_scope = new_scope
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try random perturbation
                ri = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                delta = random.randint(1, 3)
                candidate = best_rows[:]
                candidate[ri] = best_rows[ri][:]
                candidate[ri][pos] += delta
                if candidate[ri][pos] <= best_scope + 10:
                    if is_valid(candidate):
                        new_scope = scope_of(candidate)
                        if new_scope <= best_scope:
                            best_rows = candidate
                            best_scope = new_scope
        
        return best_rows

    # Strategy: binary search on scope, use backtracking, then local search
    
    # First, get an initial solution with greedy (unbounded)
    initial_rows = None
    used_init = set()
    rows_init = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = set(x - a for a in row)
                if not (new_diffs & used_init):
                    # Full check
                    full = row + [x]
                    fd = set(full[j] - full[i] for i in range(len(full)) for j in range(i + 1, len(full)))
                    if not (fd & used_init):
                        row.append(x)
                        used_init |= fd
                        break
                x += 1
        rows_init.append(row)
    initial_rows = rows_init
    initial_scope = scope_of(initial_rows)
    
    best_rows = initial_rows
    best_scope = initial_scope
    
    # Try to improve with backtracking + binary search on scope
    lo = 0
    hi = initial_scope
    
    # Binary search for minimal feasible scope
    # But backtracking may be slow, so we limit it
    for scope_attempt in range(hi, lo, -1):
        if time.time() > deadline - 20:
            break
        
        # Try backtracking with this scope
        result = backtrack([], set(), scope_attempt, n_rows, k)
        if result is not None:
            sc = scope_of(result)
            if sc < best_scope:
                best_rows = result
                best_scope = sc
            break
    
    # Local search polish
    if time.time() < deadline - 5:
        polished = local_search(best_rows, n_rows, k, max_iterations=10000)
        if polished is not None:
            ps = scope_of(polished)
            if ps < best_scope:
                best_rows = polished
                best_scope = ps
    
    # Final validation
    if not is_valid(best_rows):
        # Fallback to initial
        best_rows = initial_rows
    
    return best_rows
# EVOLVE-BLOCK-END
