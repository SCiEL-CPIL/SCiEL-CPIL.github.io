"""Difference triangle set (4,7) solver with time-bounded local search."""
import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill (and 150s wall-clock)
    
    # Total number of positive differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # We need to find n_rows rows, each with k+1 entries starting at 0, strictly increasing,
    # such that all positive differences within rows are distinct across all rows.
    # Score = max entry over all rows. We want to minimize this.
    
    # Strategy: iterative local search / simulated annealing style
    # 1. Start with a greedy construction to get an initial valid solution
    # 2. Try to reduce the scope (max entry) by swapping/adjusting entries
    # 3. Use random restarts with different greedy orderings
    
    best_solution = None
    best_scope = float('inf')
    
    def greedy_construct(seed_rng, max_scope=None):
        """Greedy construction with randomized tie-breaking."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if max_scope is not None and x > max_scope:
                    return None
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        scope = max(max(r) for r in rows)
        if max_scope is not None and scope > max_scope:
            return None
        return rows
    
    def is_valid(rows):
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
    
    # Phase 1: Get an initial solution via greedy
    initial = greedy_construct(random.Random(42))
    if initial is not None:
        best_solution = initial
        best_scope = get_scope(initial)
    
    # Phase 2: Try to improve by reducing scope
    # Local search: pick a row, try to reassign some entries to reduce max
    def local_search(rows, rng, deadline):
        """Try to reduce the scope of a valid solution."""
        rows = [list(r) for r in rows]
        current_scope = get_scope(rows)
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try each row
            for ri in range(n_rows):
                if time.time() > deadline:
                    break
                row = rows[ri]
                # Try to move the largest entry down
                # Or try swapping entries
                # Simple approach: try to rebuild one row with a lower upper bound
                max_allowed = current_scope - 1
                # Remove this row's differences from used
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])
                for d in row_diffs:
                    used.discard(d)
                
                # Try greedy rebuild with max_allowed
                new_row = [0]
                x = 0
                while len(new_row) < k + 1:
                    x += 1
                    if x > max_allowed:
                        break
                    new = {x - a for a in new_row}
                    if not (new & used):
                        new_row.append(x)
                        used |= new
                if len(new_row) == k + 1:
                    new_scope = max(new_row)
                    if new_scope < max(r[-1] for r in rows):
                        rows[ri] = new_row
                        current_scope = get_scope(rows)
                        improved = True
                        break
                else:
                    # Re-add the old diffs
                    for d in row_diffs:
                        used.add(d)
        
        return rows, current_scope
    
    # Phase 3: Randomized restarts with different strategies
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Different greedy strategies
        if attempt % 4 == 0:
            # Standard greedy
            rng = random.Random(attempt * 7 + 1)
            sol = greedy_construct(rng, max_scope=best_scope - 1)
            if sol is not None:
                scope = get_scope(sol)
                if scope < best_scope:
                    best_solution = sol
                    best_scope = scope
        elif attempt % 4 == 1:
            # Greedy with random row order
            rng = random.Random(attempt * 13 + 3)
            used = set()
            rows = [None] * n_rows
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            success = True
            for ri in row_order:
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    if x > best_scope - 1:
                        success = False
                        break
                    new = {x - a for a in row}
                    if not (new & used):
                        row.append(x)
                        used |= new
                if not success:
                    break
                rows[ri] = row
            if success and all(r is not None for r in rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_solution = rows
                    best_scope = scope
        elif attempt % 4 == 2:
            # Local search from current best
            if best_solution is not None:
                rng = random.Random(attempt * 17 + 5)
                rows, scope = local_search(best_solution, rng, deadline)
                if scope < best_scope:
                    best_solution = rows
                    best_scope = scope
        else:
            # Simulated annealing style: perturb and repair
            if best_solution is not None:
                rng = random.Random(attempt * 19 + 7)
                rows = [list(r) for r in best_solution]
                # Perturb: change one entry in a random row
                ri = rng.randint(0, n_rows - 1)
                pos = rng.randint(1, len(rows[ri]) - 1)
                old_val = rows[ri][pos]
                # Try nearby values
                for delta in rng.sample(range(-5, 6), 10):
                    new_val = old_val + delta
                    if new_val <= 0 or new_val >= rows[ri][pos + 1] if pos + 1 < len(rows[ri]) else new_val >= rows[ri][pos]:
                        continue
                    # Check increasing
                    if pos > 0 and new_val <= rows[ri][pos - 1]:
                        continue
                    if pos + 1 < len(rows[ri]) and new_val >= rows[ri][pos + 1]:
                        continue
                    # Check differences
                    row = rows[ri]
                    old_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            old_diffs.add(row[j] - row[i])
                    row[pos] = new_val
                    new_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            new_diffs.add(row[j] - row[i])
                    # Check against other rows
                    conflict = False
                    for oi in range(n_rows):
                        if oi == ri:
                            continue
                        orow = rows[oi]
                        odiffs = set()
                        for i in range(len(orow)):
                            for j in range(i + 1, len(orow)):
                                odiffs.add(orow[j] - orow[i])
                        if new_diffs & odiffs:
                            conflict = True
                            break
                    if not conflict:
                        scope = get_scope(rows)
                        if scope < best_scope:
                            best_solution = [list(r) for r in rows]
                            best_scope = scope
                        break
                    else:
                        row[pos] = old_val
    
    # Final check
    if best_solution is not None:
        if not is_valid(best_solution):
            # Fallback to initial greedy
            best_solution = greedy_construct(random.Random(0))
    
    return best_solution
