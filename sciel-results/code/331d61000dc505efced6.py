"""
Simulated Annealing constructor for the difference-triangle-set task.

Strategy: 
1. Initialize with a greedy construction to get a valid starting point.
2. Use Simulated Annealing to minimize the "scope" (max entry).
3. Moves include:
   - Rebuilding a single row using greedy logic but with a "budget" (max allowed value).
   - Shifting a single element in a row to a new valid position (if possible).
   - Swapping rows (to break symmetry if needed, though rows are distinguishable by content).
4. The cost function is the scope (max entry). If a move creates a conflict (duplicate differences), 
   it is rejected unless we are in a "repair" phase, but simpler is to only allow moves that 
   maintain validity. To explore, we can try to *reduce* the max entry of a row.

Actually, a more effective move for this specific problem is "row compaction".
We start with a valid set. We try to reduce the largest element of the most "expensive" row.
We can do this by:
- Picking a row.
- Trying to reduce its largest element to a smaller value that doesn't conflict with existing differences.
- If that fails, try reducing the second largest, etc.
- This is a form of local search / greedy compaction driven by SA temperature to accept worse (but valid) 
  intermediate states if they lead to better global compaction? 
  Actually, if we only accept valid states, the cost is monotonic decreasing or equal. 
  SA is useful if we allow *invalid* states and use a penalty. 
  Let's stick to valid-state moves but with randomization in *how* we compact.

Alternative: 
- Start with greedy.
- Loop:
  - Pick a random row.
  - Rebuild it from scratch, but with a random "bias" or different starting point, 
    ensuring it fits within the current global scope.
  - If the new row fits, accept it.
  - If we can reduce the global scope (by the new row being smaller than the previous max), great.
  - If we can't reduce scope, we might still accept it if it opens up possibilities for other rows?
    (Hard to know).
    
Let's try a "Randomized Compaction" approach:
1. Get initial greedy solution.
2. While time remains:
   a. Pick a row at random.
   b. Attempt to compact it: try to find a smaller set of differences for this row 
      that is disjoint from all differences used by other rows.
      This is a Subgraph Isomorphism / Set Packing problem. 
      We can solve it approximately by greedy backtracking with a limit.
   c. If successful and the new row has a smaller max entry than the old one, update.
   d. If successful and same max entry, update (to diversify).
   e. If we find a row that can be made significantly smaller, it might allow other rows to be smaller too.
      So, after changing a row, we can try to compact *other* rows.

This is essentially "Iterated Local Search" with a powerful local move (row compaction).
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

    deadline = time.time() + 115.0
    m = k + 1  # entries per row

    # Helper: Check if a row is valid given a set of used differences
    # Returns True if valid, and the set of differences used by this row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper: Build a row greedily, avoiding a set of forbidden differences.
    # start: first element is 0.
    # max_val: we must keep all elements <= max_val.
    # Returns None if impossible, else the row.
    def build_row_greedy(forbidden, max_val, rng=None, bias=0):
        row = [0]
        x = 0
        # We need to choose m-1 more elements.
        # To allow some randomization, we can shuffle the order of trying x?
        # No, x must increase. But we can skip some x if rng says so.
        while len(row) < m:
            x += 1
            if x > max_val:
                return None
            # Check if x can be added
            ok = True
            for a in row:
                d = x - a
                if d in forbidden:
                    ok = False
                    break
            if ok:
                # Accept with some probability to allow randomization?
                # If we accept too many, we might hit max_val.
                # If we reject, we skip.
                # Let's just accept the first valid one for speed, 
                # but we can vary the "forbidden" set or the start.
                row.append(x)
            # To make it random, we could skip valid x with small prob?
            # That's dangerous for finding a solution.
            # Better: randomize the *order* of rows or the *starting offset*?
            # The problem is symmetric in rows, but not in values.
            # Let's stick to deterministic greedy for the move, but randomize *which* row we try to compact.
            
        return row

    # Helper: Try to compact a specific row index.
    # We want to find a row of length m, starting with 0, with max entry <= current_max,
    # such that its diffs are disjoint from all_other_diffs.
    # We try to minimize the max entry.
    def try_compact_row(row_idx, all_rows, all_diffs, target_max, rng):
        # Remove current row's diffs from forbidden set
        curr_diffs = get_row_diffs(all_rows[row_idx])
        forbidden = set(all_diffs)
        forbidden.difference_update(curr_diffs)
        
        # Try to build a new row with max <= target_max
        # We can try different "strategies" for building:
        # 1. Standard greedy (smallest possible)
        # 2. Greedy with random skips (to find different configurations)
        
        # Strategy 1: Standard Greedy
        new_row = build_row_greedy(forbidden, target_max)
        if new_row is not None:
            return new_row
            
        # Strategy 2: Randomized Greedy (skip some candidates)
        # This might find a valid row if the standard greedy got stuck due to 
        # "greedy trap" (though greedy is usually good for this).
        # Actually, if standard greedy fails, it means no row exists with max <= target_max 
        # that is a *subset* of the greedy path? No, greedy finds *a* solution if one exists 
        # in the "tight" sense? Not necessarily. 
        # Let's try a backtracking search with a small depth for the last few elements?
        # Or just return None.
        
        return None

    # Initial greedy construction
    def initial_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            rows.append(row)
        return rows, used

    rows, used = initial_greedy()
    current_scope = max(max(r) for r in rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    rng = random.Random(42)

    # Main Search Loop: Iterated Local Search / Randomized Compaction
    while time.time() < deadline:
        # Pick a random row to try to improve
        idx = rng.randint(0, n_rows - 1)
        
        # Target: try to make this row smaller than its current max
        # Or just smaller than current_scope
        # Let's try to compact it to as small as possible, but not exceeding current_scope
        # Actually, if we compact it below current_scope, we might lower the global scope.
        
        old_row = rows[idx]
        old_max = max(old_row)
        
        # Try to find a new row with max < old_max
        # We can binary search or just try old_max - 1, old_max - 2, etc.
        # Or just try to build with max = old_max - 1. If success, great.
        # If we can go even lower, we'll catch it in subsequent iterations.
        
        target_max = old_max - 1
        if target_max < m - 1: # Minimum possible max is m-1 (0..m-1)
            continue
            
        new_row = try_compact_row(idx, rows, used, target_max, rng)
        
        if new_row is not None:
            # Accept the new row
            # Update used set
            old_diffs = get_row_diffs(old_row)
            new_diffs = get_row_diffs(new_row)
            
            # Remove old diffs, add new diffs
            # Note: We must ensure new_diffs doesn't conflict with other rows.
            # try_compact_row already ensured this by using 'forbidden' set.
            
            used.difference_update(old_diffs)
            used.update(new_diffs)
            
            rows[idx] = new_row
            current_scope = max(max(r) for r in rows)
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
                
        # Occasionally, try to compact a row to the *current global scope* 
        # even if it's not smaller than its own max, just to diversify?
        # Or try to compact the row with the largest max entry specifically.
        if rng.random() < 0.3:
            # Find row with max entry
            max_idx = max(range(n_rows), key=lambda i: max(rows[i]))
            max_val = max(rows[max_idx])
            if max_val > m - 1:
                # Try to reduce it
                target_max = max_val - 1
                new_row = try_compact_row(max_idx, rows, used, target_max, rng)
                if new_row is not None:
                    old_diffs = get_row_diffs(rows[max_idx])
                    new_diffs = get_row_diffs(new_row)
                    used.difference_update(old_diffs)
                    used.update(new_diffs)
                    rows[max_idx] = new_row
                    current_scope = max(max(r) for r in rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]

    return best_rows


if __name__ == "__main__":
    # Self-check: build the (4,7) instance and validate.
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
