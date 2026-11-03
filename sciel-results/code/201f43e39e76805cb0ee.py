import time
import sys
import random
import math
from collections import defaultdict, deque

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    m = k + 1  # entries per row
    rng = random.Random(42)

    # ---- Greedy initialization (fast, valid) ----
    def greedy_init(seed=42):
        rng_local = random.Random(seed)
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # We need to pick m-1 more numbers
            # Greedy: pick smallest x such that all (x - a) for a in row are unused
            # To speed up, we can skip some
            x = 0
            while len(row) < m:
                x += 1
                # Check if x is valid
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    for d in new_diffs:
                        used_diffs.add(d)
                    row.append(x)
            rows.append(row)
        return rows

    def compute_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        used = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(m):
                for j in range(i + 1, m):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # ---- Hopcroft-Karp for bipartite matching ----
    # Used to check if we can assign distinct differences to a new row given existing used diffs
    # Actually, the constraint is global. 
    # Let's use a different approach: Backtracking with MRV on the *rows* themselves?
    # No, the directive says "assign each of the N required difference slots to a unique integer d".
    
    # Let's stick to a strong Backtracking Search on the elements of the rows,
    # but use a powerful pruning: 
    # 1. We build rows one by one.
    # 2. For each row, we backtrack on the elements.
    # 3. We maintain a global set of used differences.
    # 4. We use a lower bound on the remaining scope to prune.
    
    # However, building rows one by one is often suboptimal because early rows might block later ones.
    # A better structure: Interleaved construction or full matrix backtracking.
    # Given time limits, let's try a hybrid:
    # 1. Get a good initial solution via Greedy + Local Search (fast).
    # 2. Use that as a bound.
    # 3. Run a randomized backtracking search that tries to beat the bound,
    #    using MRV (Minimum Remaining Values) to select which element to place next.
    
    # State: 
    # rows: list of lists. Some are complete, some partial.
    # We will fill the "slots" (r, i) in a specific order.
    # To allow flexibility, we can choose the next (r, i) dynamically (MRV).
    # But strict increasing order within a row is a hard constraint.
    
    # Let's implement a specific backtracking:
    # We build row by row. Within a row, we build left to right (a_0=0, a_1, ..., a_{m-1}).
    # This is simpler and respects the increasing constraint naturally.
    # The "MRV" idea applies to choosing which *row* to build next if we interleave, 
    # or which *candidate value* to try.
    
    # Let's try: Randomized Backtracking with Restarting and Pruning.
    
    best_rows = greedy_init(42)
    best_scope = compute_scope(best_rows)
    
    # Try a few greedy variants to get a tighter initial bound
    for s in range(10):
        r = greedy_init(s)
        sc = compute_scope(r)
        if sc < best_scope:
            best_scope = sc
            best_rows = r

    # Local search improvement on best_rows (simple swap/shift)
    def local_improve(rows, time_limit=5.0):
        start = time.time()
        rows = [list(r) for r in rows]
        used_diffs = set()
        for row in rows:
            for i in range(m):
                for j in range(i+1, m):
                    used_diffs.add(row[j] - row[i])
        
        scope = compute_scope(rows)
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for pos in range(m-1, 0, -1):
                    # Try to decrease row[pos] by 1
                    if row[pos] <= row[pos-1] + 1:
                        continue
                    old_val = row[pos]
                    new_val = old_val - 1
                    # Check if new_val creates duplicate diffs
                    # Remove old diffs involving pos
                    for a in range(m):
                        if a == pos: continue
                        d_old = abs(row[a] - old_val)
                        # We need to check if this diff is used elsewhere.
                        # This is expensive to check naively. 
                        # Optimization: Pre-compute diff counts? 
                        # For now, let's do a fast check:
                        # The diff d is valid if it's not in used_diffs OR if it was only used by this element.
                        # This requires counting.
                        pass
                    # This local search is too slow if implemented naively with sets.
                    # Let's skip complex local search and rely on the backtracking.
                    break
                break
        return rows

    # ---- Backtracking Search ----
    # We will try to construct rows one by one.
    # To avoid getting stuck in a bad order, we can randomize the order of rows?
    # No, rows are symmetric in the constraint (just a set of sets).
    # But the "increasing" constraint is local to the row.
    
    # Let's use a different strategy: 
    # **Iterative Deepening A* (IDA*)** or similar? 
    # No, let's use **Randomized Restarting Backtracking**.
    
    # We define a function to backtrack one row given a set of used diffs.
    def backtrack_row(row_idx, used_diffs, current_row, target_max):
        # current_row has len < m
        # We need to pick the next value x > current_row[-1]
        # such that all (x - a) for a in current_row are not in used_diffs
        # and x <= target_max (to ensure we can beat best_scope)
        
        if len(current_row) == m:
            return True
        
        last = current_row[-1]
        # Lower bound for next x: last + 1
        # Upper bound: target_max (if this is the last element) 
        # Actually, if we are not at the last element, we need to leave room for remaining elements.
        # Remaining elements: m - len(current_row) - 1
        # So x <= target_max - (m - len(current_row) - 1)
        
        rem = m - len(current_row) - 1
        max_x = target_max - rem
        
        # MRV: We don't have choices for "which element" since we go left-to-right.
        # But we can randomize the order of trying candidates?
        # Candidates: last+1 to max_x
        # To make it faster, we can jump? No, we need to check all.
        
        # Optimization: If the range is large, we might time out.
        # We can limit the search depth or branching factor.
        
        # Let's try candidates in a random order to escape local traps?
        # Or just standard DFS with pruning.
        
        # Pruning: If even the smallest possible future values (last+1, last+2...) 
        # combined with current used diffs lead to a conflict, prune.
        
        # Simple loop
        # To speed up, we can check feasibility quickly.
        
        candidates = range(last + 1, max_x + 1)
        # Randomize? 
        # Let's try a mix: 50% sequential, 50% random shuffle of a subset?
        # Sequential is usually better for finding small solutions.
        
        for x in candidates:
            # Check diffs
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                # Recurse
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack_row(row_idx, used_diffs, current_row, target_max):
                    return True
                
                # Undo
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
        
        return False

    def search_with_target(target_max, seed):
        rng_local = random.Random(seed)
        # We build rows one by one.
        # To improve diversity, we can randomize the "starting point" or order?
        # Since rows are independent except for used_diffs, the order of construction matters.
        # Let's try constructing rows in a random order? No, the "row" identity doesn't matter,
        # but the set of rows does.
        
        # Actually, we can just build row 0, then row 1, etc.
        # To add randomness, we can shuffle the *target* rows? No.
        # Let's just run multiple restarts with different random seeds for tie-breaking or candidate ordering.
        
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            current_row = [0]
            # Backtrack
            if not backtrack_row(r_idx, used_diffs, current_row, target_max):
                return None
            rows.append(current_row)
        
        return rows

    # Main Search Loop
    # Start from best_scope - 1 and go down? 
    # Or just try to find a solution with scope < best_scope.
    # We can use a "best first" approach or just iterative deepening on the target_max.
    
    # Let's try target_max from best_scope - 1 down to the theoretical lower bound.
    # Theoretical lower bound: 
    # Total diffs = n_rows * m * (m-1) / 2
    # They must be distinct integers in [1, target_max]
    # So target_max >= n_rows * m * (m-1) / 2
    
    total_diffs = n_rows * m * (m-1) // 2
    lower_bound = total_diffs
    
    # We know best_scope is a valid upper bound.
    # We want to find if there is a solution with scope < best_scope.
    
    # Strategy: 
    # 1. Try target_max = best_scope - 1
    # 2. If fail, try best_scope - 2, etc.
    # But this is slow if the gap is large.
    
    # Better: Use the backtracking search to *minimize* the scope directly?
    # We can add a heuristic to backtrack_row to prefer smaller x.
    # The current backtrack_row tries x from small to large. This naturally minimizes the row's max.
    # However, it might block future rows.
    
    # Let's run the search for a fixed time, trying to beat the current best.
    # We will use a "best effort" approach:
    # Run search_with_target(best_scope - 1). If it finds a solution, update best.
    # Repeat with lower targets if time permits.
    
    # To make it robust, we will interleave:
    # - Try to find any solution with scope < current_best
    # - If found, update current_best and continue
    # - If not found within time slice, increase the target or restart.
    
    # Given the time limit, let's just try a few specific targets.
    # The gap between greedy and optimal is usually small for these parameters.
    
    # Let's try targets: best_scope - 1, best_scope - 2, ..., down to max(lower_bound, best_scope - 10)
    # And also some random targets?
    
    # Actually, let's just run the backtracking search with target_max = best_scope - 1
    # and see if it succeeds. If it does, we improved.
    # If it fails, we try best_scope - 2, etc.
    
    # To avoid infinite loops in backtracking, we need a node limit.
    
    def backtrack_row_limited(row_idx, used_diffs, current_row, target_max, node_limit, nodes_used):
        if nodes_used[0] > node_limit:
            return False
        
        if len(current_row) == m:
            return True
        
        last = current_row[-1]
        rem = m - len(current_row) - 1
        max_x = target_max - rem
        
        if max_x < last + 1:
            return False
            
        for x in range(last + 1, max_x + 1):
            nodes_used[0] += 1
            if nodes_used[0] > node_limit:
                return False
                
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                current_row.append(x)
                for d in new_diffs:
                    used_diffs.add(d)
                
                if backtrack_row_limited(row_idx, used_diffs, current_row, target_max, node_limit, nodes_used):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
        
        return False

    def search_limited(target_max, seed, node_limit):
        rng_local = random.Random(seed)
        used_diffs = set()
        rows = []
        nodes_used = [0]
        
        for r_idx in range(n_rows):
            current_row = [0]
            if not backtrack_row_limited(r_idx, used_diffs, current_row, target_max, node_limit, nodes_used):
                return None
            rows.append(current_row)
        
        return rows

    # Search loop
    # We will try to improve best_scope.
    # We'll iterate target_max downwards.
    
    # To speed up, we can use a heuristic: 
    # If a target fails quickly, try a slightly higher one? No, we want lower.
    # If a target fails after many nodes, it's likely infeasible.
    
    # Let's allocate time to each target attempt.
    # Start with target = best_scope - 1
    
    current_target = best_scope - 1
    min_target = max(lower_bound, 10) # Safety
    
    # We will run multiple attempts for each target with different seeds to escape local minima.
    
    attempt = 0
    while time.time() < deadline and current_target >= min_target:
        # Try to find a solution with this target
        found = False
        # Try a few seeds
        for seed in range(attempt * 10, attempt * 10 + 5):
            if time.time() > deadline:
                break
            # Node limit: e.g., 50000 nodes per attempt
            # This is a heuristic. 
            sol = search_limited(current_target, seed, 50000)
            if sol is not None:
                # Verify
                if validate(sol):
                    sc = compute_scope(sol)
                    if sc < best_scope:
                        best_scope = sc
                        best_rows = sol
                    found = True
                    break
        if found:
            # We found a better solution. Try to go lower.
            current_target -= 1
            attempt = 0
        else:
            # Failed to find a solution for this target.
            # Is it infeasible? Maybe.
            # Let's move to the next lower target? 
            # No, if target T is infeasible, T-1 is also infeasible? 
            # Not necessarily. A solution with max entry T-1 is a subset of solutions with max entry T.
            # Wait. If there is a solution with max entry <= T-1, then there is a solution with max entry <= T.
            # So if T is infeasible, T-1 is infeasible.
            # Therefore, if we fail to find a solution for T, we can stop?
            # NO. The search is incomplete (node limit). We might have failed due to time, not infeasibility.
            # So we should try T-1? No, if T is infeasible, T-1 is too.
            # But we don't know if T is infeasible.
            # So we should try T-1? No, that's a smaller scope, harder to find.
            # If we couldn't find a solution for T, we definitely can't find one for T-1 (which is stricter).
            # So we should stop? 
            # Wait. If we fail to find a solution for T, it means we didn't find one *within the node limit*.
            # It might exist.
            # But if we try T-1, we are looking for an even smaller scope.
            # It is more likely to be infeasible or harder to find.
            # So if we fail T, we should probably give up on improving further?
            # Or, we can try a different search strategy for T?
            
            # Let's assume that if we fail T with reasonable effort, T-1 is not worth trying.
            # We will break the loop.
            break

    # Final validation
    if not validate(best_rows):
        # Fallback to greedy
        best_rows = greedy_init(42)
        
    return best_rows

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
