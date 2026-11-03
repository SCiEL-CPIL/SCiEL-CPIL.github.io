import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound for scope
    # Total diffs must be distinct, so max diff >= total_diffs
    # Also, each row has k diffs involving the last element, etc.
    # Simple bound: m >= k + 1 + total_diffs? No.
    # The largest difference in a row is at most m-1.
    # There are total_diffs distinct differences in [1, m-1].
    # So m-1 >= total_diffs => m >= total_diffs + 1.
    # For n_rows=4, k=7, total_diffs = 4 * 28 = 112. So m >= 113.
    # Best known is 134.
    lower_bound = total_diffs + 1
    
    # Upper bound: greedy construction gives a valid one.
    # We can estimate an upper bound or just use a large number and binary search down.
    # Let's start with a reasonable upper bound. 
    # Greedy usually produces scope around 2-3x the lower bound.
    upper_bound = lower_bound * 2 + 20

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(n_rows, k_val, rng, max_scope=None):
        """
        Greedy construction with optional scope limit.
        Returns list of rows or None if fails.
        """
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(k_val):
                found = False
                if max_scope is not None:
                    # Try random candidates in increasing order of value? 
                    # Or just random? 
                    # To minimize scope, we should prefer smaller values.
                    # But we need to ensure we can complete the row.
                    # Let's try values from row[-1]+1 to max_scope.
                    # To make it faster, we can sample.
                    candidates = []
                    lo = row[-1] + 1
                    hi = max_scope
                    if lo > hi:
                        return None
                    # Try a few random candidates, then fallback to sequential if needed?
                    # Actually, for minimization, we want the smallest possible next element.
                    # But backtracking is hard in greedy.
                    # Let's use a randomized greedy: pick a random valid x in [lo, hi].
                    # If we want to minimize scope, we should bias towards smaller x.
                    for _ in range(20):
                        # Bias towards smaller values
                        x = lo + int((rng.random() ** 2) * (hi - lo + 1))
                        if x > hi: x = hi
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                    if not found:
                        # Fallback: try sequential from lo
                        x = lo
                        while x <= hi:
                            ok = True
                            for a in row:
                                if (x - a) in used:
                                    ok = False
                                    break
                            if ok:
                                for a in row:
                                    used.add(x - a)
                                row.append(x)
                                found = True
                                break
                            x += 1
                        if not found:
                            return None
                else:
                    x = row[-1]
                    while True:
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
                            found = True
                            break
            rows.append(row)
        return rows

    def csp_backtrack(n_rows, k_val, max_scope, rng, time_limit):
        """
        CSP Backtracking to construct rows with max entry <= max_scope.
        Variables: The elements of the rows.
        We construct row by row.
        For each row, we backtrack on the elements.
        
        State:
        - rows: list of completed rows
        - current_row: list of elements for the row being built
        - used_diffs: set of all differences used so far
        
        To speed up, we can pre-compute valid next steps? 
        Or use MRV: pick the position in the current row that has the fewest valid candidates?
        Since the row must be increasing, we build left to right.
        However, the choice of x_i affects x_{i+1}...x_k.
        
        Let's try a recursive backtracking for a single row, given the used_diffs from previous rows.
        """
        
        def build_row(used_diffs, max_scope, rng):
            """
            Build a single row of length m, starting with 0, max entry <= max_scope.
            Returns the row or None.
            """
            row = [0]
            
            def backtrack(pos):
                if pos == m:
                    return True
                
                # MRV: We are at position pos. We need to choose x_pos.
                # x_pos must be > x_{pos-1} and <= max_scope.
                # Also, for all i < pos, x_pos - x_i must not be in used_diffs.
                # And x_pos - x_i must be distinct from each other? 
                # No, the differences within the row must be distinct globally.
                # So if we pick x_pos, the new diffs are {x_pos - x_i for i in 0..pos-1}.
                # These must be distinct from each other (guaranteed if x is increasing? No, x_pos - x_i are distinct if x_i are distinct).
                # And they must not be in used_diffs.
                
                # Candidate range: [row[-1] + 1, max_scope]
                # To improve efficiency, we can shuffle candidates or use heuristic ordering.
                # Heuristic: Try smaller values first to keep scope low? 
                # But we are bounded by max_scope.
                # Let's try values in random order, but biased towards smaller ones to find compact solutions faster?
                # Or just random to avoid bias.
                
                lo = row[-1] + 1
                hi = max_scope
                if lo > hi:
                    return False
                
                # Generate candidates
                # If the range is small, iterate. If large, sample.
                # For now, iterate but with a limit on attempts? 
                # No, backtracking requires exploring.
                # But if max_scope is tight, the range is small.
                
                candidates = list(range(lo, hi + 1))
                rng.shuffle(candidates)
                
                # Heuristic: Sort by some cost? 
                # Maybe prefer values that don't create "gaps" that are hard to fill?
                # For now, random order is a good baseline.
                
                for x in candidates:
                    # Check validity of x
                    new_diffs = []
                    valid = True
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    # Check if new_diffs are distinct from each other
                    # Since row is increasing, x - row[i] are distinct for distinct i.
                    # So this is always true.
                    
                    # Commit
                    row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    # Uncommit
                    row.pop()
                    for d in new_diffs:
                        used_diffs.remove(d)
                
                return False
            
            if backtrack(1):
                return row
            return None
        
        # Main loop: build rows one by one
        used_diffs = set()
        rows = []
        
        for r in range(n_rows):
            if time.time() > time_limit:
                return None
            
            # Try to build a row
            # We might need to restart if we get stuck? 
            # Backtracking within build_row handles the current row.
            # But if build_row fails, it means no row can be built with the current used_diffs?
            # Actually, build_row explores all possibilities for the current row.
            # If it fails, it means the current set of used_diffs blocks all possibilities for a new row?
            # That would imply the previous rows were "bad" choices.
            # To handle this, we would need to backtrack over previous rows too.
            # That is much more complex.
            
            # Alternative: Randomized restarts.
            # If build_row fails, return None and let the caller restart with different random seeds.
            
            row = build_row(used_diffs, max_scope, rng)
            if row is None:
                return None
            
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
            
        return rows

    def binary_search_scope(n_rows, k_val, rng, deadline):
        lo = lower_bound
        hi = upper_bound
        
        # First, find a valid upper bound with greedy
        # Greedy doesn't have a hard scope limit, but we can cap it.
        # Let's run greedy without cap to get a valid solution and its scope.
        greedy_rows = greedy_construct(n_rows, k_val, rng)
        if greedy_rows is None:
            return None, None
        greedy_scope = max(max(r) for r in greedy_rows)
        hi = min(hi, greedy_scope)
        
        best_rows = greedy_rows
        best_scope = greedy_scope
        
        # Binary search for minimal scope
        # We want to find the smallest m such that csp_backtrack succeeds.
        # CSP backtracking is expensive. We should limit its time.
        
        while lo <= hi:
            if time.time() > deadline - 10:
                break
                
            mid = (lo + hi) // 2
            
            # Try CSP backtrack with time limit
            # Give it a portion of the remaining time
            time_limit = time.time() + min(10, deadline - time.time() - 5)
            
            # Use a different random seed for each attempt?
            # Or just one attempt?
            # Let's try a few restarts if it fails quickly.
            
            success = False
            rows = None
            
            # Try up to 3 restarts for each mid
            for attempt in range(3):
                if time.time() > time_limit:
                    break
                rng_attempt = random.Random(rng.randint(0, 1000000))
                rows = csp_backtrack(n_rows, k_val, mid, rng_attempt, time_limit)
                if rows is not None:
                    success = True
                    break
            
            if success:
                best_rows = rows
                best_scope = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return best_rows, best_scope

    # Main execution
    rng = random.Random(42)
    
    # 1. Get a baseline solution using greedy
    baseline_rows = greedy_construct(n_rows, k, rng)
    if baseline_rows is None:
        # Fallback: simple sequential
        used = set()
        baseline_rows = []
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
            baseline_rows.append(row)
    
    best_rows = baseline_rows
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Try to improve using CSP Backtracking with Binary Search
    # We only do this if we have enough time
    if time.time() < deadline - 20:
        csp_rows, csp_scope = binary_search_scope(n_rows, k, rng, deadline)
        if csp_rows is not None and csp_scope < best_scope:
            best_rows = csp_rows
            best_scope = csp_scope
    
    # 3. Final polish: Local search on the best row to reduce scope further?
    # If we have time, try to reduce the scope by 1 using local search.
    if time.time() < deadline - 5:
        # Try to reduce scope by 1
        target_scope = best_scope - 1
        if target_scope >= lower_bound:
            # Use CSP backtrack with target_scope
            time_limit = time.time() + min(10, deadline - time.time() - 2)
            rng_polish = random.Random(999)
            polish_rows = csp_backtrack(n_rows, k, target_scope, rng_polish, time_limit)
            if polish_rows is not None:
                best_rows = polish_rows
                best_scope = target_scope
                
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
