import time
import random
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry such that all positive differences within rows
    are distinct across all rows.
    """
    start_time = time.time()
    DEADLINE = 120  # seconds, leave margin before 150s kill

    # Total number of differences needed: n_rows * k*(k+1)/2
    # The scope m must satisfy: number of distinct differences in [1, m] >= total_needed
    # Differences in [1, m] count = m*(m+1)/2
    # So m*(m+1)/2 >= n_rows * k*(k+1)/2
    # m^2 + m >= n_rows*(k+1)  (roughly)
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound on m
    import math
    m_lower = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))
    
    # Upper bound: start with a reasonable estimate
    # Greedy gives an upper bound; let's compute a quick greedy first
    def greedy_construct(n_rows, k, max_scope):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                if x > max_scope:
                    return None
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Quick greedy to get an initial upper bound
    initial = greedy_construct(n_rows, k, 200)
    if initial is None:
        # Try larger scope
        initial = greedy_construct(n_rows, k, 500)
    
    if initial is None:
        # Fallback: return a trivially valid but possibly large solution
        # Use a simple construction
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
    
    current_best = max(max(row) for row in initial)
    best_rows = initial
    
    # Backtracking search with limited branching
    def backtrack(n_rows, k, max_scope, timeout):
        """
        Try to construct n_rows rows with scope at most max_scope.
        Uses backtracking with limited branching factor.
        Returns list of rows or None.
        """
        start = time.time()
        used = set()
        rows = []
        
        def solve_row(row_idx, row, pos):
            """Try to complete row starting at position pos."""
            if time.time() - start > timeout:
                return False
            
            if pos == k + 1:
                rows.append(row[:])
                return solve_next_row()
            
            # Try candidates from pos+1 up to max_scope
            # But we need to be smart about which to try first
            last = row[-1]
            for x in range(last + 1, max_scope + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
                # Check if this is promising
                row.append(x)
                used.update(new_diffs)
                
                if solve_row(row_idx, row, pos + 1):
                    return True
                
                row.pop()
                used.difference_update(new_diffs)
            
            return False
        
        def solve_next_row():
            if time.time() - start > timeout:
                return False
            if len(rows) == n_rows:
                return True
            return solve_row(len(rows), [0], 0)
        
        return solve_next_row()
    
    # Try to improve upon the greedy solution
    # Binary search or linear descent on max_scope
    # Start from current_best and try to go lower
    
    # First, let's try a randomized greedy with restarts
    def randomized_greedy(n_rows, k, max_scope, seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                # Collect all valid candidates
                valid = []
                for x in range(last + 1, max_scope + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid.append(x)
                if not valid:
                    return None
                # Pick randomly from valid, but bias towards smaller
                # Weighted by inverse value to prefer smaller
                weights = [1.0 / v for v in valid]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = valid[-1]
                for v, w in zip(valid, weights):
                    cum += w
                    if r <= cum:
                        chosen = v
                        break
                row.append(chosen)
                last = chosen
                used.update({chosen - a for a in row[:-1]})
            rows.append(row)
        return rows
    
    # Try multiple randomized restarts with decreasing scope
    scope_to_try = current_best
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        if scope_to_try <= m_lower:
            break
        
        # Try to find a solution with this scope
        found = False
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            result = randomized_greedy(n_rows, k, scope_to_try, seed + attempt * 100)
            if result is not None:
                # Verify validity
                all_diffs = set()
                valid = True
                for row in result:
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            d = row[j] - row[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                if valid and max(max(row) for row in result) <= scope_to_try:
                    if max(max(row) for row in result) < current_best:
                        current_best = max(max(row) for row in result)
                        best_rows = result
                    found = True
                    break
        
        if found:
            scope_to_try -= 1
        else:
            # Couldn't find with this scope, try slightly lower
            scope_to_try = max(m_lower, scope_to_try - 2)
    
    # Final backtracking attempt if we have time
    if time.time() - start_time < DEADLINE - 20 and current_best > m_lower:
        # Try backtracking with scope = current_best - 1
        target_scope = current_best - 1
        if target_scope >= m_lower:
            timeout = min(15, DEADLINE - 25 - (time.time() - start_time))
            if timeout > 5:
                result = backtrack(n_rows, k, target_scope, timeout)
                if result is not None:
                    all_diffs = set()
                    valid = True
                    for row in result:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                d = row[j] - row[i]
                                if d in all_diffs:
                                    valid = False
                                    break
                                all_diffs.add(d)
                        if not valid:
                            break
                    if valid:
                        current_best = max(max(row) for row in result)
                        best_rows = result
    
    return best_rows
