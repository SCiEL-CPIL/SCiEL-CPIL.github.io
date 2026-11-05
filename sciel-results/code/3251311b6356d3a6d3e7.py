import time
import random

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Total number of differences per row: k*(k+1)/2
    # Total differences needed: n_rows * k*(k+1)/2
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    # The maximum possible difference is m (the largest entry)
    # We need at least total_diffs_needed distinct differences in [1, m]
    # So m must be at least total_diffs_needed
    # But also, for a single row with entries 0=a0<a1<...<ak, the differences are a_j-a_i
    # The largest difference in a row is a_k = m_row
    # We want to minimize max over rows of m_row

    # Lower bound: m >= total_diffs_needed (since we need that many distinct positive integers)
    # Also m >= k (trivially)
    
    # Use a randomized backtracking / greedy hybrid approach
    # Strategy: Try to build rows one by one, using a smarter greedy that considers
    # the global used-difference set. Use randomized restarts with different orderings.

    def greedy_row(used_diffs, k, max_val, rng):
        """Build a single row of k+1 entries starting at 0, using greedy with backtracking."""
        row = [0]
        for pos in range(1, k + 1):
            # Try candidates from small to large
            found = False
            for x in range(row[-1] + 1, max_val + 1):
                # Check all differences with previous entries
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    found = True
                    break
            if not found:
                return None
            used_diffs.update(new_diffs)
        return row

    def greedy_all_rows(used_diffs, n_rows, k, max_val, rng):
        """Build all rows greedily."""
        rows = []
        for _ in range(n_rows):
            row = greedy_row(used_diffs, k, max_val, rng)
            if row is None:
                return None
            rows.append(row)
        return rows

    # Try different max_val values starting from lower bound
    # Lower bound: we need total_diffs_needed distinct differences, so max_val >= total_diffs_needed
    # But actually the differences in one row don't cover all of [1, max_val], they're sparse
    # A better lower bound: for a row ending at m, the differences are at most m, and there are
    # k*(k+1)/2 of them. For n_rows rows, we need n_rows*k*(k+1)/2 distinct values in [1, M]
    # where M is the overall max. So M >= n_rows*k*(k+1)/2.
    
    lb = total_diffs_needed
    
    # Also, each row has max entry m_i, and the differences in that row are all <= m_i
    # The sum of all differences across all rows must be distinct, so we need at least
    # total_diffs_needed distinct values, each <= M.
    
    # Let's try max_val from lb upward
    # But for efficiency, let's also try a smarter approach:
    # Use a randomized greedy where we shuffle the order of trying candidates slightly
    
    best_result = None
    best_max_val = None
    
    # First, try deterministic greedy with increasing max_val
    for max_val in range(lb, lb + 50):
        if time.time() > deadline - 10:
            break
        used_diffs = set()
        rows = greedy_all_rows(used_diffs, n_rows, k, max_val, random.Random(max_val))
        if rows is not None:
            actual_max = max(max(row) for row in rows)
            if best_max_val is None or actual_max < best_max_val:
                best_max_val = actual_max
                best_result = rows
            break
    
    # If we found a solution, try to improve with randomized restarts
    if best_result is not None:
        # Try to find a better solution with randomized approaches
        attempts = 0
        while time.time() < deadline - 5 and attempts < 50:
            attempts += 1
            rng = random.Random(attempts * 12345 + 67890)
            
            # Try with a slightly higher max_val but randomized ordering
            # Or try a different greedy strategy: randomize which row to build first
            used_diffs = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                for pos in range(1, k + 1):
                    # Try candidates, but with some randomization in tie-breaking
                    candidates = list(range(row[-1] + 1, best_max_val + 1))
                    rng.shuffle(candidates)
                    # But we want to prefer smaller values, so sort by (value, random_key)
                    # Actually, let's just try in order but with a random offset
                    found = False
                    start_offset = rng.randint(0, 3)
                    for x in range(row[-1] + 1 + start_offset, best_max_val + 1):
                        new_diffs = set()
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            found = True
                            break
                    if not found:
                        success = False
                        break
                    used_diffs.update(new_diffs)
                if not success:
                    break
                rows.append(row)
            
            if success:
                actual_max = max(max(row) for row in rows)
                if actual_max < best_max_val:
                    best_max_val = actual_max
                    best_result = rows
                    break
    
    # If no solution found yet, try with higher max_val
    if best_result is None:
        for max_val in range(lb, lb + 200):
            if time.time() > deadline - 5:
                break
            used_diffs = set()
            rows = greedy_all_rows(used_diffs, n_rows, k, max_val, random.Random(max_val))
            if rows is not None:
                actual_max = max(max(row) for row in rows)
                if best_max_val is None or actual_max < best_max_val:
                    best_max_val = actual_max
                    best_result = rows
                break
    
    # Final fallback: very large max_val to ensure we return something valid
    if best_result is None:
        max_val = lb * 2 + 100
        used_diffs = set()
        rows = greedy_all_rows(used_diffs, n_rows, k, max_val, random.Random(42))
        if rows is not None:
            best_result = rows
    
    if best_result is None:
        # Absolute fallback: construct a trivially valid (but large) solution
        rows = []
        used_diffs = set()
        base = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                # Use a large step to avoid collisions
                x = base + j * (total_diffs_needed + 1)
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        break
                    x += 1
                row.append(x)
                used_diffs |= new_diffs
            rows.append(row)
            base = x + 1
        best_result = rows
    
    return best_result
