import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145
    num_elements = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    def elapsed():
        return time.time() - start_time

    def deadline_check():
        return elapsed() > deadline

    def solve_greedy():
        """Simple greedy: pick smallest valid next element for each row."""
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << (d - 1)
                    if (used & bit) or (new & bit):
                        valid = False
                        break
                    new |= bit
                if valid:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Get baseline
    best_result = solve_greedy()
    best_max = max(max(r) for r in best_result)

    def build_row_backtrack(used_mask, row, pos, current_max, rng, max_window):
        """Backtracking to build one row given current used_mask."""
        if deadline_check():
            return None
        if pos == num_elements:
            return row[:]

        start_val = current_max + 1
        # Use a window of candidates
        candidates = list(range(start_val, start_val + max_window))
        rng.shuffle(candidates)

        for x in candidates:
            if deadline_check():
                return None

            new_diffs = 0
            valid = True
            for a in row:
                d = x - a
                bit = 1 << (d - 1)
                if (used_mask & bit) or (new_diffs & bit):
                    valid = False
                    break
                new_diffs |= bit

            if not valid:
                continue

            row.append(x)
            result = build_row_backtrack(used_mask | new_diffs, row, pos + 1, x, rng, max_window)
            row.pop()

            if result is not None:
                return result

        return None

    def mrv_build_all_rows(rng, max_window):
        """
        Build all rows using MRV (Minimum Remaining Values) heuristic.
        Instead of building rows sequentially, we track which differences are
        still available and always build the row that has the most constraints
        (i.e., the row whose remaining elements have the fewest valid candidates).
        
        Actually, a simpler MRV approach: build rows one at a time, but for each
        row, use backtracking with a tight window. The key insight from the
        knowledge is that MRV-based global selection helps. Let's interpret this
        as: when building a row, at each position, try candidates in order of
        how many future options they leave (hard to compute exactly, so we use
        a proxy: prefer smaller candidates that pack tightly).
        
        A better structural change: use a two-phase approach.
        Phase 1: Greedy with randomization to get a good initial solution.
        Phase 2: Local search / hill climbing to improve.
        """
        used = 0
        rows = []
        
        for row_idx in range(n_rows):
            if deadline_check():
                return None
            
            row = [0]
            result = build_row_backtrack(used, row, 1, 0, rng, max_window)
            if result is None:
                return None
            
            # Update used mask
            row_diffs = 0
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    d = result[j] - result[i]
                    row_diffs |= (1 << (d - 1))
            used |= row_diffs
            rows.append(result)
        
        return rows

    def local_search(rows, rng, max_iterations):
        """
        Try to improve the solution by removing and re-adding elements.
        This is a form of simulated annealing / hill climbing.
        """
        rows = [r[:] for r in rows]
        current_max = max(max(r) for r in rows)
        
        # Compute used mask
        used = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    used |= (1 << (d - 1))
        
        improved = True
        while improved and not deadline_check():
            improved = False
            for row_idx in range(n_rows):
                if deadline_check():
                    break
                row = rows[row_idx]
                # Try to reduce the largest element
                if len(row) < 2:
                    continue
                max_elem = row[-1]
                # Try replacing max_elem with something smaller
                for new_val in range(max_elem - 1, 0, -1):
                    if new_val <= row[-2]:
                        break
                    # Check if new_val can replace max_elem
                    new_diffs = 0
                    valid = True
                    for a in row[:-1]:
                        d = new_val - a
                        bit = 1 << (d - 1)
                        # Remove old diffs involving max_elem
                        old_bit = 1 << (max_elem - a - 1)
                        # We need to check against (used without old max_elem diffs)
                        # This is complex, so let's just check against full used
                        # and exclude the diffs that involve max_elem
                        pass
                    
                    # Simpler: rebuild the row with new_val and check
                    test_row = row[:-1] + [new_val]
                    test_diffs = 0
                    valid = True
                    for i in range(len(test_row)):
                        for j in range(i + 1, len(test_row)):
                            d = test_row[j] - test_row[i]
                            bit = 1 << (d - 1)
                            if test_diffs & bit:
                                valid = False
                                break
                            test_diffs |= bit
                    if not valid:
                        continue
                    
                    # Check against other rows' diffs
                    other_diffs = 0
                    for other_idx in range(n_rows):
                        if other_idx == row_idx:
                            continue
                        other_row = rows[other_idx]
                        for i in range(len(other_row)):
                            for j in range(i + 1, len(other_row)):
                                d = other_row[j] - other_row[i]
                                other_diffs |= (1 << (d - 1))
                    
                    if test_diffs & other_diffs:
                        continue
                    
                    # Valid replacement
                    rows[row_idx] = test_row
                    current_max = min(current_max, max_elem)
                    improved = True
                    break
        
        return rows, current_max

    # Run multiple randomized attempts
    for attempt in range(200):
        if deadline_check():
            break
        rng = random.Random(attempt * 7919 + 13)
        
        # Vary the window size
        max_window = 30 + (attempt % 5) * 10
        
        result = mrv_build_all_rows(rng, max_window)
        if result is None:
            continue
        
        max_val = max(max(r) for r in result)
        
        # Try local search
        if not deadline_check():
            result_ls, max_val_ls = local_search(result, rng, 50)
            if max_val_ls < max_val:
                max_val = max_val_ls
                result = result_ls
        
        if max_val < best_max:
            best_max = max_val
            best_result = [r[:] for r in result]

    return best_result
