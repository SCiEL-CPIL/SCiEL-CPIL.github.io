"""Construct difference triangle set (7,5) - minimize largest entry."""
import time
import itertools

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Each row has k+1 entries: 0 = a0 < a1 < ... < ak
    # We need all differences a_j - a_i (i<j) across all rows to be distinct
    # Total differences per row: k*(k+1)/2
    # Total differences needed: n_rows * k*(k+1)/2
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Generate all valid single rows with max entry <= bound
    # A row is a tuple (0, a1, a2, ..., ak) with 0 < a1 < ... < ak <= bound
    # Its differences are all a_j - a_i for 0 <= i < j <= k
    
    def get_row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def generate_rows(bound):
        """Generate all valid rows with max entry <= bound, sorted by max entry."""
        # Row is (0, a1, a2, ..., ak) where 1 <= a1 < a2 < ... < ak <= bound
        rows = []
        # Use combinations for the remaining k entries from 1..bound
        # But we need exactly k entries from range(1, bound+1)
        # That's C(bound, k) possibilities - could be large
        # For k=5, bound=112: C(112,5) ~ 64M - too many to enumerate all
        
        # Instead, use backtracking to generate rows incrementally
        # and track which ones are valid (all internal diffs distinct - always true for increasing)
        # Actually all rows with strictly increasing entries have distinct internal diffs
        # Wait no - we need the differences within a single row to be distinct too
        # For a row (0, a1, ..., ak), diffs are a_j - a_i for i<j
        # These must all be distinct within the row as well
        
        # Let's generate rows via backtracking
        result = []
        
        def backtrack(pos, current, used_in_row):
            if time.time() > deadline:
                return
            if pos == k:
                # Check all differences are distinct
                diffs = set()
                ok = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        d = current[j] - current[i]
                        if d in diffs:
                            ok = False
                            break
                        diffs.add(d)
                    if not ok:
                        break
                if ok:
                    result.append((tuple(current), diffs, current[-1]))
                return
            
            start = current[-1] + 1 if pos > 0 else 1
            for val in range(start, bound + 1):
                # Check if adding val keeps all diffs distinct so far
                ok = True
                for i in range(len(current)):
                    d = val - current[i]
                    if d in used_in_row:
                        ok = False
                        break
                if ok:
                    new_used = used_in_row | {val - current[i] for i in range(len(current))}
                    current.append(val)
                    backtrack(pos + 1, current, new_used)
                    current.pop()
        
        backtrack(0, [0], set())
        # Sort by max entry (last element)
        result.sort(key=lambda x: x[2])
        return result
    
    # For efficiency, let's use a smarter approach:
    # Precompute a good set of candidate rows up to some bound
    # Then use backtracking to select n_rows non-conflicting rows
    
    # Start with a reasonable bound and increase
    # The greedy solution gives us an initial bound
    used = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        greedy_rows.append(row)
    
    initial_bound = max(max(r) for r in greedy_rows)
    
    # Generate candidate rows up to initial_bound + some margin
    # But C(112, 5) is too large. Let's be smarter.
    
    # Alternative: use a more targeted search
    # For each candidate max value M, try to build rows with max <= M
    
    # Let's use a different approach: iterative improvement
    # Start with greedy, then try to improve by replacing rows
    
    best_rows = [list(r) for r in greedy_rows]
    best_max = initial_bound
    
    def compute_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def try_improve(rows, deadline):
        """Try to improve the set by local search."""
        best = [list(r) for r in rows]
        best_max = max(max(r) for r in best)
        best_diffs = compute_all_diffs(best)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(n_rows):
                # Try to find a better row for position idx
                other_diffs = set()
                for j in range(n_rows):
                    if j != idx:
                        row = best[j]
                        for i in range(len(row)):
                            for l in range(i + 1, len(row)):
                                other_diffs.add(row[l] - row[i])
                
                current_max = max(best[idx])
                # Try to find a row with smaller max that doesn't conflict
                # Generate candidate rows with max < current_max
                found_better = False
                # Try max values from 1 upward
                for target_max in range(1, current_max):
                    if time.time() > deadline:
                        return best, best_max, best_diffs
                    
                    # Find a row with exactly this max that has no conflict with other_diffs
                    # Row: (0, a1, ..., a_{k-1}, target_max)
                    # Need to choose a1, ..., a_{k-1} from 1..target_max-1
                    # such that all diffs are distinct and not in other_diffs
                    
                    def find_row(target_max, other_diffs, deadline):
                        # Backtrack to find (a1, ..., a_{k-1}) 
                        # such that all diffs in the full row are distinct and avoid other_diffs
                        partial = [0]
                        
                        def bt(pos, partial, row_diffs):
                            if time.time() > deadline:
                                return None
                            if pos == k:
                                # Full row found
                                return list(partial)
                            
                            start = partial[-1] + 1
                            for val in range(start, target_max):  # must be < target_max
                                # Check new diffs
                                new_diffs = set()
                                ok = True
                                for a in partial:
                                    d = val - a
                                    if d in row_diffs or d in other_diffs:
                                        ok = False
                                        break
                                    new_diffs.add(d)
                                if ok:
                                    result = bt(pos + 1, partial + [val], row_diffs | new_diffs)
                                    if result is not None:
                                        return result
                            return None
                        
                        # First, check that target_max itself doesn't create conflicts
                        # Diffs from target_max to all existing elements
                        init_diffs = set()
                        for a in [0]:
                            d = target_max - a
                            if d in other_diffs:
                                return None
                            init_diffs.add(d)
                        
                        result = bt(1, [0], init_diffs)
                        if result is not None:
                            result.append(target_max)
                        return result
                    
                    row = find_row(target_max, other_diffs, deadline)
                    if row is not None:
                        # Verify
                        row_diffs = set()
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                row_diffs.add(row[j] - row[i])
                        if row_diffs.isdisjoint(other_diffs):
                            new_rows = [list(r) for r in best]
                            new_rows[idx] = row
                            new_max = max(max(r) for r in new_rows)
                            if new_max < best_max:
                                best = new_rows
                                best_max = new_max
                                best_diffs = compute_all_diffs(best)
                                improved = True
                                found_better = True
                                break
                    # Limit search for efficiency
                    if target_max > 30:  # Don't search too deep for each position
                        break
        return best, best_max, best_diffs
    
    # Run improvement
    best_rows, best_max, _ = try_improve(best_rows, deadline)
    
    # If we still have time, try more aggressive search
    if time.time() < deadline - 10:
        # Try different starting points
        import random
        random.seed(42)
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Create a perturbed version
            perturbed = [list(r) for r in best_rows]
            # Randomly replace one row with a random valid row
            idx = random.randint(0, n_rows - 1)
            # Generate a random row
            candidates = list(range(1, best_max + 1))
            if len(candidates) >= k:
                chosen = sorted(random.sample(candidates, k))
                perturbed[idx] = [0] + chosen
            
            # Check validity
            all_diffs = set()
            valid = True
            for row in perturbed:
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in row_diffs or d in all_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                if not valid:
                    break
                all_diffs |= row_diffs
            
            if valid:
                cur_max = max(max(r) for r in perturbed)
                if cur_max < best_max:
                    best_rows = perturbed
                    best_max = cur_max
                
                # Try to improve from here
                improved_rows, improved_max, _ = try_improve(perturbed, deadline)
                if improved_max < best_max:
                    best_rows = improved_rows
                    best_max = improved_max
    
    return best_rows
