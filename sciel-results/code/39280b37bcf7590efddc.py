"""Improved program for difference triangle set (7,5)."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Strategy:
    1. Use constraint-satisfaction backtracking with most-constrained-first heuristic.
    2. Model the global set of used differences as a resource to be partitioned.
    3. For each candidate element, score by how many future difference slots remain.
    4. Multiple random restarts with different orderings and perturbations.
    """
    deadline = time.time() + 110  # 110s safety margin (limit is 150s)
    best_rows = None
    best_max = float('inf')
    
    # Number of differences per row
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Lower bound on max entry
    lower_bound = total_diffs_needed
    
    # Precompute: for a given max_val, how many differences are available in [1, max_val]
    # We need total_diffs_needed distinct differences, each in [1, max_val]
    # So max_val >= total_diffs_needed
    
    def get_row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_solution(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_row_diffs(row)
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def backtracking_construct(n_rows, k, max_val, rng, time_limit=None):
        """
        Backtracking search with most-constrained-first heuristic.
        Builds rows one element at a time, choosing the next element that
        leaves the most remaining valid difference slots for subsequent elements.
        """
        if time_limit is None:
            time_limit = deadline
        
        used = set()
        rows = []
        current_row = [0]
        
        # Track which positions we're filling
        # We'll build each row completely before moving to the next
        # But we can also interleave: build all rows element-by-element
        
        # Strategy: build row by row, but within each row, use backtracking
        # with a heuristic that considers the global impact
        
        def score_candidate(x, current_row, used, remaining_elements_in_row, rows_completed):
            """
            Score a candidate value x for the next position.
            Lower score = better (more constrained = preferred in backtracking).
            We want to pick candidates that:
            1. Don't conflict with used differences
            2. Leave enough room for future elements
            3. Minimize the max entry
            """
            # Check if x creates any conflicts
            new_diffs = set()
            for a in current_row:
                d = x - a
                if d > 0:
                    if d in used:
                        return float('inf')  # Conflict
                    new_diffs.add(d)
            
            # Score: prefer smaller x (to keep max small)
            # But also consider how many future difference slots are blocked
            # For each future element position, count how many values would be blocked
            future_blocked = 0
            for future_pos in range(len(current_row) + 1, k + 2):  # 1-indexed positions
                # The next element must be > x
                # For each possible next value y > x, check if y - a creates conflicts
                # This is expensive, so approximate: count how many differences from x
                # to potential future values would be in [1, max_val] and not in used
                pass
            
            # Simple heuristic: prefer smaller x
            return x * 10 + len(new_diffs)
        
        def backtrack(row_idx, pos_in_row, current_row, used):
            """
            Recursive backtracking.
            row_idx: which row we're building
            pos_in_row: which position in the current row (0-indexed, 0 is always 0)
            current_row: the current row being built
            used: set of all differences used so far
            """
            if time.time() > time_limit:
                return False
            
            # If we've completed all rows
            if row_idx == n_rows:
                return True
            
            # If we've completed the current row
            if pos_in_row == k + 1:
                rows.append(current_row[:])
                return backtrack(row_idx + 1, 0, [0], used)
            
            # Generate candidates for this position
            # The minimum valid value is current_row[-1] + 1
            min_val = current_row[-1] + 1 if current_row else 1
            max_val_for_pos = max_val
            
            # Get candidate values
            candidates = list(range(min_val, max_val_for_pos + 1))
            
            # Shuffle for randomization
            rng.shuffle(candidates)
            
            # Sort by heuristic score (most constrained first)
            def heuristic(x):
                # Check conflicts
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d > 0:
                        if d in used:
                            return float('inf')
                        new_diffs.add(d)
                
                # Prefer smaller values to keep max small
                # But also consider how many future slots are affected
                # Approximate: number of differences created
                return x * 100 + len(new_diffs)
            
            # Filter out conflicting candidates
            valid_candidates = []
            for x in candidates:
                new_diffs = set()
                conflict = False
                for a in current_row:
                    d = x - a
                    if d > 0:
                        if d in used:
                            conflict = True
                            break
                        new_diffs.add(d)
                if not conflict:
                    valid_candidates.append((x, new_diffs, heuristic(x)))
            
            # Sort by heuristic (most constrained first = lowest score)
            valid_candidates.sort(key=lambda t: t[2])
            
            # Try each candidate
            for x, new_diffs, _ in valid_candidates:
                # Check if this leaves enough room for future elements
                # The remaining elements in this row need to be > x
                # And they need to create distinct differences not in used | new_diffs
                
                # Try this candidate
                old_used = used.copy()
                used |= new_diffs
                current_row.append(x)
                
                if backtrack(row_idx, pos_in_row + 1, current_row, used):
                    return True
                
                # Backtrack
                current_row.pop()
                used = old_used
            
            return False
        
        # Start backtracking
        result = backtrack(0, 0, [0], set())
        if result:
            return rows
        return None
    
    def greedy_construct(n_rows, k, max_val, rng):
        """Simple greedy construction for baseline."""
        used = set()
        rows = []
        
        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                placed = False
                # Try values from smallest to largest
                for x in range(row[-1] + 1, max_val + 1):
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = x - a
                        if d > 0:
                            if d in used:
                                conflict = True
                                break
                            new_diffs.add(d)
                    if not conflict:
                        row.append(x)
                        used |= new_diffs
                        placed = True
                        break
                if not placed:
                    return None
            rows.append(row)
        
        return rows
    
    # Phase 1: Get a baseline with greedy
    baseline_max = max(lower_bound, 90)
    rows = greedy_construct(n_rows, k, baseline_max, random.Random(42))
    if rows is not None:
        max_val = max(max(r) for r in rows)
        if max_val < best_max:
            best_max = max_val
            best_rows = [r[:] for r in rows]
    
    # Phase 2: Try backtracking with various max_val values
    # Start from lower_bound and go up
    current_max = lower_bound
    max_backtrack_attempts = 50
    
    while current_max < best_max and time.time() < deadline:
        # Try a few backtracking attempts with this max_val
        for attempt in range(max_backtrack_attempts):
            if time.time() > deadline:
                break
            
            rng = random.Random(random.randint(0, 10**9))
            rows = backtracking_construct(n_rows, k, current_max, rng, time.time() + 2)
            if rows is not None:
                max_val = max(max(r) for r in rows)
                if max_val < best_max:
                    best_max = max_val
                    best_rows = [r[:] for r in rows]
                    # If we found a solution, we can try to improve further
                    break
        
        current_max += 1
    
    # Phase 3: Local search to improve the best solution
    if best_rows is not None and time.time() < deadline - 5:
        # Try to reduce the max entry by local search
        # For each row, try to find a smaller max by adjusting elements
        
        def local_search_improve(rows, max_iterations=1000):
            """Try to reduce the max entry via local search."""
            if time.time() > deadline - 5:
                return rows
            
            current_rows = [r[:] for r in rows]
            current_max = max(max(r) for r in current_rows)
            best_local = current_max
            best_local_rows = [r[:] for r in current_rows]
            
            used = set()
            for row in current_rows:
                used |= get_row_diffs(row)
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random row and try to modify it
                row_idx = random.randint(0, len(current_rows) - 1)
                row = current_rows[row_idx]
                
                # Try to reduce the max element of this row
                # Find the position of the max element
                max_pos = row.index(max(row))
                if max_pos == 0:
                    continue
                
                # Try replacing the max element with a smaller value
                old_max = row[max_pos]
                for new_val in range(row[max_pos - 1] + 1, old_max):
                    # Check if this creates conflicts
                    new_row = row[:]
                    new_row[max_pos] = new_val
                    
                    new_diffs = get_row_diffs(new_row)
                    other_diffs = set()
                    for i, r in enumerate(current_rows):
                        if i != row_idx:
                            other_diffs |= get_row_diffs(r)
                    
                    if not (new_diffs & other_diffs):
                        # Also check that new_row is strictly increasing
                        if all(new_row[i] < new_row[i+1] for i in range(len(new_row)-1)):
                            # Accept this change
                            current_rows[row_idx] = new_row
                            new_max = max(max(r) for r in current_rows)
                            if new_max < best_local:
                                best_local = new_max
                                best_local_rows = [r[:] for r in current_rows]
                            break
            
            return best_local_rows if best_local < current_max else current_rows
        
        improved_rows = local_search_improve(best_rows)
        if improved_rows is not None:
            new_max = max(max(r) for r in improved_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved_rows
    
    # Return the best solution found
    if best_rows is not None:
        return best_rows
    else:
        # Fallback: return a valid but potentially suboptimal solution
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
