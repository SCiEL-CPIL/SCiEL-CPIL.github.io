import time
import random
import sys
from itertools import combinations

def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i):
            diffs.add(row[i] - row[j])
    return diffs

def _check_valid(rows, k):
    """Verify that all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for d in _row_diffs(row):
            if d in all_diffs:
                return False
            all_diffs.add(d)
    return True

def _generate_row_patterns(k, M, deadline):
    """
    Generate all valid row patterns (subsets of {0,...,M} of size k+1 containing 0)
    that are internally consistent (all pairwise differences distinct within the row).
    Returns list of (pattern_tuple, frozenset_of_diffs).
    """
    patterns = []
    
    # We need to choose k elements from {1, ..., M} to complete the row [0, ...]
    # The row must have all pairwise differences distinct (Sidon-like property)
    
    def is_valid_pattern(subset):
        """Check if {0} ∪ subset has all pairwise differences distinct."""
        row = [0] + sorted(subset)
        diffs = set()
        for i in range(len(row)):
            for j in range(i):
                d = row[i] - row[j]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    # Generate combinations of k elements from {1, ..., M}
    # This can be expensive, so we use a heuristic approach
    # For small M and k, enumerate; for larger, use random sampling
    
    if M <= 30:
        # Full enumeration for small M
        for combo in combinations(range(1, M + 1), k):
            if time.time() > deadline:
                break
            if is_valid_pattern(combo):
                row = (0,) + combo
                diffs = frozenset(_row_diffs(list(row)))
                patterns.append((row, diffs))
    else:
        # Random sampling for larger M
        rng = random.Random(12345)
        seen = set()
        attempts = 0
        max_attempts = 50000
        while len(patterns) < 2000 and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            combo = tuple(sorted(rng.sample(range(1, M + 1), k)))
            if combo in seen:
                continue
            seen.add(combo)
            if is_valid_pattern(combo):
                row = (0,) + combo
                diffs = frozenset(_row_diffs(list(row)))
                patterns.append((row, diffs))
    
    return patterns

def _dsatur_coloring(patterns, n_rows, deadline):
    """
    Use DSATUR-style greedy coloring to select n_rows independent sets (rows)
    that don't share any difference values.
    
    patterns: list of (pattern_tuple, frozenset_of_diffs)
    Returns: list of selected patterns (tuples) or None if not enough can be selected
    """
    if len(patterns) < n_rows:
        return None
    
    n = len(patterns)
    # Build conflict graph: edge between i and j if their diff sets intersect
    # For efficiency, use a dictionary mapping diff value to list of pattern indices
    
    diff_to_patterns = {}
    for idx, (pat, diffs) in enumerate(patterns):
        for d in diffs:
            if d not in diff_to_patterns:
                diff_to_patterns[d] = []
            diff_to_patterns[d].append(idx)
    
    # Conflict matrix (sparse representation)
    # conflict[i] = set of indices that conflict with i
    conflict = [set() for _ in range(n)]
    for d, pats in diff_to_patterns.items():
        if len(pats) > 1:
            for i in range(len(pats)):
                for j in range(i + 1, len(pats)):
                    conflict[pats[i]].add(pats[j])
                    conflict[pats[j]].add(pats[i])
    
    # DSATUR: maintain saturation degree (number of different "colors" adjacent)
    # Here, "color" = selected row. We want to pick n_rows mutually non-conflicting patterns.
    # This is equivalent to finding an independent set of size n_rows.
    
    # Greedy approach: repeatedly pick the pattern with minimum conflicts among unselected,
    # prioritizing those that conflict with fewest remaining candidates.
    
    selected = []
    unselected = set(range(n))
    
    for _ in range(n_rows):
        if time.time() > deadline:
            return None
        
        if not unselected:
            return None
        
        # Score each candidate: prefer those with fewer conflicts to remaining unselected
        # and lower max entry (to minimize scope)
        best_idx = None
        best_score = None
        
        for idx in unselected:
            # Count conflicts with other unselected
            conflicts_with_unselected = len(conflict[idx] & unselected)
            # Prefer fewer conflicts
            # Also prefer smaller max entry
            max_entry = patterns[idx][0][-1]
            score = (conflicts_with_unselected, max_entry)
            
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is None:
            return None
        
        selected.append(best_idx)
        unselected.discard(best_idx)
        # Remove all patterns that conflict with the selected one
        for c in conflict[best_idx]:
            unselected.discard(c)
    
    return [patterns[i][0] for i in selected]

def _local_improve(rows, k, deadline):
    """
    Try to reduce the maximum entry by locally modifying rows.
    For each row, try to replace it with a pattern that has a smaller max entry
    while maintaining validity.
    """
    if time.time() > deadline:
        return rows
    
    used_diffs = set()
    for row in rows:
        used_diffs |= _row_diffs(list(row))
    
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and iteration < max_iterations and time.time() < deadline:
        improved = False
        iteration += 1
        
        for r_idx in range(len(rows)):
            if time.time() > deadline:
                return rows
            
            current_max = max(rows[r_idx])
            if current_max == 0:
                continue
            
            # Try to find a replacement row with smaller max
            # Generate candidate rows with max < current_max
            # This is expensive, so we use a limited search
            
            # Remove current row's diffs temporarily
            current_diffs = _row_diffs(list(rows[r_idx]))
            other_diffs = used_diffs - current_diffs
            
            # Try random replacements
            rng = random.Random(hash(tuple(rows[r_idx])) + iteration)
            for attempt in range(20):
                if time.time() > deadline:
                    return rows
                
                # Generate a random valid row with max < current_max
                max_allowed = current_max - 1
                if max_allowed < k:
                    break
                
                # Random subset of size k from {1, ..., max_allowed}
                subset = tuple(sorted(rng.sample(range(1, max_allowed + 1), k)))
                candidate_row = (0,) + subset
                
                # Check internal validity
                cand_diffs = set()
                valid = True
                for i in range(len(candidate_row)):
                    for j in range(i):
                        d = candidate_row[i] - candidate_row[j]
                        if d in cand_diffs:
                            valid = False
                            break
                        cand_diffs.add(d)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Check no conflict with other rows
                if cand_diffs & other_diffs:
                    continue
                
                # Accept the improvement
                rows[r_idx] = list(candidate_row)
                used_diffs = other_diffs | cand_diffs
                improved = True
                break
    
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Theoretical lower bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    best_solution = None
    best_M = float('inf')
    
    # Search M from lower bound to known best + margin
    M_start = max(total_diffs_needed, 112)
    M_end = 135
    
    # Strategy 1: DSATUR-based graph coloring approach
    for M in range(M_start, M_end + 1):
        if time.time() > deadline:
            break
        
        # Generate valid row patterns within scope M
        patterns = _generate_row_patterns(k, M, deadline)
        
        if len(patterns) < n_rows:
            continue
        
        # Use DSATUR-style independent set selection
        selected = _dsatur_coloring(patterns, n_rows, deadline)
        
        if selected:
            rows = [list(p) for p in selected]
            if _check_valid(rows, k):
                max_entry = max(max(row) for row in rows)
                if max_entry <= M:
                    # Try to improve locally
                    rows = _local_improve(rows, k, deadline)
                    if _check_valid(rows, k):
                        final_max = max(max(row) for row in rows)
                        if final_max <= best_M:
                            best_solution = rows
                            best_M = final_max
                        return rows
    
    # Fallback: simple greedy without M constraint
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
