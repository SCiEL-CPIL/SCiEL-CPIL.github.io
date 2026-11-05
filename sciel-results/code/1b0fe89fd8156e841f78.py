"""Difference triangle set (7,5) solver using greedy with backtracking and cost-based assignment."""
import sys
import time
import random

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within each row are distinct across all rows.
    
    Uses a multi-strategy approach with randomized restarts and cost-based heuristics
    (penalizing small differences to preserve them for later, i.e., preferring large diffs).
    """
    
    def find_row_with_heuristic(used_global, max_span_limit, prefer_large=False, randomize=False):
        """
        Find a row of k+1 integers starting at 0 with span <= max_span_limit.
        prefer_large: If True, iterate x from max to min (penalize small diffs).
        randomize: If True, shuffle valid candidates.
        """
        row = [0]
        
        def backtrack(row, used_local):
            if len(row) == k + 1:
                return row[:]
            
            last = row[-1]
            remaining = k + 1 - len(row)
            if last + remaining > max_span_limit:
                return None
            
            max_next = max_span_limit - (remaining - 1)
            
            # Generate candidate range
            candidates = list(range(last + 1, max_next + 1))
            
            if prefer_large:
                candidates.reverse()
            elif randomize:
                random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_global or d in used_local:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                row.append(x)
                result = backtrack(row, used_local | new_diffs)
                if result is not None:
                    return result
                row.pop()
            
            return None
        
        return backtrack(row, set())

    def solve_with_strategy(prefer_large=False, randomize=False, seed=0):
        """
        Attempt to construct all rows using the specified heuristic.
        Returns list of rows or None if failed.
        """
        random.seed(seed)
        used = set()
        rows = []
        
        for r in range(n_rows):
            found = False
            # Try spans in increasing order
            # Upper bound estimation: counting bound is 105, known best 112.
            # We can search up to 150 to be safe.
            for span in range(k, 150):
                result = find_row_with_heuristic(used, span, prefer_large, randomize)
                if result is not None:
                    # Update used
                    for i in range(len(result)):
                        for j in range(i + 1, len(result)):
                            used.add(result[j] - result[i])
                    rows.append(result)
                    found = True
                    break
            
            if not found:
                return None
        
        return rows

    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Original (Smallest first)
    rows = solve_with_strategy(prefer_large=False, randomize=False, seed=42)
    if rows:
        scope = max(row[-1] for row in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Strategy 2: Prefer Large (Penalize small)
    rows = solve_with_strategy(prefer_large=True, randomize=False, seed=42)
    if rows:
        scope = max(row[-1] for row in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
    # Strategy 3: Randomized Smallest
    for i in range(10):
        rows = solve_with_strategy(prefer_large=False, randomize=True, seed=i)
        if rows:
            scope = max(row[-1] for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
                
    # Strategy 4: Randomized Largest
    for i in range(10):
        rows = solve_with_strategy(prefer_large=True, randomize=True, seed=100+i)
        if rows:
            scope = max(row[-1] for row in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows

    if best_rows:
        return best_rows
        
    # Fallback: simple greedy (should not be reached if above works)
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
