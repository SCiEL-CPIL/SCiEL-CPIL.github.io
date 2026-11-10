import time
import sys

sys.setrecursionlimit(10000)

def generate_golomb_rulers(k, max_span):
    rulers = []
    
    def backtrack(marks, used_diffs):
        if len(marks) == k + 1:
            max_mark = marks[-1]
            rulers.append((max_mark, used_diffs, tuple(marks)))
            return
        
        last = marks[-1]
        for next_mark in range(last + 1, max_span + 1):
            valid = True
            new_diffs = 0
            for prev in marks:
                d = next_mark - prev
                if (used_diffs >> d) & 1:
                    valid = False
                    break
                new_diffs |= (1 << d)
            
            if valid:
                backtrack(marks + [next_mark], used_diffs | new_diffs)
                
    backtrack([0], 0)
    return rulers

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate candidates
    max_gen_span = 150
    candidates = generate_golomb_rulers(k, max_gen_span)
    candidates.sort(key=lambda x: x[0])
    
    # If generation takes too long, we might have issues, but for k=6 it's fast.
    
    # Iterative deepening on span limit
    # Find min and max span in candidates
    if not candidates:
        # Fallback
        return _greedy_fallback(n_rows, k)
        
    min_span = candidates[0][0]
    max_span = candidates[-1][0]
    
    # We can skip spans that don't have enough candidates
    # But just iterating L is fine.
    
    for L in range(min_span, max_span + 1):
        if time.time() > deadline:
            break
            
        # Filter candidates with span <= L
        # Since sorted, we can just take a prefix
        # But we need to check which ones are <= L
        # Binary search or simple loop?
        # Simple loop is fine for small N.
        
        valid_cands = [c for c in candidates if c[0] <= L]
        
        if len(valid_cands) < n_rows:
            continue
            
        # Search for any packing
        result = _search_packing(valid_cands, n_rows, deadline)
        if result:
            return result
            
    # If no solution found in time or within limit
    return _greedy_fallback(n_rows, k)

def _search_packing(candidates, n_rows, deadline):
    # Backtracking to find ANY solution
    # candidates is sorted by span
    
    def backtrack(idx, used_diffs, current_rows):
        if time.time() > deadline:
            return None
            
        if len(current_rows) == n_rows:
            return current_rows
        
        for i in range(idx, len(candidates)):
            row = candidates[i]
            diff_bitset = row[1]
            
            if diff_bitset & used_diffs:
                continue
                
            res = backtrack(i + 1, used_diffs | diff_bitset, current_rows + [row])
            if res:
                return res
        return None
        
    return backtrack(0, 0, [])

def _greedy_fallback(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            valid = True
            for a in row:
                d = x - a
                if (used >> d) & 1:
                    valid = False
                    break
                new |= (1 << d)
            if valid:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
