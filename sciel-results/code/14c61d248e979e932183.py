"""Difference triangle set (7,5) - Efficient backtracking CSP solver."""
import time

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    # Greedy fallback
    def greedy():
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
    
    best = greedy()
    best_scope = max(max(r) for r in best)
    
    # Backtracking solver with forward checking
    def solve(M, time_limit):
        nonlocal best, best_scope
        
        if M >= best_scope:
            return False
        
        used = set()
        rows = []
        
        def bt(ri, row):
            nonlocal used, rows
            
            if time.time() > time_limit:
                return False
            
            if ri == n_rows:
                sc = max(max(r) for r in rows)
                if sc < best_scope:
                    best_scope = sc
                    best = [list(r) for r in rows]
                return True
            
            if len(row) == k + 1:
                rows.append(list(row))
                r = bt(ri + 1, [0])
                if r:
                    return True
                rows.pop()
                return False
            
            mn = row[-1] + 1
            mx = M
            
            for v in range(mn, mx + 1):
                if time.time() > time_limit:
                    return False
                
                ok = True
                nd = []
                for p in row:
                    d = v - p
                    if d in used:
                        ok = False
                        break
                    nd.append(d)
                
                if not ok:
                    continue
                
                # Simple forward check: enough room for remaining values
                rem = k + 1 - len(row) - 1
                if rem > M - v:
                    continue
                
                row.append(v)
                for d in nd:
                    used.add(d)
                
                r = bt(ri, row)
                if r:
                    return True
                
                for d in nd:
                    used.discard(d)
                row.pop()
            
            return False
        
        return bt(0, [0])
    
    # Try to find solution with scope less than best_scope
    # Start from a reasonable lower bound
    lower = max(50, best_scope - 30)
    
    for M in range(lower, best_scope):
        if time.time() > deadline - 5:
            break
        if solve(M, deadline - 5):
            break
    
    return best
