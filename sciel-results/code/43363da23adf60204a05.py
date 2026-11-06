import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    required = set(range(1, 31))
    
    # Pre-compute: for each candidate x in 1..100, which diffs it creates with 0
    # and which other candidates it conflicts with
    
    MAX_D = 100
    
    # For a Sidon set {0, x1, ..., x8}, all pairwise differences must be distinct
    # x_j - x_i for j > i, and x_i - 0 = x_i
    
    # Strategy: Backtrack choosing x1, x2, ..., x8
    # At each step, maintain set of used differences
    # Prune: if we can't cover all required diffs, stop
    
    def solve_for_D(D, time_limit):
        """Find {0, x1..x7, D} that is Sidon and covers 1..30"""
        # x8 = D, so we need to find x1..x7 in 1..D-1
        # Diffs involving D: D-x1, D-x2, ..., D-x7, D-0=D
        # Diffs among x1..x7: xj-xi for j>i
        # Diffs involving 0: x1, x2, ..., x7, D
        
        # Total diffs: 7 (from 0) + 7 (from D) + C(7,2)=21 (among x1..x7) = 35
        # Plus D-0=D, so 36 total. All must be distinct.
        
        used = set()
        elements = [0]
        
        def backtrack(idx, remaining_required):
            if time.time() > time_limit:
                return None
            
            if idx == 8:  # We've chosen x1..x7, now add D
                # Check D doesn't conflict
                for e in elements:
                    d = D - e
                    if d in used:
                        return None
                # Check coverage
                final_diffs = used | {D - e for e in elements} | {D}
                if required.issubset(final_diffs):
                    return elements + [D]
                return None
            
            # Choose x_idx from elements[-1]+1 to D-1
            # But we need room for remaining elements
            start = elements[-1] + 1
            end = D - (8 - idx)  # Leave room for remaining
            
            if start > end:
                return None
            
            for c in range(start, end + 1):
                # Check Sidon constraint
                new_diffs = set()
                valid = True
                for e in elements:
                    d = c - e
                    if d in used or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Forward checking: can we still cover required diffs?
                # This is complex, so skip for now
                
                elements.append(c)
                used.update(new_diffs)
                
                result = backtrack(idx + 1, required - new_diffs)
                
                elements.pop()
                used.difference_update(new_diffs)
                
                if result is not None:
                    return result
            
            return None
        
        return backtrack(1, required)
    
    # Try D from lower bound upward
    # Lower bound: need 36 distinct diffs, max diff is D
    # Minimum D such that C(D+1,2) >= 36 and we can cover 1..30
    # C(D+1,2) = D(D+1)/2 >= 36 => D >= 8
    # But we also need to cover 1..30, so D >= 30
    # Actually, the diameter must be at least 30 since diff 30 must appear
    
    best_solution = None
    
    for D in range(30, 100):
        if time.time() > deadline - 10:
            break
        
        sol = solve_for_D(D, deadline - 15)
        if sol is not None:
            best_solution = sol
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy construction
    a = [0]
    used = set()
    
    for _ in range(8):
        best_x = None
        for x in range(1, 500):
            if x in a:
                continue
            new_diffs = set()
            valid = True
            for v in a:
                d = abs(x - v)
                if d in used or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                best_x = x
                break
        
        if best_x is None:
            # Add large element
            x = max(a) + 31
            while True:
                new_diffs = set()
                valid = True
                for v in a:
                    d = x - v
                    if d in used or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    best_x = x
                    break
                x += 1
        
        a.append(best_x)
        a.sort()
        for v in a[:-1]:
            used.add(best_x - v)
    
    # Verify and return
    return a

