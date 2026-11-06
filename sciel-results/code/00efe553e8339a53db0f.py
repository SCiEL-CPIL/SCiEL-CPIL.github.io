"""Rewritten program for the suite task; fresh design with incremental checks."""
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Greedy construction with incremental O(N^2) check per candidate
    # We maintain a set of all triple sums seen so far.
    # When adding a new candidate x, we only need to check sums involving x:
    #   x+x+x, x+x+y, x+y+y, x+y+z (for y,z in selected, y<=z)
    # This is O(N^2) per candidate instead of O(N^3) for full recomputation.
    
    selected = []
    triple_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        n = len(selected)
        
        # Check all new triple sums involving x
        # Case 1: x+x+x
        s = 3 * x
        if s in triple_sums:
            continue
        
        # Case 2: x+x+y for each y in selected
        conflict = False
        for y in selected:
            s = 2 * x + y
            if s in triple_sums:
                conflict = True
                break
        if conflict:
            continue
        
        # Case 3: x+y+y for each y in selected
        for y in selected:
            s = x + 2 * y
            if s in triple_sums:
                conflict = True
                break
        if conflict:
            continue
        
        # Case 4: x+y+z for each pair (y,z) in selected with y<=z
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in triple_sums:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        
        # All checks passed; add x and update triple_sums
        selected.append(x)
        
        # Add all new triple sums involving x
        triple_sums.add(3 * x)
        for y in selected[:-1]:
            triple_sums.add(2 * x + y)
            triple_sums.add(x + 2 * y)
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                triple_sums.add(x + y + z)
    
    return selected
