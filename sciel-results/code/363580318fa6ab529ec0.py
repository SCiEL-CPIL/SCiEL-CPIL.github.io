import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s margin
    
    n = 128
    k = 4
    
    # Precompute all 4-APs
    aps = []
    for d in range(1, n//3 + 1):
        for a in range(1, n - 3*d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
            
    # Backtracking search
    colors = [0] * (n + 1)  # 1-indexed
    colors[0] = -1 # unused
    
    # Symmetry breaking: fix first few colors
    # c[1]=0, c[2]=1, c[3]=2, c[4]=3 ?
    # Let's just fix c[1]=0 to break color permutation symmetry.
    colors[1] = 0
    
    # Forward checking
    # For each position i, check if assigning color c is valid with respect to previous positions.
    
    def is_valid(i, c):
        # Check all 4-APs involving i and previously assigned positions
        # An AP (a, a+d, a+2d, a+3d) involves i if i is one of the terms.
        # We only need to check APs where the other 3 terms are already assigned.
        
        # Case 1: i is the last term (a+3d)
        # a+3d = i => a = i-3d
        # Terms: i-3d, i-2d, i-d, i
        for d in range(1, (i-1)//3 + 1):
            a = i - 3*d
            if a < 1: continue
            t1, t2, t3, t4 = a, a+d, a+2*d, i
            # t1, t2, t3 are assigned
            if colors[t1] != -1 and colors[t2] != -1 and colors[t3] != -1:
                cols = {colors[t1], colors[t2], colors[t3], c}
                if len(cols) < 3:
                    return False
        
        # Case 2: i is the 3rd term (a+2d)
        # a+2d = i => a = i-2d
        # Terms: i-2d, i-d, i, i+d
        # i+d is not assigned yet. So we can't check the full AP.
        # We can only check if the partial assignment is consistent?
        # No, we must wait until all 4 are assigned.
        # However, we can check if it's possible to complete.
        # For now, we only check APs where all 4 are assigned.
        # This is less effective for pruning.
        
        # To improve pruning, we can check if the partial set of colors in an AP
        # already violates the condition.
        # E.g., if we have 3 terms assigned and they are all color 0,
        # and the 4th is unassigned, we can assign it anything, but the AP will have <3 colors?
        # No, if 3 are 0, the 4th can be 1, making {0,1} size 2. FAIL.
        # So if 3 are 0, the 4th MUST be different from 0? No, even if 4th is 1, size is 2.
        # So if 3 are 0, it's IMPOSSIBLE to satisfy "at least 3 colors".
        # So we can prune if any 3 assigned terms in an AP are the same color.
        
        # Let's add this check.
        for d in range(1, (i-1)//3 + 1):
            # i is 4th
            a = i - 3*d
            if a < 1: continue
            t1, t2, t3 = a, a+d, a+2*d
            if colors[t1] != -1 and colors[t2] != -1 and colors[t3] != -1:
                if colors[t1] == colors[t2] == colors[t3]:
                    return False
                # Also if {t1,t2,t3} has size 1, covered above.
                # If {t1,t2,t3} has size 2, e.g., 0,0,1.
                # Then 4th must be 2 or 3 to make size 3.
                # So c must be in {2,3} if k=4.
                # This is a constraint on c, not a direct prune unless c is fixed.
                
        # Check if i is 3rd term
        for d in range(1, (i-1)//2 + 1):
            a = i - 2*d
            if a < 1: continue
            t1, t2, t3 = a, a+d, i
            # t1, t2 are assigned. t4 = i+d is unassigned.
            if colors[t1] != -1 and colors[t2] != -1:
                # If t1, t2, i are all same, then t4 must be different.
                # But even if t4 is different, size is 2. FAIL.
                if colors[t1] == colors[t2] == c:
                    return False
                # If t1, t2 are same, and i is different?
                # e.g., 0,0,1. t4 must be 2 or 3.
                # This is a constraint on t4.
                
        # Check if i is 2nd term
        for d in range(1, (i-1)//1 + 1):
            a = i - d
            if a < 1: continue
            t1, t2 = a, i
            # t1 assigned. t3, t4 unassigned.
            # No strong prune yet.
            
        return True

    def backtrack(i):
        if time.time() > deadline:
            return False
        if i > n:
            return True
            
        for c in range(k):
            if is_valid(i, c):
                colors[i] = c
                if backtrack(i + 1):
                    return True
                colors[i] = -1
        return False

    if backtrack(2):
        return colors[1:n+1]
        
    # Fallback to greedy if backtracking fails or times out
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check all 4-APs ending at x
            for d in range(1, (x-1)//3 + 1):
                a = x - 3*d
                if a < 1: continue
                t1, t2, t3 = a, a+d, a+2*d
                if len({color, colors[t1-1], colors[t2-1], colors[t3-1]}) < 3:
                    valid = False
                    break
            if not valid: continue
            # Check all 4-APs where x is 3rd term
            for d in range(1, (x-1)//2 + 1):
                a = x - 2*d
                if a < 1: continue
                t1, t2 = a, a+d
                # t4 = x+d is not assigned yet.
                # We can't check the full constraint.
                # But we can check if the current 3 terms force a violation?
                # If t1, t2, x are all same, then t4 must be different, but size will be 2.
                if colors[t1-1] == colors[t2-1] == color:
                    valid = False
                    break
            if not valid: continue
            colors.append(color)
            break
        else:
            # Should not happen if k is large enough
            colors.append(0)
            
    return colors
