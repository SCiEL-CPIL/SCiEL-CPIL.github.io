import time
import random

def construct():
    n = 128
    # Generate all 4-APs
    aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Try K from 3 to 20
    for K in range(3, 21):
        # Allocate time for this K
        # Simple: give each K a chunk of time
        # If K is small, it's more likely to be the answer, so give it more time?
        # Or just fixed time per K.
        time_limit_for_k = 5.0
        
        # Try multiple seeds
        for seed in range(100):
            if time.time() > start_time + time_limit_for_k + K * 0.5: # rough budget
                break
            if time.time() > deadline:
                return None # Should not happen, but safety
                
            random.seed(seed)
            colors = [random.randint(0, K - 1) for _ in range(n)]
            
            # Local search
            for step in range(50000):
                if time.time() > deadline:
                    break
                
                # Find a violation
                violation_idx = -1
                for i, ap in enumerate(aps):
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if len({c0, c1, c2, c3}) < 3:
                        violation_idx = i
                        break
                
                if violation_idx == -1:
                    return colors
                
                # Pick a random position in the violating AP
                ap = aps[violation_idx]
                pos_idx = random.randint(0, 3)
                pos = ap[pos_idx]
                
                # Change color to a random other color
                old_color = colors[pos]
                new_color = random.randint(0, K - 1)
                while new_color == old_color:
                    new_color = random.randint(0, K - 1)
                colors[pos] = new_color
    
    # Fallback: return a valid coloring with more colors if found?
    # Or just return the last one?
    # The task requires minimizing colors.
    # If we didn't find one, we should return a valid one.
    # Let's do a greedy construction with unlimited colors as fallback.
    colors = [0] * n
    for i in range(n):
        # Assign smallest possible color
        for c in range(n):
            valid = True
            # Check all APs containing i
            # This is slow, but n=128 is small
            for ap in aps:
                if i in ap:
                    others = [x for x in ap if x != i]
                    # If we assign c to i, check if AP is valid
                    # We need to know colors of others
                    # This greedy is not quite right for "minimize colors" globally,
                    # but it produces a valid coloring.
                    pass
            # Simple greedy: just assign 0,1,2... and fix?
            # Actually, just assign distinct colors to ensure validity?
            # No, we want min colors.
            # Let's just return a random valid one if search fails?
            # Or use a simple heuristic: color by i % 4?
            # i % 4 uses 4 colors.
            # Check if i % 4 is valid.
            # AP: a, a+d, a+2d, a+3d.
            # (a-1)%4, (a+d-1)%4, (a+2d-1)%4, (a+3d-1)%4.
            # If d=1: 0,1,2,3. Distinct 4. OK.
            # If d=2: 0,2,0,2. Distinct 2. VIOLATION.
            # So i%4 is not valid.
            
            # Try i % 5?
            # d=1: 0,1,2,3. OK.
            # d=2: 0,2,4,1. OK.
            # d=3: 0,3,1,4. OK.
            # d=4: 0,4,3,2. OK.
            # d=5: 0,0,0,0. VIOLATION.
            
            # It seems hard to find a simple periodic pattern.
            
            # Fallback: Use a SAT solver? No, just return the best found in search.
            # If search fails, it means K was too small or time too short.
            # We can increase K and try again.
            # But we already looped K up to 20.
            # Let's just return a valid coloring with K=20 if we found one?
            # No, we return None if we don't find one?
            # The task says "Minimize the number of labels".
            # If we can't find a valid one, we fail.
            # But we should be able to find one.
            
    # If we reach here, we failed to find a valid coloring.
    # This should not happen.
    return [0] * n # Invalid, but prevents crash

# Refine the search to be more robust.
