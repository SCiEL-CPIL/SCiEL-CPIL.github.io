"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _all_4tp(n):
    """Return list of (a, d) for all nonconstant 4-term progressions in [1..n]."""
    res = []
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            res.append((a, d))
    return res

def _check_validity(colors, n):
    """Check if every 4TP in [1..n] uses at least 3 distinct colors."""
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            c0 = colors[a - 1]
            c1 = colors[a + d - 1]
            c2 = colors[a + 2 * d - 1]
            c3 = colors[a + 3 * d - 1]
            if len({c0, c1, c2, c3}) < 3:
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = 128
    
    # Precompute all 4TPs for quick checking
    # For each position x (1-indexed), list of (a, d) such that x is one of the 4 terms
    # Actually, we need to check all 4TPs globally. Let's precompute for incremental checking.
    
    # For each position x, which 4TPs does it participate in?
    # A 4TP is (a, a+d, a+2d, a+3d) for a>=1, d>=1, a+3d<=n
    # Position x can be at index 0,1,2,3 in the progression.
    
    # Let's build a list of all 4TPs as tuples of 4 positions (0-indexed internally)
    all_4tps = []
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            all_4tps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    # For each position, list of 4TP indices that include it
    pos_to_4tps = [[] for _ in range(n)]
    for idx, tps in enumerate(all_4tps):
        for pos in tps:
            pos_to_4tps[pos].append(idx)
    
    # Greedy construction with backtracking for small k
    # Try k = 2, 3, 4, ... colors
    # With k=2: every 4TP must have at least 3 distinct colors -> impossible with 2 colors
    # So minimum is at least 3.
    
    # Try k=3 first
    # Use a randomized greedy with restarts
    
    best_solution = None
    best_k = 128  # fallback
    
    for k in range(3, 128):
        if time.time() > deadline - 5:
            break
        if best_solution is not None:
            break
            
        found = False
        max_restarts = 50 if k <= 4 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
                
            colors = [-1] * n
            colors[0] = restart % k
            success = True
            
            for x in range(1, n):
                if time.time() > deadline - 2:
                    success = False
                    break
                    
                # Try colors in random order
                color_order = list(range(k))
                random.shuffle(color_order)
                
                placed = False
                for c in color_order:
                    # Check if placing c at position x violates any 4TP
                    # A 4TP is violated if all 4 positions are assigned and <3 distinct colors
                    # Since we're filling left to right, when we place at x,
                    # any 4TP that includes x and has all other positions already assigned
                    # must be checked.
                    
                    valid = True
                    for tp_idx in pos_to_4tps[x]:
                        tp = all_4tps[tp_idx]
                        # Check if all other positions in this 4TP are assigned
                        all_assigned = True
                        distinct = {c}
                        for pos in tp:
                            if pos == x:
                                continue
                            if colors[pos] == -1:
                                all_assigned = False
                                break
                            distinct.add(colors[pos])
                        if all_assigned:
                            if len(distinct) < 3:
                                valid = False
                                break
                    
                    if valid:
                        colors[x] = c
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                # Verify full validity
                if _check_validity(colors, n):
                    best_solution = colors
                    best_k = k
                    found = True
                    break
    
    if best_solution is None:
        # Fallback: use a simple periodic pattern that should work with more colors
        # Pattern: repeat [0, 1, 2, 0, 1, 2, ...] - this uses 3 colors
        # But we need to verify it works. If not, use more colors.
        # Actually, let's just use a pattern with enough colors to guarantee validity.
        # With k=4: pattern [0,1,2,3,0,1,2,3,...]
        # Any 4TP (a, a+d, a+2d, a+3d): the colors are c[a%4], c[(a+d)%4], etc.
        # If d%4==0, all same color -> only 1 distinct -> invalid!
        # So simple periodic won't work for all d.
        
        # Use a pattern based on binary representation or just use 4 colors with a different pattern.
        # Safest fallback: assign colors based on x % 4, but verify.
        # Actually, let's just do a final greedy with k=4,5,6... until it works.
        for k in range(4, 128):
            if time.time() > deadline - 1:
                break
            for restart in range(10):
                colors = [-1] * n
                colors[0] = restart % k
                success = True
                for x in range(1, n):
                    if time.time() > deadline - 0.5:
                        success = False
                        break
                    color_order = list(range(k))
                    random.shuffle(color_order)
                    placed = False
                    for c in color_order:
                        valid = True
                        for tp_idx in pos_to_4tps[x]:
                            tp = all_4tps[tp_idx]
                            all_assigned = True
                            distinct = {c}
                            for pos in tp:
                                if pos == x:
                                    continue
                                if colors[pos] == -1:
                                    all_assigned = False
                                    break
                                distinct.add(colors[pos])
                            if all_assigned:
                                if len(distinct) < 3:
                                    valid = False
                                    break
                        if valid:
                            colors[x] = c
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                if success and _check_validity(colors, n):
                    best_solution = colors
                    best_k = k
                    break
            if best_solution is not None:
                break
    
    if best_solution is None:
        # Ultimate fallback: use all 128 colors, each position gets its own color
        # This trivially satisfies the condition since all 4 terms have distinct colors
        best_solution = list(range(128))
    
    return best_solution

# EVOLVE-BLOCK-END
