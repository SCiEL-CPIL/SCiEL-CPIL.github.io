import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # pos_to_aps: for each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Precompute neighbor constraints for fast validity checks
    # For each position i, and for each AP it's part of, we need to know the other 3 positions.
    # But simpler: just iterate pos_to_aps[pos] and check the 4-tuple.
    
    # --- Step 1: Greedy Construction ---
    # We want to minimize colors. Greedy picks the smallest valid color.
    # Constraint: Every 4-AP must have >= 3 distinct colors.
    # This implies:
    # 1. No 4-AP can have < 3 colors.
    #    Forbidden: 1 color (all same), 2 colors (e.g., A,A,A,B or A,A,B,B).
    #    Actually, "at least 3 distinct" means the set size is 3 or 4.
    #    So size 1 and size 2 are forbidden.
    
    # Greedy approach:
    # Iterate positions 0 to 127.
    # For each position, try colors 0, 1, 2, ...
    # Check if assigning this color creates any invalid 4-AP ending at this position.
    # Note: 4-APs can end at position i if i = a + 3d.
    # Also, we must ensure that we don't create a situation where a 3-AP ending at i
    # becomes impossible to extend? 
    # The seed program checks:
    # 1. 4-APs ending at i: {c[a], c[a+d], c[a+2d], c[i]} must have size >= 3.
    # 2. 3-APs ending at i: {c[a], c[a+d], c[i]}. 
    #    Why check 3-APs? 
    #    If c[a] == c[a+d] == c[i], then any 4-AP (a, a+d, a+2d, a+3d) where a+2d is future?
    #    No, the constraint is on 4-APs.
    #    However, if we have a 3-AP a, a+d, a+2d all with color X, and we set c[a+2d]=X,
    #    does that force a violation?
    #    Consider a 4-AP: a, a+d, a+2d, a+3d.
    #    If c[a]=c[a+d]=c[a+2d]=X, then for the set {X, X, X, c[a+3d]} to have size >= 3,
    #    c[a+3d] must be Y and Z? No, size of {X, Y} is 2. 
    #    So if 3 elements in a 4-AP are the same, the 4th CANNOT make the size >= 3.
    #    Therefore, we must avoid creating any 3-term AP with all same colors, 
    #    because it will be part of a 4-term AP (a, a+d, a+2d, a+3d) where the first 3 are same.
    #    Wait, is every 3-AP part of a 4-AP?
    #    Yes, if a+3d < n. If a+3d >= n, it's not a valid 4-AP in the range.
    #    So we only need to worry about 3-APs that can be extended to a 4-AP within [0, n).
    #    Specifically, if we assign c[i] such that there exists d, a such that i = a+2d, 
    #    and c[a] == c[a+d] == c[i], AND a+3d < n, then the 4-AP (a, a+d, a+2d, a+3d) 
    #    has 3 same colors. The 4th color c[a+3d] can be anything, but the set will be {X, Y} max size 2.
    #    So this is a hard constraint: No 3-term AP that is a prefix of a valid 4-term AP can be monochromatic.
    
    greedy_colors = [-1] * n
    used_colors = set()
    
    for i in range(n):
        best_color = -1
        # Try existing colors first, then new ones? 
        # To minimize total colors, we should prefer existing colors.
        # But greedy usually tries 0, 1, 2...
        # Let's try colors in order 0..127.
        for color in range(128):
            valid = True
            
            # Check 4-term APs ending at i
            # i = a + 3d => a = i - 3d
            for d in range(1, (i // 3) + 1):
                a = i - 3 * d
                if a < 0: continue
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                c3 = greedy_colors[a + 2 * d]
                # All must be assigned
                if c1 == -1 or c2 == -1 or c3 == -1:
                    continue # Should not happen if we go left to right
                vals = {c1, c2, c3, color}
                if len(vals) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check 3-term APs ending at i that can be extended to a 4-term AP
            # i = a + 2d => a = i - 2d
            # Extension is at a + 3d = i + d
            # So we need i + d < n
            for d in range(1, (i // 2) + 1):
                a = i - 2 * d
                if a < 0: continue
                if i + d >= n:
                    continue # Cannot be extended to a 4-AP in range
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                if c1 == -1 or c2 == -1:
                    continue
                if c1 == color and c2 == color:
                    valid = False
                    break
            
            if valid:
                best_color = color
                break
        
        if best_color == -1:
            # Fallback: should not happen with 128 colors, but just in case
            best_color = 0
            
        greedy_colors[i] = best_color
        used_colors.add(best_color)
        
    num_greedy_colors = len(used_colors)
    
    # If greedy found a solution with 3 colors, it's optimal (min is 3).
    # Why 3? 2 colors is impossible for n=128? 
    # With 2 colors, any 4-AP must have >= 3 colors. Impossible.
    # So min is at least 3.
    if num_greedy_colors <= 3:
        return greedy_colors
        
    # If greedy used > 3 colors, we try to improve.
    # Strategy: Backtracking with dynamic variable ordering (MRV) and value ordering.
    # We start from the greedy solution and try to reduce colors, or start fresh with K=3.
    
    # Let's try to find a 3-coloring using backtracking.
    # If 3 fails, try 4?
    
    def solve_with_k_colors(k):
        if time.time() > deadline:
            return None
        
        colors = [-1] * n
        domains = [set(range(k)) for _ in range(n)]
        
        # Precompute constraints for fast checking
        # For each position, list of (ap_index, other_positions_tuple)
        
        def is_valid(pos, color):
            # Check all APs containing pos
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                vals = [colors[a], colors[b], colors[c], colors[d_pos]]
                # Replace pos with color
                # If pos == a: vals[0] = color
                # etc.
                # Simpler: create a list of 4 values
                v = [colors[a], colors[b], colors[c], colors[d_pos]]
                if pos == a: v[0] = color
                elif pos == b: v[1] = color
                elif pos == c: v[2] = color
                elif pos == d_pos: v[3] = color
                
                # Check if this assignment violates the constraint
                # If any value is -1, we can't fully check, but we can check if it's ALREADY impossible.
                # If all 4 are assigned, check size >= 3.
                # If 3 are assigned and they are all same, then the 4th MUST be different, but even then size is 2.
                # So if 3 are same, it's invalid.
                assigned = [x for x in v if x != -1]
                if len(assigned) == 4:
                    if len(set(assigned)) < 3:
                        return False
                elif len(assigned) == 3:
                    if len(set(assigned)) == 1:
                        return False
                # If 2 assigned:
                # If they are same (e.g., 0,0), then the other 2 must provide at least 2 new colors?
                # No, {0,0,x,y}. Size >= 3 implies {x,y} must have size 2 and both != 0? 
                # Or one is new and other is new?
                # If x=0, y=1, set is {0,1} size 2. Invalid.
                # If x=1, y=2, set is {0,1,2} size 3. Valid.
                # So if 2 are same, the other 2 MUST be distinct and different from the first.
                # This is a strong constraint for forward checking.
                # But for now, simple check is enough for backtracking.
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            # MRV: Find unassigned variable with smallest domain?
            # For simplicity, use sequential order 0..n-1.
            # But dynamic ordering is better.
            # Let's stick to sequential for now to avoid complex state management,
            # but use value ordering: try colors already used in neighbors?
            
            # Value ordering: try colors 0..k-1
            for color in range(k):
                colors[idx] = color
                if is_valid(idx, color):
                    if backtrack(idx + 1):
                        return True
                colors[idx] = -1
            
            return False
        
        # Start backtracking
        if backtrack(0):
            return colors
        return None

    # Try 3 colors
    sol = solve_with_k_colors(3)
    if sol is not None:
        return sol
        
    # If 3 fails, try 4?
    # Given time constraints, if 3 fails quickly, we might have time for 4.
    # But 3-coloring is likely possible. If not, 4 is.
    
    sol = solve_with_k_colors(4)
    if sol is not None:
        return sol
        
    # Fallback: return greedy solution
    return greedy_colors
