import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for a in range(1, n+1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e <= n:
                aps.append((a-1, b-1, c-1, e-1))
                
    num_aps = len(aps)
    
    # Precompute for each position, which APs it is in, and the other 3 positions
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            others = [x for x in ap if x != pos]
            pos_to_aps[pos].append((idx, others[0], others[1], others[2]))
            
    def check_ap_status(c1, c2, c3, new_c):
        # Returns True if violation (distinct < 3)
        # c1, c2, c3 are the other 3 colors. new_c is the color at the changed position.
        # Total colors: {c1, c2, c3, new_c}
        
        # Case 1: c1 == c2 == c3
        if c1 == c2 and c2 == c3:
            return True # Always violation (1 or 2 colors max)
            
        # Case 2: All distinct c1, c2, c3
        if c1 != c2 and c2 != c3 and c1 != c3:
            return False # Always valid (3 or 4 colors)
            
        # Case 3: Exactly two are equal
        # Subcase 3a: c1 == c2 != c3
        if c1 == c2:
            # Colors are A, A, B, new
            # Violation if new is A or B
            return new_c == c1 or new_c == c3
            
        # Subcase 3b: c1 == c3 != c2
        if c1 == c3:
            # Colors are A, B, A, new
            # Violation if new is A or B
            return new_c == c1 or new_c == c2
            
        # Subcase 3c: c2 == c3 != c1
        if c2 == c3:
            # Colors are A, B, B, new
            # Violation if new is A or B
            return new_c == c1 or new_c == c2
            
        # Should not reach here if logic above is exhaustive
        return False

    best_colors = None
    best_violations = float('inf')
    
    # 3 colors is impossible for N=128 (W(3,4)=75 implies mono AP exists)
    # So we start with 4 colors.
    
    for num_colors in [4, 5]:
        if time.time() > deadline - 10:
            break
            
        for restart in range(50):
            if time.time() > deadline - 2:
                break
                
            colors = [random.randint(0, num_colors-1) for _ in range(n)]
            
            # Initial violation count
            violations = 0
            ap_status = [False] * num_aps
            for i, ap in enumerate(aps):
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                # Use check_ap_status with one as new? No, just count distinct.
                # For init, simple set is fine.
                if len({c0, c1, c2, c3}) < 3:
                    violations += 1
                    ap_status[i] = True
                    
            if violations == 0:
                return colors
                
            # Local search
            for step in range(5000):
                if time.time() > deadline - 1:
                    break
                    
                if violations == 0:
                    return colors
                    
                pos = random.randint(0, n-1)
                old_color = colors[pos]
                
                best_delta = 0
                best_new_color = -1
                best_changes = None
                
                for new_color in range(num_colors):
                    if new_color == old_color:
                        continue
                        
                    delta = 0
                    changes = []
                    
                    for ap_idx, o1, o2, o3 in pos_to_aps[pos]:
                        was_violating = ap_status[ap_idx]
                        
                        c1 = colors[o1]
                        c2 = colors[o2]
                        c3 = colors[o3]
                        
                        new_violating = check_ap_status(c1, c2, c3, new_color)
                        
                        if was_violating and not new_violating:
                            delta -= 1
                        elif not was_violating and new_violating:
                            delta += 1
                            
                        changes.append((ap_idx, new_violating))
                        
                    if delta < best_delta:
                        best_delta = delta
                        best_new_color = new_color
                        best_changes = changes
                        
                if best_new_color != -1 and best_delta < 0:
                    colors[pos] = best_new_color
                    violations += best_delta
                    for ap_idx, new_v in best_changes:
                        ap_status[ap_idx] = new_v
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]
                        
                elif step % 50 == 0 and best_delta == 0:
                    # Allow sideways move to escape plateau
                    # Pick a random new color that doesn't increase violations
                    valid_moves = []
                    for new_color in range(num_colors):
                        if new_color == old_color: continue
                        delta = 0
                        for ap_idx, o1, o2, o3 in pos_to_aps[pos]:
                            was_violating = ap_status[ap_idx]
                            c1 = colors[o1]
                            c2 = colors[o2]
                            c3 = colors[o3]
                            new_violating = check_ap_status(c1, c2, c3, new_color)
                            if was_violating and not new_violating: delta -= 1
                            elif not was_violating and new_violating: delta += 1
                        if delta <= 0:
                            valid_moves.append((new_color, delta))
                            
                    if valid_moves:
                        new_color, delta = random.choice(valid_moves)
                        colors[pos] = new_color
                        violations += delta
                        # Update ap_status
                        for ap_idx, o1, o2, o3 in pos_to_aps[pos]:
                            c1 = colors[o1]
                            c2 = colors[o2]
                            c3 = colors[o3]
                            # Re-calc status for this AP
                            # We need the 4th color which is new_color
                            # check_ap_status(c1, c2, c3, new_color)
                            ap_status[ap_idx] = check_ap_status(c1, c2, c3, new_color)
                        
                        if violations < best_violations:
                            best_violations = violations
                            best_colors = colors[:]
                            
            if best_violations == 0:
                return best_colors
                
    if best_colors is not None:
        return best_colors
    return [x % 4 for x in range(n)]
