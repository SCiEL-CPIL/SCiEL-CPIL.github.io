import random

def construct():
    n = 128
    aps = []
    for i in range(n):
        for d in range(1, (n - i) // 3 + 1):
            if i + 3 * d < n:
                aps.append((i, i + d, i + 2 * d, i + 3 * d))
                
    def count_violations(colors):
        violations = 0
        for a, b, c, d in aps:
            # Inline check for speed
            c1 = colors[a]
            c2 = colors[b]
            c3 = colors[c]
            c4 = colors[d]
            # Check if at least 3 distinct
            # If len({c1,c2,c3,c4}) < 3, violation
            # This is equivalent to: 
            # Not (c1==c2==c3==c4) and not (c1==c2 and c3==c4 and c1!=c3) and ...
            # Simpler: 
            if c1 == c2:
                if c3 == c4:
                    if c1 == c3:
                        violations += 1 # 1 color
                    # else 2 colors (AA BB) -> violation
                    else:
                        violations += 1
                else:
                    if c1 == c3 or c1 == c4:
                        # A B B A or A B A B etc?
                        # If c1==c2, and c3!=c4.
                        # Colors: {c1, c3, c4}.
                        # If c3==c4 is false, we have c1, c3, c4.
                        # If c1==c3, colors are {c1, c4} -> 2 colors -> violation
                        # If c1==c4, colors are {c1, c3} -> 2 colors -> violation
                        # If c1!=c3 and c1!=c4, colors are {c1, c3, c4} -> 3 colors -> OK
                        if c1 == c3 or c1 == c4:
                            violations += 1
            else:
                # c1 != c2
                if c3 == c4:
                    # A B B B? No, c3==c4.
                    # Colors: {c1, c2, c3}.
                    # If c1==c3, colors {c1, c2} -> 2 -> violation
                    # If c2==c3, colors {c1, c2} -> 2 -> violation
                    # Else 3 colors -> OK
                    if c1 == c3 or c2 == c3:
                        violations += 1
                else:
                    # c1!=c2, c3!=c4
                    # Colors: {c1, c2, c3, c4}
                    # Violation if distinct count < 3.
                    # This happens if only 2 distinct colors are present.
                    # i.e., {c1,c2,c3,c4} is a subset of {x,y}
                    # Check if c1==c3 and c2==c4 (ABAB) -> 2 colors
                    # or c1==c4 and c2==c3 (ABBA) -> 2 colors
                    # or c1==c2 (already handled)
                    # or c3==c4 (already handled)
                    # So check:
                    if (c1 == c3 and c2 == c4) or (c1 == c4 and c2 == c3):
                        violations += 1
                    # Also cases like A A B C? No, c1!=c2.
                    # What about A B A A? c1=A, c2=B, c3=A, c4=A.
                    # c1!=c2. c3!=c4? No c3==c4. Handled above.
                    # What about A B C B? c1!=c2, c3!=c4? C!=B.
                    # Colors {A,B,C}. 3 colors. OK.
                    # What about A B C C? c3==c4. Handled.
                    # What about A B B C? c1!=c2. c3!=c4.
                    # c1==c3? A==B No. c1==c4? A==C No.
                    # c2==c3? B==B Yes.
                    # Wait, my logic for c1!=c2 and c3!=c4 only checked ABAB and ABBA.
                    # Let's just use set. It's safer.
                    pass

        # The manual logic is error prone. Let's use set but optimize the loop.
        # Given N=128, set creation is the bottleneck.
        # Let's stick to the set version but ensure it's the only thing we do.
        
        # Re-doing count_violations with set for correctness
        violations = 0
        for a, b, c, d in aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations

    def local_search(colors, max_iters=100):
        colors = list(colors)
        current_violations = count_violations(colors)
        if current_violations == 0:
            return colors, current_violations
        
        for _ in range(max_iters):
            improved = False
            # Try to improve
            # Pick a random position
            i = random.randint(0, n - 1)
            old_color = colors[i]
            
            best_new_color = old_color
            best_new_violations = current_violations
            
            for new_color in [0, 1, 2, 3]:
                if new_color == old_color:
                    continue
                colors[i] = new_color
                new_violations = count_violations(colors)
                if new_violations < best_new_violations:
                    best_new_violations = new_violations
                    best_new_color = new_color
                # Restore
            colors[i] = best_new_color
            current_violations = best_new_violations
            
            if best_new_color != old_color:
                improved = True
                
            if not improved:
                # Stuck, try swapping two random positions
                i1 = random.randint(0, n - 1)
                i2 = random.randint(0, n - 1)
                if i1 != i2:
                    c1, c2 = colors[i1], colors[i2]
                    colors[i1], colors[i2] = c2, c1
                    new_violations = count_violations(colors)
                    if new_violations < current_violations:
                        current_violations = new_violations
                        improved = True
                    else:
                        colors[i1], colors[i2] = c1, c2
                        # If stuck, break and restart
                        break
                        
            if current_violations == 0:
                break
                
        return colors, current_violations

    best_colors = None
    best_violations = float('inf')
    
    # 3-color search
    for attempt in range(200):
        colors = [random.randint(0, 2) for _ in range(n)]
        colors, viol = local_search(colors, max_iters=100)
        if viol < best_violations:
            best_violations = viol
            best_colors = colors
        if best_violations == 0:
            break
            
    if best_violations == 0:
        return best_colors
        
    # 4-color search if 3 fails
    for attempt in range(100):
        colors = [random.randint(0, 3) for _ in range(n)]
        colors, viol = local_search(colors, max_iters=100)
        if viol < best_violations:
            best_violations = viol
            best_colors = colors
        if best_violations == 0:
            break

    return best_colors
