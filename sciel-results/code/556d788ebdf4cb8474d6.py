import time
import random
import math
from collections import defaultdict
from itertools import combinations

def _normalize_line(A, B, C):
    if A == 0 and B == 0:
        return (0, 0, 1) if C != 0 else (0, 0, 0)
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0:
        if B < 0:
            B = -B
            C = -C
        elif B == 0:
            if C < 0:
                C = -C
    return (A, B, C)

def _line_from_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return _normalize_line(A, B, C)

def _line_from_3_points(p1, p2, p3):
    # Check collinearity first
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if (x2 - x1) * (y3 - y1) != (y2 - y1) * (x3 - x1):
        return None
    return _line_from_points(p1, p2)

def _count_4_lines(points):
    """
    Returns (count_4, is_valid, line_map)
    line_map: dict mapping line_key -> set of point_indices on that line
    """
    n = len(points)
    line_map = defaultdict(set)
    
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_from_points(points[i], points[j])
            line_map[key].add(i)
            line_map[key].add(j)
            
    count_4 = 0
    is_valid = True
    for key, indices in line_map.items():
        cnt = len(indices)
        if cnt == 4:
            count_4 += 1
        elif cnt >= 5:
            is_valid = False
    return count_4, is_valid, line_map

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial Configuration: 4x4 Grid
    # This is a strong seed.
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    
    current_points = [p[:] for p in initial_points]
    current_score, current_valid, current_line_map = _count_4_lines(current_points)
    
    best_points = [p[:] for p in current_points]
    best_score = current_score
    
    if not current_valid:
        # Fallback if initial is invalid (should not happen for 4x4 grid)
        return best_points

    random.seed(42)
    
    # Precompute all potential lines defined by any 3 points?
    # No, we will generate them on the fly for mutations.
    
    # Strategy:
    # 1. Identify current 4-lines.
    # 2. Pick a random 4-line L.
    # 3. Pick a point P from L to move.
    # 4. Pick 3 other points Q, R, S not including P.
    # 5. Check if Q, R, S are collinear. If so, let M be the line.
    # 6. Check if moving P to a position on M (distinct from Q,R,S) creates a valid state.
    #    Actually, P must be placed on M. But P is a specific point.
    #    Wait, we are moving P to a NEW position P'.
    #    P' must be on a line formed by 3 other points?
    #    Or P' must be such that it completes a 4-line with 3 existing points?
    
    # Let's refine the mutation:
    # "Line-Swap" Mutation:
    # 1. Select a point P to move.
    # 2. Select 3 other points A, B, C.
    # 3. If A, B, C are collinear, let L_new be the line.
    # 4. We want to place P' on L_new such that P' is an integer point, distinct from others,
    #    and ideally creates a new 4-line (i.e., P' is the 4th point on L_new).
    #    But L_new already has A, B, C. If we place P' on L_new, L_new will have 4 points.
    #    We must ensure no other point lies on L_new (else 5 collinear).
    #    Also, we must ensure that moving P away from its old lines doesn't break validity 
    #    (it only reduces counts, so validity is preserved unless we create a 5-line).
    #    And we must ensure P' doesn't land on any other existing line that already has 3 points 
    #    (which would create a 5-line).
    
    # Algorithm:
    # 1. Pick random index idx for P.
    # 2. Pick 3 random other indices i, j, k.
    # 3. Check if points[i], points[j], points[k] are collinear.
    #    If not, try a random small step move (fallback to random walk).
    #    If yes, let L be the line.
    # 4. Check if any other point (not i,j,k,idx) is on L.
    #    If yes, L would become 5+ points if we add P'. Invalid.
    # 5. Find an integer point P' on L that is:
    #    - Distinct from all current points.
    #    - Within bounds.
    #    - Not on any other line that currently has 3 points (to avoid creating a 5-line).
    #      Actually, if P' is on another line M with 3 points, M becomes 4. That's fine!
    #      We just need to ensure M doesn't already have 4 points (would become 5).
    #      And L doesn't already have 4 points (would become 5).
    # 6. Evaluate the new configuration.
    
    # To make this efficient, we need a fast way to find a valid P' on L.
    # We can iterate along L.
    
    # Let's implement a helper to find a valid spot on a line.
    
    def find_valid_point_on_line(line_key, points, exclude_idx, current_line_map):
        """
        Finds a point on the line 'line_key' that is valid to add.
        Returns (x, y) or None.
        """
        A, B, C = line_key
        # Parametric form: 
        # If B != 0, x can be any integer, y = (-A*x - C)/B. 
        # For y to be integer, (-A*x - C) must be divisible by B.
        # This requires solving a linear Diophantine equation or checking residues.
        # Since A, B, C are normalized, gcd(A, B) = 1? Not necessarily, but gcd(A,B,C)=1.
        # Actually, for the line to pass through integer points, the step vector is (B/g, -A/g) where g=gcd(A,B).
        # Let g = gcd(A, B).
        # Then x = x0 + t*(B/g), y = y0 - t*(A/g).
        # We need a base point (x0, y0) on the line.
        
        # Find one integer point on the line
        # If B != 0:
        #   y = (-A*x - C)/B.
        #   We need B | (A*x + C).
        #   Let g = gcd(A, B). Then g | C (since g|A and g|B and g|C? No, gcd(A,B,C)=1 implies g=1? 
        #   Wait. If gcd(A,B,C)=1, it does NOT imply gcd(A,B)=1.
        #   Example: 2x + 4y + 1 = 0. gcd(2,4,1)=1. But no integer solutions!
        #   Wait, the line is defined by 3 integer points. So integer solutions MUST exist.
        #   If 3 integer points lie on Ax+By+C=0, then there are infinite integer points.
        #   The condition for existence is gcd(A,B) | C.
        
        g_ab = math.gcd(A, B)
        if B != 0:
            if C % g_ab != 0:
                return None # Should not happen for valid lines
            # Solution exists.
            # Find one solution.
            # A/g * x + B/g * y = -C/g
            # Let A' = A/g_ab, B' = B/g_ab, C' = -C/g_ab
            A_prime = A // g_ab
            B_prime = B // g_ab
            C_prime = -C // g_ab
            
            # Find x0, y0 such that A'x + B'y = C'
            # Use extended Euclidean? Or just scan?
            # Since we are in a search loop, scanning a small range might be faster than EE.
            # But we need a point far enough to be distinct?
            # Let's just pick a random t in a range and check.
            
            # Get a base point from one of the points on the line?
            # We don't have the points here, only the line key.
            # But we can find any integer point.
            # Let's scan x in a window around 0.
            for x in range(-10, 11):
                val = A_prime * x - C_prime
                if val % B_prime == 0:
                    y = val // B_prime
                    # We have a base point (x, y).
                    # Step vector is (B_prime, -A_prime)
                    dx_step = B_prime
                    dy_step = -A_prime
                    
                    # Now scan t to find a valid point
                    # We want to avoid points already in 'points' (except exclude_idx? No, exclude_idx is the one moving, 
                    # so we must avoid all other points).
                    # And we want to avoid creating 5-lines.
                    
                    # Let's check a range of t
                    for t in range(-50, 51):
                        nx = x + t * dx_step
                        ny = y + t * dy_step
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        # Check distinctness
                        is_dup = False
                        for k in range(16):
                            if k == exclude_idx:
                                continue
                            if points[k][0] == nx and points[k][1] == ny:
                                is_dup = True
                                break
                        if is_dup:
                            continue
                        
                        # Check if this point creates a 5-line
                        # It is on 'line_key'. If 'line_key' already has 4 points (excluding the one moving? 
                        # No, 'line_key' has 3 points A,B,C. If we add P', it has 4. 
                        # But what if another point is already on 'line_key'? 
                        # We checked that before calling this? 
                        # We should check that 'line_key' count in current_line_map is exactly 3 
                        # (the points A,B,C) and not 4 (which would mean a 4th point is already there).
                        # If count is 4, adding P' makes 5. Invalid.
                        
                        if len(current_line_map.get(line_key, set())) >= 4:
                            return None # Should have been filtered before, but safe check
                        
                        # Check other lines through (nx, ny)
                        # For every other point Q, the line P'Q must not have 3 other points on it.
                        # i.e., count of points on line(P', Q) must be <= 2 (currently, excluding P').
                        # If it is 3, adding P' makes 4. That is VALID and GOOD.
                        # If it is 4, adding P' makes 5. INVALID.
                        
                        for k in range(16):
                            if k == exclude_idx:
                                continue
                            # Line between (nx, ny) and points[k]
                            l_key = _line_from_points([nx, ny], points[k])
                            # Count how many points are on this line (excluding P' and points[k] is not needed, 
                            # we just need to know if adding P' makes it 5).
                            # Current count of points on l_key in the set {0..15} \ {exclude_idx}
                            # Note: points[exclude_idx] is NOT on l_key unless it collinearly aligns, 
                            # but we are moving it away. So we consider the set of 15 fixed points.
                            # Let's count how many of the 15 fixed points are on l_key.
                            # If count >= 4, then adding P' makes it 5.
                            cnt = 0
                            for m in range(16):
                                if m == exclude_idx:
                                    continue
                                if m == k:
                                    continue # k is on the line by definition
                                # Check if points[m] is on l_key
                                Ax, By, Cx = l_key
                                if Ax * points[m][0] + By * points[m][1] + Cx == 0:
                                    cnt += 1
                            # cnt is number of OTHER points on the line.
                            # Total points on line after move = 1 (P') + 1 (Q) + cnt = 2 + cnt.
                            # We need 2 + cnt <= 4 => cnt <= 2.
                            if cnt > 2:
                                break # Invalid
                        else:
                            # All checks passed
                            return [nx, ny]
                    break
        elif A != 0:
            # Vertical line x = -C/A
            if C % A != 0:
                return None
            x0 = -C // A
            if abs(x0) > 1000:
                return None
            # Scan y
            for y in range(-1000, 1001):
                nx = x0
                ny = y
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                is_dup = False
                for k in range(16):
                    if k == exclude_idx:
                        continue
                    if points[k][0] == nx and points[k][1] == ny:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                if len(current_line_map.get(line_key, set())) >= 4:
                    return None
                
                valid = True
                for k in range(16):
                    if k == exclude_idx:
                        continue
                    l_key = _line_from_points([nx, ny], points[k])
                    cnt = 0
                    for m in range(16):
                        if m == exclude_idx:
                            continue
                        if m == k:
                            continue
                        Ax, By, Cx = l_key
                        if Ax * points[m][0] + By * points[m][1] + Cx == 0:
                            cnt += 1
                    if cnt > 2:
                        valid = False
                        break
                if valid:
                    return [nx, ny]
        return None

    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # 1. Pick a random point to move
        idx = random.randint(0, 15)
        ox, oy = current_points[idx]
        
        # 2. Pick 3 other points to define a target line
        # We want to form a new 4-line.
        # So we pick 3 points that are collinear.
        # If they are not collinear, we can't form a line with just them.
        # We can try to find a line defined by 2 points and hope the 3rd is on it?
        # No, we need 3 points to define the line uniquely in a way that we can check "is there a 4th?".
        # Actually, any 2 points define a line. We can check if a 3rd point is on it.
        
        # Strategy: Pick 2 random points A, B. 
        # Check if any other point C is on line AB.
        # If yes, we have a 3-point line.
        # If no, the line AB has only 2 points. Adding P' makes 3. Not a 4-line.
        # So we specifically look for 3-point lines to upgrade to 4-point lines.
        
        found_line = False
        target_line_key = None
        target_points_indices = []
        
        # Try a few random pairs to find a 3-point line
        for _ in range(10):
            i, j = random.sample(range(16), 2)
            if i == idx or j == idx:
                continue
            l_key = _line_from_points(current_points[i], current_points[j])
            # Check if any other point k (not i, j, idx) is on this line
            for k in range(16):
                if k == i or k == j or k == idx:
                    continue
                Ax, By, Cx = l_key
                if Ax * current_points[k][0] + By * current_points[k][1] + Cx == 0:
                    # Found a 3-point line
                    target_line_key = l_key
                    target_points_indices = [i, j, k]
                    found_line = True
                    break
            if found_line:
                break
        
        if found_line:
            # Check if this line already has 4 points (including idx? No, idx is not on it by construction)
            # If it has 4 points, we can't add a 5th.
            # We need exactly 3 points on it (other than idx).
            # Since we found i,j,k on it, and idx is not on it, the count is at least 3.
            # If there is a 4th point m on it, count is 4. Adding idx makes 5. Invalid.
            cnt_on_line = 0
            for m in range(16):
                if m == idx:
                    continue
                Ax, By, Cx = target_line_key
                if Ax * current_points[m][0] + By * current_points[m][1] + Cx == 0:
                    cnt_on_line += 1
            
            if cnt_on_line == 3:
                # Find a valid spot for idx on this line
                new_pos = find_valid_point_on_line(target_line_key, current_points, idx, current_line_map)
                if new_pos:
                    # Apply move
                    current_points[idx] = new_pos
                    new_score, new_valid, new_line_map = _count_4_lines(current_points)
                    
                    if new_valid:
                        if new_score > current_score:
                            current_score = new_score
                            current_line_map = new_line_map
                            if new_score > best_score:
                                best_score = new_score
                                best_points = [p[:] for p in current_points]
                        else:
                            # Accept with small probability to escape local minima?
                            # Or just reject?
                            if random.random() < 0.1:
                                current_score = new_score
                                current_line_map = new_line_map
                            else:
                                current_points[idx] = [ox, oy]
                    else:
                        current_points[idx] = [ox, oy]
                else:
                    # No valid spot found on this line. Revert (didn't change)
                    pass
            else:
                # Line has 4+ points. Can't use it.
                pass
        else:
            # No 3-point line found. Do a small random step.
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            nx = ox + dx
            ny = oy + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                is_dup = False
                for k in range(16):
                    if k == idx:
                        continue
                    if current_points[k][0] == nx and current_points[k][1] == ny:
                        is_dup = True
                        break
                if not is_dup:
                    current_points[idx] = [nx, ny]
                    new_score, new_valid, new_line_map = _count_4_lines(current_points)
                    if new_valid:
                        if new_score > current_score or (new_score == current_score and random.random() < 0.2):
                            current_score = new_score
                            current_line_map = new_line_map
                            if new_score > best_score:
                                best_score = new_score
                                best_points = [p[:] for p in current_points]
                        else:
                            current_points[idx] = [ox, oy]
                    else:
                        current_points[idx] = [ox, oy]

    return best_points
