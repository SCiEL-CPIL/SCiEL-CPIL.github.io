import time
import random
import math
from itertools import combinations

# AG(2,4) Structure
# Field F4 = {0, 1, alpha, alpha+1}
# Let's map F4 to 0,1,2,3 for indexing
# 0 -> 0
# 1 -> 1
# alpha -> 2
# alpha+1 -> 3

# Points are (u, v) where u, v in {0,1,2,3}
# Index = u * 4 + v

def _get_ag24_lines():
    lines = []
    # Vertical lines: u = c
    for c in range(4):
        line = set()
        for v in range(4):
            line.add(c * 4 + v)
        lines.append(frozenset(line))
        
    # Non-vertical lines: v = m * u + c
    # Multiplication table for F4:
    # 0*x = 0
    # 1*x = x
    # 2*x (alpha): 
    #   2*0 = 0
    #   2*1 = 2
    #   2*2 = 2*2 = alpha^2 = alpha+1 = 3
    #   2*3 = 2*(2+1) = 2*2 + 2*1 = 3 + 2 = 5 = 1 (since 2+2=0, 3+2=1? No. 3=alpha+1. 2*(alpha+1) = 2alpha+2 = (alpha+1)+2 = alpha+3 = alpha+1 = 2? 
    #   Let's recompute 2*3 in F4.
    #   3 = alpha + 1.
    #   2 * (alpha + 1) = 2*alpha + 2*1 = (alpha^2) + 2 = (alpha+1) + 2 = alpha + 3.
    #   In F4, 3 = alpha+1. So alpha + 3 = alpha + alpha + 1 = 1.
    #   So 2*3 = 1.
    # 3*x (alpha+1):
    #   3*0 = 0
    #   3*1 = 3
    #   3*2 = (2+1)*2 = 2*2 + 1*2 = 3 + 2 = 1
    #   3*3 = (2+1)*(2+1) = 2*2 + 2*1 + 1*2 + 1*1 = 3 + 2 + 2 + 1 = 3 + 0 + 1 = 0?
    #   Wait. 2+2=0. So 3+2+2+1 = 3+1 = 2+1+1 = 2.
    #   Let's check 3*3 = (alpha+1)^2 = alpha^2 + 2alpha + 1 = (alpha+1) + 0 + 1 = alpha = 2.
    #   So 3*3 = 2.
    
    mult_table = {
        0: [0, 0, 0, 0],
        1: [0, 1, 2, 3],
        2: [0, 2, 3, 1],
        3: [0, 3, 1, 2]
    }
    
    for m in range(4):
        for c in range(4):
            line = set()
            for u in range(4):
                v = mult_table[m][u] + c
                # Addition in F4:
                # 0+0=0, 0+1=1, 0+2=2, 0+3=3
                # 1+1=0, 1+2=3, 1+3=2
                # 2+2=0, 2+3=1
                # 3+3=0
                # XOR works if we map 0,1,2,3 to 00,01,10,11?
                # 0=00, 1=01, 2=10, 3=11.
                # 1+1 = 01+01 = 00 = 0. Correct.
                # 2+2 = 10+10 = 00 = 0. Correct.
                # 3+3 = 11+11 = 00 = 0. Correct.
                # 1+2 = 01+10 = 11 = 3. Correct.
                # So v = mult_table[m][u] ^ c
                v = mult_table[m][u] ^ c
                line.add(u * 4 + v)
            lines.append(frozenset(line))
            
    return lines

AG24_LINES = _get_ag24_lines()

def _count_four_point_lines(points):
    n = len(points)
    count = 0
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(dx, dy)
            if g != 0:
                dxn = dx // g
                dyn = dy // g
            else:
                dxn, dyn = 0, 0
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            a = dyn
            b = -dxn
            c = a * p1[0] + b * p1[1]
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            pts_on_line = []
            for k in range(n):
                pk = points[k]
                if a * pk[0] + b * pk[1] == c:
                    pts_on_line.append(k)
            if len(pts_on_line) == 4:
                count += 1
            elif len(pts_on_line) >= 5:
                return -1
    return count

def _is_valid(points):
    n = len(points)
    seen = set()
    for p in points:
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    seen_lines = set()
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = math.gcd(dx, dy)
            if g != 0:
                dxn = dx // g
                dyn = dy // g
            else:
                continue
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            a = dyn
            b = -dxn
            c = a * p1[0] + b * p1[1]
            line_key = (a, b, c)
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)
            cnt = 0
            for k in range(n):
                pk = points[k]
                if a * pk[0] + b * pk[1] == c:
                    cnt += 1
            if cnt >= 5:
                return False
    return True

def _evaluate(points):
    if not _is_valid(points):
        return -1
    return _count_four_point_lines(points)

def _ag24_cost(points):
    """Calculate collinearity error for AG(2,4) lines."""
    cost = 0.0
    for line in AG24_LINES:
        pts = [points[i] for i in line]
        # Check all triples
        # Use first point as reference
        p1 = pts[0]
        for i in range(1, 4):
            for j in range(i+1, 4):
                p2 = pts[i]
                p3 = pts[j]
                # Area of triangle p1, p2, p3
                area = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                cost += area * area
    return cost

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find AG(2,4) embedding
    best_points = None
    best_score = -1
    
    # Random search for AG(2,4) embedding
    # We fix 3 points to break symmetry
    # (0,0), (1,0), (0,1)
    # The other 13 points are searched.
    
    # However, the embedding might not align with these.
    # So we do full random search with SA on cost.
    
    random.seed(42)
    
    # Initial best from old method
    grid_points = []
    for x in range(-2, 4):
        for y in range(-2, 4):
            grid_points.append([x, y])
            
    # Quick old search to get a baseline
    for restart in range(10):
        if time.time() > deadline - 60:
            break
        current_points = [p[:] for p in random.sample(grid_points, 16)]
        current_score = _evaluate(current_points)
        if current_score > best_score:
            best_score = current_score
            best_points = [p[:] for p in current_points]
            
    # Now try AG(2,4) embedding search
    # Use larger grid for initial random points
    M = 20
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        # Initialize points randomly in [-M, M]
        current_points = [[random.randint(-M, M), random.randint(-M, M)] for _ in range(16)]
        
        # Ensure distinct
        valid = True
        seen = set()
        for p in current_points:
            if (p[0], p[1]) in seen:
                valid = False
                break
            seen.add((p[0], p[1]))
        if not valid:
            continue
            
        T = 100.0
        T_min = 0.1
        cooling = 0.99
        current_cost = _ag24_cost(current_points)
        
        while T > T_min and time.time() < deadline - 5:
            i = random.randint(0, 15)
            old_x, old_y = current_points[i]
            
            # Move to random nearby point
            new_x = random.randint(-M, M)
            new_y = random.randint(-M, M)
            
            # Check distinct
            if (new_x, new_y) in seen:
                # Remove old, add new? 
                # Simpler: just skip if collision
                # But we need to update seen set
                pass
            else:
                # Update seen
                seen.remove((old_x, old_y))
                seen.add((new_x, new_y))
                current_points[i] = [new_x, new_y]
                
                new_cost = _ag24_cost(current_points)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        # Found perfect embedding
                        # Verify
                        if _is_valid(current_points):
                            score = _count_four_point_lines(current_points)
                            if score > best_score:
                                best_score = score
                                best_points = [p[:] for p in current_points]
                        break
                else:
                    if random.random() < math.exp((current_cost - new_cost) / T):
                        current_cost = new_cost
                    else:
                        # Revert
                        current_points[i] = [old_x, old_y]
                        seen.remove((new_x, new_y))
                        seen.add((old_x, old_y))
                        
            T *= cooling
            
    # Final check
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        
    return best_points
