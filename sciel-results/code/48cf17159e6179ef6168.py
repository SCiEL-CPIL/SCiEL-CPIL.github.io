import time
import math
from itertools import combinations

def is_perfect_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def are_collinear(p1, p2, p3):
    # p1, p2, p3 are (x, y) integers
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

def are_concyclic(p1, p2, p3, p4):
    # Check if 4 points are concyclic
    # Determinant of:
    # x  y  x^2+y^2  1
    # ...
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # Using the formula for the determinant
    # | x1 y1 x1^2+y1^2 1 |
    # | x2 y2 x2^2+y2^2 1 |
    # | x3 y3 x3^2+y3^2 1 |
    # | x4 y4 x4^2+y4^2 1 |
    
    # We can subtract row 4 from rows 1, 2, 3 to simplify
    # Let's just compute the full determinant using integer arithmetic
    # Since numbers might get large, but for small coordinates it's fine.
    
    # Alternative: Check if the perpendicular bisectors of P1P2, P2P3, P3P4 intersect at a single point.
    # Or check if the power of P4 with respect to circle P1P2P3 is 0.
    
    # Circle through P1, P2, P3:
    # (x-a)^2 + (y-b)^2 = r^2
    # x^2 - 2ax + a^2 + y^2 - 2by + b^2 = r^2
    # x^2 + y^2 - 2ax - 2by + (a^2+b^2-r^2) = 0
    # Let C = a^2+b^2-r^2
    # x^2 + y^2 - 2ax - 2by + C = 0
    
    # We have:
    # x1^2 + y1^2 - 2a x1 - 2b y1 + C = 0
    # x2^2 + y2^2 - 2a x2 - 2b y2 + C = 0
    # x3^2 + y3^2 - 2a x3 - 2b y3 + C = 0
    
    # This is a linear system for 2a, 2b, C.
    # Let A = 2a, B = 2b, C = C
    # -x1*A - y1*B + C = -(x1^2 + y1^2)
    # -x2*A - y2*B + C = -(x2^2 + y2^2)
    # -x3*A - y3*B + C = -(x3^2 + y3^2)
    
    # Subtract row 3 from row 1 and 2:
    # -x1*A - y1*B + x3*A + y3*B = -(x1^2 + y1^2) + (x3^2 + y3^2)
    # (x3 - x1)A + (y3 - y1)B = x3^2 + y3^2 - x1^2 - y1^2
    
    # -x2*A - y2*B + x3*A + y3*B = -(x2^2 + y2^2) + (x3^2 + y3^2)
    # (x3 - x2)A + (y3 - y2)B = x3^2 + y3^2 - x2^2 - y2^2
    
    # Let:
    # D1 = x3^2 + y3^2 - x1^2 - y1^2
    # D2 = x3^2 + y3^2 - x2^2 - y2^2
    # a1 = x3 - x1
    # b1 = y3 - y1
    # a2 = x3 - x2
    # b2 = y3 - y2
    
    # a1*A + b1*B = D1
    # a2*A + b2*B = D2
    
    # Determinant of this 2x2 system:
    # Det = a1*b2 - a2*b1
    # If Det == 0, points are collinear (which we already check).
    
    # A = (D1*b2 - D2*b1) / Det
    # B = (a1*D2 - a2*D1) / Det
    
    # We need A, B to be rational. Since we are in integers, we can keep them as fractions or check if the resulting circle passes through P4.
    # Instead of solving for A, B, C explicitly, we can check if P4 satisfies the circle equation defined by P1, P2, P3.
    # The condition for P4 to be on the circle is that the determinant of the 4x4 matrix is 0.
    
    # Let's compute the 4x4 determinant directly.
    # To avoid huge numbers, we can use the fact that we only care if it's 0.
    
    # Row 1: x1, y1, x1^2+y1^2, 1
    # Row 2: x2, y2, x2^2+y2^2, 1
    # Row 3: x3, y3, x3^2+y3^2, 1
    # Row 4: x4, y4, x4^2+y4^2, 1
    
    # Subtract Row 4 from Rows 1, 2, 3
    # R1': x1-x4, y1-y4, (x1^2+y1^2)-(x4^2+y4^2), 0
    # R2': x2-x4, y2-y4, (x2^2+y2^2)-(x4^2+y4^2), 0
    # R3': x3-x4, y3-y4, (x3^2+y3^2)-(x4^2+y4^2), 0
    
    # The determinant is the determinant of the 3x3 matrix formed by the first 3 columns of R1', R2', R3'.
    
    r1x = x1 - x4
    r1y = y1 - y4
    r1z = x1*x1 + y1*y1 - x4*x4 - y4*y4
    
    r2x = x2 - x4
    r2y = y2 - y4
    r2z = x2*x2 + y2*y2 - x4*x4 - y4*y4
    
    r3x = x3 - x4
    r3y = y3 - y4
    r3z = x3*x3 + y3*y3 - x4*x4 - y4*y4
    
    det = (r1x * (r2y * r3z - r2z * r3y) -
           r1y * (r2x * r3z - r2z * r3x) +
           r1z * (r2x * r3y - r2y * r3x))
           
    return det == 0

def check_set_valid(points):
    n = len(points)
    # Check distinct
    if len(set(points)) != n:
        return False
    
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return False
            
    # Check no 3 collinear
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if are_collinear(points[i], points[j], points[k]):
                    return False
    
    # Check no 4 concyclic
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
                        
    return True

def construct():
    start_time = time.time()
    deadline = 110.0 # 120s limit, keep 10s margin
    
    # Seed with a known 4-point set
    base_set = [(0, 0), (24, 0), (0, 7), (24, 7)]
    
    # Generate candidates
    # We want points (x, y) such that distance to (0,0) is integer? No, just any integer point.
    # But to be efficient, let's restrict to points that are at integer distance from (0,0).
    # This ensures the first connection is valid.
    # Then we check distances to other points in the base set.
    
    L = 500
    candidates = set()
    candidates.add((0,0))
    
    # Generate Pythagorean triples
    # x = k*(m^2-n^2), y = k*(2mn), z = k*(m^2+n^2)
    # Or x = k*(2mn), y = k*(m^2-n^2)
    for m in range(2, 30):
        for n in range(1, m):
            if (m-n) % 2 == 0: # coprime and opposite parity
                continue
            if math.gcd(m, n) != 1:
                continue
            xm = m*m - n*n
            ym = 2*m*n
            zm = m*m + n*n
            for k in range(1, L // zm + 1):
                x = k * xm
                y = k * ym
                if abs(x) <= L and abs(y) <= L:
                    candidates.add((x, y))
                    candidates.add((y, x))
                    candidates.add((-x, y))
                    candidates.add((x, -y))
                    candidates.add((-x, -y))
                    candidates.add((y, -x))
                    candidates.add((-y, x))
                    candidates.add((-y, -x))
    
    # Filter candidates that are at integer distance from all points in base_set
    # This is a strong filter.
    valid_candidates = []
    for p in candidates:
        if p in base_set:
            continue
        valid = True
        for bp in base_set:
            dx = p[0] - bp[0]
            dy = p[1] - bp[1]
            if not is_perfect_square(dx*dx + dy*dy):
                valid = False
                break
        if valid:
            valid_candidates.append(p)
            
    # Sort candidates by some heuristic (e.g., distance from origin)
    valid_candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    best_set = base_set[:]
    
    # Try to add points one by one
    current_set = base_set[:]
    
    for target_size in range(5, 9):
        if time.time() > deadline:
            break
        
        found = False
        # Shuffle candidates for variety? Or just iterate.
        # To find a 6-point set, we need to add 2 points.
        # We can do a search:
        # For each candidate c1:
        #   new_set = current_set + [c1]
        #   if check_set_valid(new_set):
        #      For each candidate c2 (not c1, not in current_set):
        #         new_set2 = new_set + [c2]
        #         if check_set_valid(new_set2):
        #             return new_set2
        #      # If we didn't find c2, we can still return new_set if target_size == 5
        #      if target_size == 5:
        #          return new_set
        #      # Else, we might need to backtrack or try different c1
        #      # But for now, let's just try to find a valid extension.
        
        # Simple greedy with backtracking for 1 step
        for i, c1 in enumerate(valid_candidates):
            if time.time() > deadline:
                break
            
            new_set = current_set + [c1]
            if not check_set_valid(new_set):
                continue
            
            if target_size == len(new_set):
                return new_set
            
            # Try to add one more
            for j, c2 in enumerate(valid_candidates):
                if c2 == c1:
                    continue
                if c2 in current_set:
                    continue
                
                new_set2 = new_set + [c2]
                if check_set_valid(new_set2):
                    if target_size == len(new_set2):
                        return new_set2
                    # If we found a valid set larger than target, return it?
                    # The task says maximize. So return the largest found.
                    # But we are iterating target_size.
                    # If we found a set of size 6 while looking for 5, we should probably return it.
                    # Let's just return any valid set found that is larger than the current best.
                    if len(new_set2) > len(best_set):
                        best_set = new_set2
                        return new_set2 # Or continue searching for larger?
                        # The task says "Maximize". So we should keep searching if time permits.
                        # But returning early with a valid larger set is good.
                        # However, we might find an even larger one.
                        # Let's update best_set and break out to try larger targets?
                        # No, the loop structure is a bit rigid.
                        # Let's just return the first valid set of size >= target_size.
                        # Actually, if we find 6, we are done (8 is the goal, but 6 is better than 5).
                        # Let's aim for the highest possible.
                        # If we find 6, we can try to extend to 7.
                        # So, if we find a valid set, update best_set.
                        # Then break out of inner loops and continue outer loop to try to add more.
                        # But `return` exits the function.
                        # So we should structure it to find the max.
                        
                        # Let's change strategy:
                        # Find the largest valid set we can.
                        # We will use a recursive search or iterative deepening.
                        
                        # For now, let's just return the first valid set of size > 5.
                        # This guarantees an improvement over 5.
                        return new_set2
        
        # If we didn't return, we didn't find a set of size target_size
        # Try to update best_set if we found a valid set of size target_size in the c1 loop
        # (This is handled by the return inside)
        
    # If we didn't find anything better, return the best we have (base_set)
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0
    
    # Known 4-point set
    base_set = [(0, 0), (24, 0), (0, 7), (24, 7)]
    
    # Generate candidates
    L = 500
    candidates = set()
    candidates.add((0,0))
    
    for m in range(2, 30):
        for n in range(1, m):
            if (m-n) % 2 == 0:
                continue
            if math.gcd(m, n) != 1:
                continue
            xm = m*m - n*n
            ym = 2*m*n
            zm = m*m + n*n
            for k in range(1, L // zm + 1):
                x = k * xm
                y = k * ym
                if abs(x) <= L and abs(y) <= L:
                    candidates.add((x, y))
                    candidates.add((y, x))
                    candidates.add((-x, y))
                    candidates.add((x, -y))
                    candidates.add((-x, -y))
                    candidates.add((y, -x))
                    candidates.add((-y, x))
                    candidates.add((-y, -x))
    
    valid_candidates = []
    for p in candidates:
        if p in base_set:
            continue
        valid = True
        for bp in base_set:
            dx = p[0] - bp[0]
            dy = p[1] - bp[1]
            if not is_perfect_square(dx*dx + dy*dy):
                valid = False
                break
        if valid:
            valid_candidates.append(p)
            
    valid_candidates.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    
    best_set = base_set[:]
    current_set = base_set[:]
    
    # Try to extend to 5, then 6, etc.
    for target_size in range(5, 9):
        if time.time() > deadline:
            break
            
        # Try to add (target_size - len(current_set)) points
        # This is a simple greedy approach.
        # For each candidate, check if it can be added.
        
        extended = False
        for c1 in valid_candidates:
            if time.time() > deadline:
                break
            if c1 in current_set:
                continue
            
            new_set = current_set + [c1]
            if not check_set_valid(new_set):
                continue
            
            # If we just need 1 more point
            if len(new_set) == target_size:
                return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in new_set]}
            
            # If we need more, try to add another
            # This only works if target_size - len(current_set) == 2
            if target_size - len(current_set) == 2:
                for c2 in valid_candidates:
                    if time.time() > deadline:
                        break
                    if c2 == c1 or c2 in current_set:
                        continue
                    
                    new_set2 = new_set + [c2]
                    if check_set_valid(new_set2):
                        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in new_set2]}
                        
        # If we didn't return, we didn't find a set of this size with the current current_set
        # We could try a different base_set or restart, but for now, we just break
        # and return the best we have.
        
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_set]}
# EVOLVE-BLOCK-END
