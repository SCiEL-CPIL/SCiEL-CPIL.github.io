import numpy as np
from itertools import product

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    
    # Field F_49 = F_7[alpha] where alpha^2 = 3
    # Elements represented as (a, b) meaning a + b*alpha
    # Multiplication: (a,b)*(c,d) = (ac+3bd, ad+bc) mod 7
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + 3*b[1]*a[1]) % p, 
                (a[0]*b[1] + a[1]*b[0]) % p)
    
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
    
    # Precompute all field elements as tuples
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Precompute multiplication table
    mul_table = {}
    for a in field_elements:
        for b in field_elements:
            mul_table[(a, b)] = field_mul(a, b)
    
    # Precompute all lines in F_49^2
    # A line is determined by a point and a direction
    # Directions: v = (1, m) for m in F_49, or v = (0, 1) for vertical
    # Each line has 49 points
    
    # Generate all lines as frozensets of point tuples
    # Point: ((xa, xb), (ya, yb)) -> we'll use (xa, xb, ya, yb)
    
    all_lines = []
    seen_lines = set()
    
    # Vertical lines: x = (xa, xb), varying y
    for xa in range(p):
        for xb in range(p):
            points_on_line = []
            for ya in range(p):
                for yb in range(p):
                    points_on_line.append((xa, xb, ya, yb))
            line_tuple = tuple(sorted(points_on_line))
            if line_tuple not in seen_lines:
                seen_lines.add(line_tuple)
                all_lines.append(frozenset(points_on_line))
    
    # Non-vertical lines: direction (1, m) where m = (ma, mb)
    for ma in range(p):
        for mb in range(p):
            # Direction v = (1, (ma, mb))
            # Line through point (x0, y0): {(x0 + t, y0 + t*(ma,mb)) : t in F_49}
            # where t = (ta, tb)
            
            for x0a in range(p):
                for x0b in range(p):
                    for y0a in range(p):
                        for y0b in range(p):
                            # Generate all 49 points on this line
                            points_on_line = []
                            for ta in range(p):
                                for tb in range(p):
                                    # x = x0 + t
                                    xa = (x0a + ta) % p
                                    xb = (x0b + tb) % p
                                    # y = y0 + t * m
                                    t_m = field_mul((ta, tb), (ma, mb))
                                    ya = (y0a + t_m[0]) % p
                                    yb = (y0b + t_m[1]) % p
                                    points_on_line.append((xa, xb, ya, yb))
                            
                            # Check if this line is new
                            # Use a canonical form: sort and take a hash
                            # But frozenset works
                            line_tuple = tuple(sorted(points_on_line))
                            if line_tuple not in seen_lines:
                                seen_lines.add(line_tuple)
                                all_lines.append(frozenset(points_on_line))
    
    # Total lines should be 49 * 50 = 2450
    # (49 vertical + 49*49 non-vertical = 49 + 2401 = 2450)
    
    # Build point-to-lines incidence
    # For each point, which lines contain it?
    point_to_lines = {}
    for point in product(range(p), repeat=4):
        point_to_lines[point] = []
    
    for line in all_lines:
        for point in line:
            point_to_lines[point].append(line)
    
    # Now, Nikodym set N: every point x (in F_49^2) must lie on a line L
    # such that L \ {x} ⊆ N
    # Equivalently: for each x, there exists a line L through x such that
    # L ∩ (F_49^2 \ N) ⊆ {x}
    # i.e., L ∩ complement_of_N ⊆ {x}
    
    # Let C = complement of N. We want to maximize |C|.
    # Condition: for every x, there exists a line L through x such that
    # L ∩ C ⊆ {x}
    # If x ∉ C (x ∈ N): L ∩ C = ∅, so L ⊆ N
    # If x ∈ C: L ∩ C = {x}, so L \ {x} ⊆ N
    
    # This is a complex constraint. Let's think about it differently.
    
    # The current solution uses a unital-based construction. Let me understand it.
    # The unital in F_49^2: U = {(x,y) : y^2 = x + x^2*alpha^2} or similar
    # Actually, the code checks: (2*xa + ya^2 - yb^2*3) % 7 == 0
    # i.e., ya^2 - 3*yb^2 = 2*xa (in F_7, ignoring xb)
    # This doesn't look like a standard unital.
    
    # Let me just implement a greedy approach:
    # Start with N = F_49^2 (all points). C = ∅.
    # Try to add points to C (remove from N) while maintaining validity.
    
    # Validity check for a given C:
    # For each point x, check if there exists a line L through x such that L ∩ C ⊆ {x}
    
    # This is expensive. Let's precompute for each line, which points are in C.
    
    # Greedy: try to add points to C one by one, checking validity after each addition.
    # To speed up, we can check only the points affected (points on lines through the new point).
    
    # Actually, let's start from the current solution and try to remove more points.
    
    # First, let's reconstruct the current solution's N.
    k = 3  # alpha^2 = 3
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    N_points = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue
                    N_points.add((xa, xb, ya, yb))
    
    C_points = set()
    all_points = set(product(range(p), repeat=4))
    C_points = all_points - N_points
    
    # Now try to grow C (shrink N) greedily
    # For each point not in C, try adding it to C and check if validity is maintained.
    
    # To check validity efficiently:
    # Maintain for each line, the set of points in C on that line.
    # A point x is "covered" if there exists a line L through x with |L ∩ C| ≤ 1 
    # and if |L ∩ C| = 1, that point is x.
    
    # Let's maintain line_C_count[line] = number of points in C on that line
    # and line_C_points[line] = set of points in C on that line
    
    line_C_points = {}
    for line in all_lines:
        line_C_points[line] = set()
    
    for point in C_points:
        for line in point_to_lines[point]:
            line_C_points[line].add(point)
    
    def is_valid_current():
        """Check if current C is valid (i.e., N = all_points - C is a Nikodym set)."""
        for point in all_points:
            found_good_line = False
            for line in point_to_lines[point]:
                c_on_line = line_C_points[line]
                if len(c_on_line) == 0:
                    # L ∩ C = ∅, so L ⊆ N. Valid for x not in C.
                    # If x in C, this can't happen since x would be in c_on_line.
                    # Actually if x in C, x is in c_on_line, so len >= 1.
                    # So this case only arises if x not in C.
                    found_good_line = True
                    break
                elif len(c_on_line) == 1 and point in c_on_line:
                    # L ∩ C = {x}. Valid.
                    found_good_line = True
                    break
            if not found_good_line:
                return False
        return True
    
    # Verify current state is valid
    # assert is_valid_current()
    
    # Now greedily try to add more points to C
    # Sort candidate points (those not in C) by some heuristic
    # Try each, check validity, if valid, keep it.
    
    candidates = list(all_points - C_points)
    
    # Shuffle for randomness
    import random
    random.seed(42)
    random.shuffle(candidates)
    
    improved = False
    for point in candidates:
        # Try adding point to C
        C_points.add(point)
        for line in point_to_lines[point]:
            line_C_points[line].add(point)
        
        if is_valid_current():
            improved = True
        else:
            # Revert
            C_points.remove(point)
            for line in point_to_lines[point]:
                line_C_points[line].discard(point)
    
    # Second pass with different order
    random.seed(123)
    random.shuffle(candidates)
    for point in candidates:
        if point in C_points:
            continue
        C_points.add(point)
        for line in point_to_lines[point]:
            line_C_points[line].add(point)
        
        if is_valid_current():
            improved = True
        else:
            C_points.remove(point)
            for line in point_to_lines[point]:
                line_C_points[line].discard(point)
    
    # Third pass
    random.seed(456)
    random.shuffle(candidates)
    for point in candidates:
        if point in C_points:
            continue
        C_points.add(point)
        for line in point_to_lines[point]:
            line_C_points[line].add(point)
        
        if is_valid_current():
            improved = True
        else:
            C_points.remove(point)
            for line in point_to_lines[point]:
                line_C_points[line].discard(point)
    
    # Final N
    N_final = all_points - C_points
    
    # Convert to required format: array of shape (k, 2, 2)
    points_list = []
    for (xa, xb, ya, yb) in sorted(N_final):
        points_list.append([[xa, xb], [ya, yb]])
    
    if not points_list:
        # Fallback: return all points
        for (xa, xb, ya, yb) in sorted(all_points):
            points_list.append([[xa, xb], [ya, yb]])
    
    return np.array(points_list, dtype=np.int64)
