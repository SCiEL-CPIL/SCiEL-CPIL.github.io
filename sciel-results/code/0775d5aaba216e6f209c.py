"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = list(base_points)
    best_D = 1
    
    def is_square(n):
        """Exact check if n is a perfect square (for non-negative integers)."""
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n
    
    def dist_sq_int(p1, p2, D):
        """Return squared distance as integer if all coords are integers, else None."""
        x1, y1 = p1
        x2, y2 = p2
        dx = x1 - x2
        dy = y1 - y2
        # For integer points with D=1: d^2 = dx^2 + dy^2
        # For general D: d^2 = dx^2 + D*dy^2
        return dx * dx + D * dy * dy
    
    def all_dist_rational_int(points, D=1):
        """Check if all pairwise squared distances are perfect squares (for integer coords)."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq_int(points[i], points[j], D)
                if not is_square(d2):
                    return False
        return True
    
    def no_three_collinear(points):
        """Check no three points are collinear using exact integer arithmetic."""
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx12 = x2 - x1
                dy12 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    dx13 = x3 - x1
                    dy13 = y3 - y1
                    # Cross product
                    if dx12 * dy13 - dx13 * dy12 == 0:
                        return False
        return True
    
    def no_four_concyclic(points):
        """Check no four points are concyclic using exact integer arithmetic."""
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            a1 = x1 * x1 + y1 * y1
            for j in range(i + 1, n):
                x2, y2 = points[j]
                a2 = x2 * x2 + y2 * y2
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    a3 = x3 * x3 + y3 * y3
                    for l in range(k + 1, n):
                        x4, y4 = points[l]
                        a4 = x4 * x4 + y4 * y4
                        # Determinant of 4x4 matrix:
                        # | a1  x1  y1  1 |
                        # | a2  x2  y2  1 |
                        # | a3  x3  y3  1 |
                        # | a4  x4  y4  1 |
                        # Using expansion along last column:
                        # det = a1*(x2*y3 - x2*y4 - x3*y2 + x3*y4 + x4*y2 - x4*y3)
                        #     - x1*(a2*y3 - a2*y4 - a3*y2 + a3*y4 + a4*y2 - a4*y3)
                        #     + y1*(a2*x3 - a2*x4 - a3*x2 + a3*x4 + a4*x2 - a4*x3)
                        #     - 1*(a2*x3*y4 - a2*x4*y3 - a3*x2*y4 + a3*x4*y2 + a4*x2*y3 - a4*x3*y2)
                        
                        # Let me use a cleaner formulation
                        # det = |a1 x1 y1 1|
                        #       |a2 x2 y2 1|
                        #       |a3 x3 y3 1|
                        #       |a4 x4 y4 1|
                        # Subtract row1 from rows 2,3,4:
                        # |a1 x1 y1 1|
                        # |a2-a1 x2-x1 y2-y1 0|
                        # |a3-a1 x3-x1 y3-y1 0|
                        # |a4-a1 x4-x1 y4-y1 0|
                        # det = 1 * det of 3x3:
                        # |a2-a1 x2-x1 y2-y1|
                        # |a3-a1 x3-x1 y3-y1|
                        # |a4-a1 x4-x1 y4-y1|
                        
                        ba = a2 - a1
                        bx = x2 - x1
                        by = y2 - y1
                        ca = a3 - a1
                        cx = x3 - x1
                        cy = y3 - y1
                        da = a4 - a1
                        dx = x4 - x1
                        dy = y4 - y1
                        
                        det3 = (ba * (cy * dy - cy * dy)  # wait, let me be careful
                               # 3x3 det = ba*(cy*dy - cy*dy) ... no
                               # |ba bx by|
                               # |ca cx cy|
                               # |da dx dy|
                               # = ba*(cy*dy - cy*dy) ... 
                               # = ba*(cy*dy - cy*dy)
                               # Standard 3x3:
                               # = ba*(cy*dy - cy*dy)
                               # Let me just compute it properly:
                               )
                        # 3x3 determinant:
                        # ba*(cy*dy - cy*dy) ... I'm confusing myself
                        # det = ba*(cy*dy - cy*dy)
                        # No: det = ba*(cy*dy - cy*dy)
                        # Let me just write it out:
                        # det3 = ba*(cy*dy - cy*dy)
                        # Actually: 
                        # det3 = ba*(cy*dy - cy*dy)
                        # I need to be more careful:
                        # |ba bx by|
                        # |ca cx cy|
                        # |da dx dy|
                        # = ba*(cy*dy - cy*dy)
                        # = ba*(cy*dy) - ba*(cy*dy)
                        # Let me just compute:
                        det3 = (ba * (cy * dy - cy * dy)  # This is wrong, let me restart
                               )
                        # OK let me just do it the standard way:
                        # det = ba*(cy*dy - cy*dy)
                        # I'll compute each term:
                        t1 = ba * (cy * dy - cy * dy)  # still wrong
                        
                        # Let me just use the direct formula:
                        # det3 = ba*(cy*dy - cy*dy)
                        # No. The 3x3 det is:
                        # a(ei - fh) - b(di - fg) + c(dh - eg)
                        # where matrix is:
                        # [a b c] = [ba bx by]
                        # [d e f] = [ca cx cy]
                        # [g h i] = [da dx dy]
                        a_, b_, c_ = ba, bx, by
                        d_, e_, f_ = ca, cx, cy
                        g_, h_, i_ = da, dx, dy
                        det3 = a_ * (e_ * i_ - f_ * h_) - b_ * (d_ * i_ - f_ * g_) + c_ * (d_ * h_ - e_ * g_)
                        
                        if det3 == 0:
                            return False
        return True
    
    def validate(points, D=1):
        """Full validation for integer points."""
        if not all_dist_rational_int(points, D):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    # Verify the base set is valid
    if validate(base_points, 1):
        best_set = list(base_points)
    
    def try_add_point(current_points, D=1, max_coord=300):
        """Try to find a new point to add to current_points using systematic search."""
        n = len(current_points)
        existing = set(current_points)
        
        # Spiral search from origin for better locality
        for radius in range(0, max_coord + 1):
            if time.time() > deadline - 5:
                return None
            # Check points on the square of size radius
            if radius == 0:
                candidates = [(0, 0)]
            else:
                candidates = []
                # Top and bottom edges
                for x in range(-radius, radius + 1):
                    candidates.append((x, radius))
                    candidates.append((x, -radius))
                # Left and right edges (excluding corners already added)
                for y in range(-radius + 1, radius):
                    candidates.append((radius, y))
                    candidates.append((-radius, y))
            
            for (x, y) in candidates:
                if (x, y) in existing:
                    continue
                # Quick check: distances to all existing points must be perfect squares
                valid = True
                for (px, py) in current_points:
                    dx = x - px
                    dy = y - py
                    d2 = dx * dx + D * dy * dy
                    if not is_square(d2):
                        valid = False
                        break
                if not valid:
                    continue
                # Check collinearity with all pairs
                valid = True
                for i in range(n):
                    x1, y1 = current_points[i]
                    for j in range(i + 1, n):
                        x2, y2 = current_points[j]
                        # Cross product of (x2-x1, y2-y1) and (x-x1, y-y1)
                        if (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                # Check concyclicity with all triples
                valid = True
                for i in range(n):
                    x1, y1 = current_points[i]
                    a1 = x1 * x1 + y1 * y1
                    for j in range(i + 1, n):
                        x2, y2 = current_points[j]
                        a2 = x2 * x2 + y2 * y2
                        for k in range(j + 1, n):
                            x3, y3 = current_points[k]
                            a3 = x3 * x3 + y3 * y3
                            # Check if (x,y) is concyclic with points i,j,k
                            # Use the same 3x3 det approach
                            ba = a2 - a1
                            bx = x2 - x1
                            by = y2 - y1
                            ca = a3 - a1
                            cx = x3 - x1
                            cy = y3 - y1
                            da = x * x + y * y - a1
                            dx_ = x - x1
                            dy_ = y - y1
                            a_, b_, c_ = ba, bx, by
                            d_, e_, f_ = ca, cx, cy
                            g_, h_, i_ = da, dx_, dy_
                            det3 = a_ * (e_ * i_ - f_ * h_) - b_ * (d_ * i_ - f_ * g_) + c_ * (d_ * h_ - e_ * g_)
                            if det3 == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                return current_points + [(x, y)]
        return None
    
    # Try to add a 6th point
    if time.time() < deadline - 20:
        result = try_add_point(best_set, 1, 250)
        if result is not None:
            best_set = result
    
    # If we found 6 points, try to find a 7th
    if len(best_set) >= 6 and time.time() < deadline - 10:
        result = try_add_point(best_set, 1, 200)
        if result is not None:
            best_set = result
    
    # If we found 7 points, try to find an 8th
    if len(best_set) >= 7 and time.time() < deadline - 5:
        result = try_add_point(best_set, 1, 150)
        if result is not None:
            best_set = result
    
    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
