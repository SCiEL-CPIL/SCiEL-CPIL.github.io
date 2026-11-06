"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
import time

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    line_points = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            key = (A, B, C)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    return ordinary

def _is_valid(points):
    """Check if points are valid: 12 distinct, |x|,|y| <= 10^9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for p in points:
        if len(p) != 2:
            return False
        x, y = p
        if not isinstance(x, int) or not isinstance(y, int):
            return False
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    A = y1 - y0
    B = x0 - x1
    C = x1 * y0 - x0 * y1
    for i in range(2, len(points)):
        x, y = points[i]
        if A * x + B * y + C != 0:
            return True
    return False

def _search_3x4_grid_variants():
    """Search over 3x4 grid variants with small integer shifts."""
    best_points = None
    best_count = float('inf')
    
    # Try 4 columns x 3 rows with various column spacings and row shifts
    for c1 in range(0, 4):
        for c2 in range(0, 4):
            for c3 in range(0, 4):
                for r1 in range(-2, 3):
                    for r2 in range(-2, 3):
                        # Columns at x=0, x=1+c1, x=2+c1+c2, x=3+c1+c2+c3
                        x0 = 0
                        x1 = 1 + c1
                        x2 = 2 + c1 + c2
                        x3 = 3 + c1 + c2 + c3
                        # Rows at y=0, y=1+r1, y=2+r1+r2
                        y0 = 0
                        y1 = 1 + r1
                        y2 = 2 + r1 + r2
                        
                        # Check all x distinct and all y distinct
                        xs = [x0, x1, x2, x3]
                        ys = [y0, y1, y2]
                        if len(set(xs)) != 4 or len(set(ys)) != 3:
                            continue
                        
                        pts = [(xi, yj) for xi in xs for yj in ys]
                        if not _is_valid(pts):
                            continue
                        cnt = _count_ordinary(pts)
                        if cnt < best_count:
                            best_count = cnt
                            best_points = pts
                        if best_count <= 6:
                            return best_points, best_count
    
    return best_points, best_count

def _search_pappus_extended():
    """Try Pappus-like configurations extended to 12 points."""
    best_points = None
    best_count = float('inf')
    
    # Pappus configuration: 9 points with 0 ordinary lines
    # Extended: add 3 more points strategically
    # Base Pappus points (one standard embedding)
    # Lines: {a,b,c}, {d,e,f}, {a,d,g}, {b,e,h}, {c,f,i}
    # Pappus: g,h,i collinear
    
    # Try various Pappus embeddings
    pappus_configs = [
        # Standard Pappus: two triangles
        [(0,0),(1,0),(2,0), (0,1),(1,2),(2,3), (0,2),(1,1),(2,4)],
        # Another embedding
        [(0,0),(1,1),(2,2), (0,1),(1,0),(2,1), (0,2),(1,3),(2,4)],
    ]
    
    for base in pappus_configs:
        # Try adding 3 points at various positions
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                for dx2 in range(-3, 4):
                    for dy2 in range(-3, 4):
                        for dx3 in range(-3, 4):
                            for dy3 in range(-3, 4):
                                pts = base + [(dx,dy),(dx2,dy2),(dx3,dy3)]
                                if not _is_valid(pts):
                                    continue
                                cnt = _count_ordinary(pts)
                                if cnt < best_count:
                                    best_count = cnt
                                    best_points = pts
                                if best_count <= 6:
                                    return best_points, best_count
    
    return best_points, best_count

def _search_two_line_configs():
    """Search over configurations with points on 2-3 parallel lines."""
    best_points = None
    best_count = float('inf')
    
    # Two lines with 6 points each
    for s1 in range(1, 4):  # spacing on line 1
        for s2 in range(1, 4):  # spacing on line 2
            for offset in range(-5, 6):
                line1 = [(i * s1, 0) for i in range(6)]
                line2 = [(i * s2 + offset, 1) for i in range(6)]
                pts = line1 + line2
                if not _is_valid(pts):
                    continue
                cnt = _count_ordinary(pts)
                if cnt < best_count:
                    best_count = cnt
                    best_points = pts
                if best_count <= 6:
                    return best_points, best_count
    
    # Three lines with 4 points each
    for s1 in range(1, 3):
        for s2 in range(1, 3):
            for s3 in range(1, 3):
                for off2 in range(-3, 4):
                    for off3 in range(-3, 4):
                        line1 = [(i * s1, 0) for i in range(4)]
                        line2 = [(i * s2 + off2, 1) for i in range(4)]
                        line3 = [(i * s3 + off3, 2) for i in range(4)]
                        pts = line1 + line2 + line3
                        if not _is_valid(pts):
                            continue
                        cnt = _count_ordinary(pts)
                        if cnt < best_count:
                            best_count = cnt
                            best_points = pts
                        if best_count <= 6:
                            return best_points, best_count
    
    return best_points, best_count

def construct():
    start_time = time.time()
    deadline = 120.0
    
    best_points = None
    best_count = float('inf')
    
    # First try the known 4x3 grid
    pts_grid = [(i, j) for i in range(4) for j in range(3)]
    if _is_valid(pts_grid):
        cnt = _count_ordinary(pts_grid)
        if cnt < best_count:
            best_count = cnt
            best_points = pts_grid
        if best_count <= 6:
            return best_points
    
    # Search 3x4 grid variants
    if time.time() < deadline - 30:
        pts, cnt = _search_3x4_grid_variants()
        if pts is not None and cnt < best_count:
            best_count = cnt
            best_points = pts
        if best_count <= 6:
            return best_points
    
    # Search two-line and three-line configurations
    if time.time() < deadline - 15:
        pts, cnt = _search_two_line_configs()
        if pts is not None and cnt < best_count:
            best_count = cnt
            best_points = pts
        if best_count <= 6:
            return best_points
    
    # Search Pappus-extended configurations
    if time.time() < deadline - 5:
        pts, cnt = _search_pappus_extended()
        if pts is not None and cnt < best_count:
            best_count = cnt
            best_points = pts
    
    if best_points is not None:
        return best_points
    
    # Fallback: 4x3 grid
    return [[x, y] for x in range(4) for y in range(3)]
# EVOLVE-BLOCK-END
