import itertools
import time

def count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    # Map from line representation to count of points on that line
    # A line can be represented by (A, B, C) where Ax + By + C = 0, normalized.
    # Or simply by the pair of points that define it, but we need to merge lines.
    # Better: Use a dictionary keyed by the normalized line equation.
    
    line_counts = {}
    
    # Iterate over all pairs to define lines
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Calculate line coefficients A, B, C for Ax + By + C = 0
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            # Normalize (A, B, C)
            # Find gcd of A, B, C to make the representation unique
            # Handle zeros
            g = 0
            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return abs(a)
            
            g = gcd(A, B)
            if g == 0:
                g = gcd(g, C)
            
            if g != 0:
                A //= g
                B //= g
                C //= g
            
            # Ensure a consistent sign (e.g., first non-zero of A, B, C is positive)
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            key = (A, B, C)
            line_counts[key] = 0
            
    # Now count points on each unique line
    for key in line_counts:
        A, B, C = key
        count = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                count += 1
        line_counts[key] = count
        
    ordinary = 0
    for count in line_counts.values():
        if count == 2:
            ordinary += 1
            
    return ordinary

def generate_grid(r, c):
    return [[i, j] for i in range(r) for j in range(c)]

def generate_pencil(n_lines, pts_per_line):
    # Use distinct integer directions
    dirs = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (3, 1), (1, 3), (2, -1), (1, -2)]
    pts = []
    for i in range(n_lines):
        ux, uy = dirs[i]
        for d in range(1, pts_per_line + 1):
            pts.append([d * ux, d * uy])
    return pts

def construct():
    start_time = time.time()
    deadline = 110.0 # 120s limit, keep margin
    
    candidates = []
    
    # 1. 3x4 Grid
    candidates.append(generate_grid(3, 4))
    # 2. 4x3 Grid
    candidates.append(generate_grid(4, 3))
    # 3. 2x6 Grid
    candidates.append(generate_grid(2, 6))
    # 4. 6x2 Grid
    candidates.append(generate_grid(6, 2))
    # 5. 1x12 Line (All collinear, invalid per "not all collinear")
    # Skip.
    
    # 6. Pencil with 6 lines, 2 points each
    candidates.append(generate_pencil(6, 2))
    # 7. Pencil with 4 lines, 3 points each
    candidates.append(generate_pencil(4, 3))
    # 8. Pencil with 3 lines, 4 points each
    candidates.append(generate_pencil(3, 4))
    
    # 9. A "cross" shape or similar
    # Center (0,0), arms of length 3?
    # (0,0) is 1 point. 4 arms * 3 points = 12 points. Total 13.
    # Remove center? 12 points on 4 lines through origin, 3 points each (excluding origin).
    # This is similar to pencil(4,3) but centered at 0?
    # My generate_pencil puts points at d=1,2,3. So (1,0), (2,0), (3,0).
    # This is a "half-pencil".
    
    # Let's try a specific known good configuration if we can find one.
    # The "Böröczky" configuration for 9 points has 0 ordinary lines? No, 9 points, 0 ordinary is impossible?
    # Actually, 9 points can have 0 ordinary lines (e.g. 3x3 grid has 0 ordinary? 
    # 3x3 grid:
    # 3 horiz (3 pts), 3 vert (3 pts), 2 diag (3 pts).
    # Other lines: (0,0)-(1,2) slope 2. (0,1)-(1,3) out of bounds.
    # In 3x3 grid, points (0,0)..(2,2).
    # Line (0,0)-(1,2): contains (0,0), (1,2). Is there a 3rd? No. So it's ordinary.
    # So 3x3 grid has ordinary lines.
    
    # Let's just evaluate the candidates.
    best_points = None
    best_score = float('inf')
    
    for pts in candidates:
        # Check validity: distinct, integer, bounds
        if len(set(map(tuple, pts))) != 12:
            continue
        if any(abs(x) > 10**9 or abs(y) > 10**9 for x, y in pts):
            continue
        # Check not all collinear
        if len(pts) > 1:
            # Check if all points are on the line defined by first two
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            all_collinear = True
            for x, y in pts[2:]:
                if A * x + B * y + C != 0:
                    all_collinear = False
                    break
            if all_collinear:
                continue
        
        score = count_ordinary_lines(pts)
        if score < best_score:
            best_score = score
            best_points = [list(p) for p in pts]
            
        # Time check
        if time.time() > deadline:
            break
            
    if best_points is None:
        # Fallback: simple grid
        best_points = generate_grid(3, 4)
        
    return best_points
