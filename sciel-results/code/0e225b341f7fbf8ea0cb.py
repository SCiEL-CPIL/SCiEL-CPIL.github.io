import itertools

def _count_ordinary_lines(points):
    """Count the number of lines that contain exactly 2 of the given points."""
    n = len(points)
    if n < 2:
        return 0
    lines = set()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Canonical line representation: (A, B, C) for Ax + By + C = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            # Normalize
            from math import gcd
            g = gcd(A, gcd(B, C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                A, B, C = 0, 0, 0
            # Ensure canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            lines.add((A, B, C))
    
    # Count lines that contain exactly 2 points
    ordinary = 0
    for line in lines:
        A, B, C = line
        cnt = 0
        for x, y in points:
            if A * x + B * y + C == 0:
                cnt += 1
        if cnt == 2:
            ordinary += 1
    return ordinary

def construct():
    # Start with the current 12-point configuration
    base_points = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
        [0, 3],
        [3, 3]
    ]
    
    best_points = base_points
    best_count = _count_ordinary_lines(base_points)
    
    # Try moving each point to nearby integer positions
    # to see if we can reduce the ordinary line count
    for idx in range(12):
        x0, y0 = base_points[idx]
        # Try a small neighborhood
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x0 + dx, y0 + dy
                # Check bounds
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                # Create new points list
                new_points = [p[:] for p in base_points]
                new_points[idx] = [nx, ny]
                
                # Check distinctness
                if len(set(tuple(p) for p in new_points)) != 12:
                    continue
                
                # Check not all collinear
                collinear = True
                for i in range(1, 12):
                    if not (new_points[i][0] - new_points[0][0]) * (new_points[1][1] - new_points[0][1]) == \
                       (new_points[i][1] - new_points[0][1]) * (new_points[1][0] - new_points[0][0]):
                        collinear = False
                        break
                if collinear:
                    continue
                
                cnt = _count_ordinary_lines(new_points)
                if cnt < best_count:
                    best_count = cnt
                    best_points = new_points
    
    return best_points
