import itertools
import random
from math import gcd

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
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(A, gcd(B, C))
            if g != 0:
                A //= g
                B //= g
                C //= g
            else:
                A, B, C = 0, 0, 0
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            lines.add((A, B, C))
    
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
    # We will use Simulated Annealing on a small grid to find the minimum.
    # If we find 6, we return it.
    # The coordinates can be scaled up if needed, but small integers are preferred.
    
    import random
    random.seed(42)
    
    best_points = None
    best_count = float('inf')
    
    # Initial guess: 3x3 grid + 3 points
    # Try a few structured starts
    starts = []
    
    # 3x3 grid
    grid = [[x, y] for x in range(3) for y in range(3)]
    # Add 3 points far away to break symmetry? Or nearby?
    # Let's just start with random restarts in a small box.
    
    N = 12
    LIMIT = 20 # Search in [-20, 20]
    
    def is_valid(points):
        if len(set(tuple(p) for p in points)) != N:
            return False
        # Check not all collinear
        x0, y0 = points[0]
        x1, y1 = points[1]
        for i in range(2, N):
            xi, yi = points[i]
            if (xi - x0) * (y1 - y0) != (yi - y0) * (x1 - x0):
                return True
        return False

    # Simulated Annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.01
    iterations = 50000 # Total iterations across restarts
    
    for restart in range(20):
        # Initialize
        if restart == 0:
            # Start with 3x3 grid + 3 random
            pts = [[x, y] for x in range(3) for y in range(3)]
            for _ in range(3):
                pts.append([random.randint(-LIMIT, LIMIT), random.randint(-LIMIT, LIMIT)])
        else:
            pts = []
            while len(pts) < N:
                p = [random.randint(-LIMIT, LIMIT), random.randint(-LIMIT, LIMIT)]
                if p not in pts:
                    pts.append(p)
        
        if not is_valid(pts):
            continue
            
        current_count = _count_ordinary_lines(pts)
        current_pts = [p[:] for p in pts]
        
        if current_count < best_count:
            best_count = current_count
            best_points = [p[:] for p in current_pts]
            if best_count == 6:
                return best_points
        
        # Annealing loop
        for i in range(iterations // 20):
            if T < min_T:
                break
            # Pick a point to move
            idx = random.randint(0, N - 1)
            old_x, old_y = current_pts[idx]
            # Random move
            new_x = random.randint(-LIMIT, LIMIT)
            new_y = random.randint(-LIMIT, LIMIT)
            
            new_pts = [p[:] for p in current_pts]
            new_pts[idx] = [new_x, new_y]
            
            if not is_valid(new_pts):
                T *= cooling
                continue
                
            new_count = _count_ordinary_lines(new_pts)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                current_pts = new_pts
                current_count = new_count
                
                if current_count < best_count:
                    best_count = current_count
                    best_points = [p[:] for p in current_pts]
                    if best_count == 6:
                        return best_points
            
            T *= cooling
    
    return best_points
