import math
import time

def construct():
    points = [[x, y] for x in range(4) for y in range(3)]
    
    def get_lines(pts):
        n = len(pts)
        lines = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                A = dy
                B = -dx
                C = x1 * dy - y1 * dx
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A, B, C = -A, -B, -C
                if A == 0 and B == 0:
                    continue
                g = math.gcd(A, B)
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        return lines

    def count_ordinary(lines):
        return sum(1 for pts in lines.values() if len(pts) == 2)

    best_points = points
    best_lines = get_lines(points)
    best_count = count_ordinary(best_lines)
    
    start_time = time.time()
    deadline = start_time + 110
    
    while time.time() < deadline:
        improved = False
        lines = get_lines(points)
        ordinary_lines = [key for key, pts in lines.items() if len(pts) == 2]
        
        # Try to move a point onto an ordinary line to make it a 3-point line
        for key in ordinary_lines:
            A, B, C = key
            pts_on_line = lines[key]
            i, j = list(pts_on_line)
            
            # Try to move each other point k onto this line
            for k in range(12):
                if k in pts_on_line:
                    continue
                
                px, py = points[k]
                
                # Find integer point on Ax + By + C = 0 closest to (px, py)
                # A x + B y = -C
                g = math.gcd(A, B)
                if -C % g != 0:
                    continue
                
                A_prime = A // g
                B_prime = B // g
                C_prime = -C // g
                
                # Find particular solution
                # A' x + B' y = C'
                # Extended Euclidean
                def egcd(a, b):
                    if b == 0:
                        return (1, 0, a)
                    else:
                        x, y, g = egcd(b, a % b)
                        return (y, x - (a // b) * y, g)
                
                x0, y0, _ = egcd(A_prime, B_prime)
                x0 *= C_prime
                y0 *= C_prime
                
                # General solution:
                # x = x0 + t * B_prime
                # y = y0 - t * A_prime
                
                # Minimize (x - px)^2 + (y - py)^2
                # (x0 + t B - px)^2 + (y0 - t A - py)^2
                # Let dx = px - x0, dy = py - y0
                # (t B - dx)^2 + (-t A - dy)^2
                # t^2 (B^2 + A^2) - t (2 B dx - 2 A dy) + (dx^2 + dy^2)
                # Optimal t = (2 B dx - 2 A dy) / (2 (A^2 + B^2))
                
                num = B_prime * (px - x0) - A_prime * (py - y0)
                den = A_prime * A_prime + B_prime * B_prime
                if den == 0:
                    continue
                t = num / den
                t1 = int(t)
                t2 = t1 + 1
                
                best_t = None
                best_dist = float('inf')
                best_pt = None
                
                for t_cand in [t1, t2]:
                    nx = x0 + t_cand * B_prime
                    ny = y0 - t_cand * A_prime
                    dist = (nx - px)**2 + (ny - py)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_t = t_cand
                        best_pt = (nx, ny)
                
                if best_pt is None:
                    continue
                
                nx, ny = best_pt
                if abs(nx) > 10**9 or abs(ny) > 10**9:
                    continue
                
                # Check if this point is already taken
                if any(nx == p[0] and ny == p[1] for p in points):
                    continue
                
                # Try the move
                new_points = points.copy()
                new_points[k] = [nx, ny]
                new_lines = get_lines(new_points)
                new_count = count_ordinary(new_lines)
                
                if new_count < best_count:
                    best_count = new_count
                    best_points = new_points
                    points = new_points
                    improved = True
                    break # Restart loop with new configuration
            
            if improved:
                break
        
        if not improved:
            # If no improvement via snapping, try small random shifts?
            # Or break to avoid infinite loop.
            break
            
    return best_points
