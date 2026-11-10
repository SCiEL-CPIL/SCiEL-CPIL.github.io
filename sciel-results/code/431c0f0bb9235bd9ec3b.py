import random
import time
from itertools import combinations

def construct():
    # 16 points
    n = 16
    # Initial state: 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    
    def get_score(pts):
        """Returns (is_valid, score)"""
        # Check validity: no 5 collinear
        # And calculate score: count lines with exactly 4 points
        
        # Map from line representation to count
        # Line rep: (A, B, C) normalized such that gcd(A,B,C)=1 and A>0 or A==0 and B>0
        lines = {}
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                
                # Line equation: (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
                A = y2 - y1
                B = -(x2 - x1)
                C = x2*y1 - x1*y2
                
                # Normalize
                g = __import__('math').gcd(A, __import__('math').gcd(B, C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                else:
                    # This shouldn't happen for distinct points
                    continue
                    
                # Canonical sign
                if A < 0 or (A == 0 and B < 0):
                    A = -A
                    B = -B
                    C = -C
                elif A == 0 and B == 0:
                    continue # Same point, shouldn't happen
                
                key = (A, B, C)
                
                if key not in lines:
                    lines[key] = 0
                lines[key] += 1
        
        # Each line with k points is counted k*(k-1)/2 times in the pairs.
        # So if count is C, the number of points on the line is k such that k*(k-1)/2 = C.
        # k=2 -> 1
        # k=3 -> 3
        # k=4 -> 6
        # k=5 -> 10
        
        score = 0
        is_valid = True
        
        for key, count in lines.items():
            # Solve k*(k-1)/2 = count
            # k^2 - k - 2*count = 0
            # k = (1 + sqrt(1 + 8*count)) / 2
            disc = 1 + 8 * count
            sqrt_disc = int(__import__('math').isqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                # Should not happen if points are distinct integers?
                # Actually, it always happens for integer pairs.
                # But just in case, we can iterate or just check specific cases
                pass
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != count:
                # This implies the count doesn't correspond to an integer k, which is impossible
                # for a set of distinct points.
                # However, due to normalization, distinct lines might merge? No.
                # This check is redundant if logic is correct.
                pass
            
            if k >= 5:
                is_valid = False
            if k == 4:
                score += 1
                
        return is_valid, score

    def random_point():
        return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

    def try_move_to_line(pts, k, i, j):
        """
        Try to move point k to a position on the line defined by points i and j,
        such that it becomes the 4th point on that line (or extends it).
        Returns new point or None.
        """
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        
        # Vector
        dx = x2 - x1
        dy = y2 - y1
        
        # We want an integer point (x,y) = (x1 + t*dx, y1 + t*dy) for integer t?
        # No, that only gives points on the "lattice" of the segment.
        # We need any integer point on the infinite line.
        # Line: A x + B y + C = 0
        # A = dy, B = -dx, C = x1*y2 - x2*y1
        
        A = dy
        B = -dx
        C = x1*y2 - x2*y1
        
        # Find integer solutions to A x + B y = -C
        # General solution:
        # x = x_p + (B/g) * t
        # y = y_p - (A/g) * t
        # where g = gcd(A, B)
        
        g = __import__('math').gcd(A, B)
        if g == 0: return None
        if -C % g != 0: return None # No integer solution (shouldn't happen for i,j integer)
        
        # Find particular solution (xp, yp)
        # Use extended gcd
        def egcd(a, b):
            if b == 0:
                return (1, 0, a)
            x1, y1, d = egcd(b, a % b)
            return (y1, x1 - (a // b) * y1, d)
        
        x, y, d = egcd(A, B)
        # A*x + B*y = d
        # We want A*x' + B*y' = -C
        # x' = x * (-C // d)
        # y' = y * (-C // d)
        xp = x * (-C // d)
        yp = y * (-C // d)
        
        # General solution
        # x = xp + (B//g) * t
        # y = yp - (A//g) * t
        
        step_x = B // g
        step_y = -A // g
        
        # We need to pick a t such that the point is within [-1000, 1000]
        # and is not equal to pts[i] or pts[j] or pts[k] (current pos)
        # Also, we prefer t values that are "close" to make it a nice grid-like structure?
        # Or just random t?
        
        # Let's try a few t values
        # The points i and j correspond to specific t values.
        # Let's just pick random t in a reasonable range, say [-50, 50]
        # This keeps points somewhat close to the segment.
        
        for _ in range(20):
            t = random.randint(-50, 50)
            nx = xp + step_x * t
            ny = yp + step_y * t
            
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                # Check if this point is distinct from i, j, and current k
                if [nx, ny] != pts[i] and [nx, ny] != pts[j] and [nx, ny] != pts[k]:
                    # Check if it's already used by another point?
                    # We are moving k, so it will be distinct from others if it wasn't before?
                    # We must ensure it doesn't collide with other points l != k, i, j
                    # Actually, if it collides with another point, it's still a valid move
                    # (two points at same spot? No, points must be distinct).
                    # The problem says "16 distinct integer points".
                    # So we must ensure [nx, ny] is not in pts[l] for l != k.
                    is_distinct = True
                    for l in range(n):
                        if l == k: continue
                        if pts[l] == [nx, ny]:
                            is_distinct = False
                            break
                    if is_distinct:
                        return [nx, ny]
        return None

    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for safety
    
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.999
    
    current_pts = points
    current_valid, current_score = get_score(current_pts)
    best_pts = current_pts
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        new_pts = [p[:] for p in current_pts]
        
        move_type = random.random()
        k = random.randint(0, n-1)
        
        if move_type < 0.3:
            # Random jump
            new_pts[k] = random_point()
        elif move_type < 0.7:
            # Try to complete a line with 2 other points
            # Pick 2 other points
            others = [i for i in range(n) if i != k]
            i, j = random.sample(others, 2)
            new_pos = try_move_to_line(new_pts, k, i, j)
            if new_pos:
                new_pts[k] = new_pos
            else:
                # Fallback to small random move
                new_pts[k] = [current_pts[k][0] + random.randint(-5, 5), 
                              current_pts[k][1] + random.randint(-5, 5)]
        else:
            # Small local move
            new_pts[k] = [current_pts[k][0] + random.randint(-10, 10), 
                          current_pts[k][1] + random.randint(-10, 10)]
        
        # Clamp to bounds
        new_pts[k][0] = max(-1000, min(1000, new_pts[k][0]))
        new_pts[k][1] = max(-1000, min(1000, new_pts[k][1]))
        
        new_valid, new_score = get_score(new_pts)
        
        if not new_valid:
            # Reject invalid states
            # In SA, we typically only consider valid states or penalize heavily.
            # Since validity is a hard constraint, we reject.
            # To avoid getting stuck, we might want to allow temporary invalidity?
            # No, "no five collinear" is strict.
            # If we reject all invalid, we might get stuck in a local optimum.
            # But "no 5 collinear" is a sparse constraint in the high-dimensional space.
            # Most random moves are valid.
            pass
        else:
            delta = new_score - current_score
            if delta > 0 or random.random() < __import__('math').exp(delta / T):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = [p[:] for p in current_pts]
        
        T = max(T_min, T * cooling_rate)
        
    return best_pts

# EVOLVE-BLOCK-START
def construct():
    import random
    import time
    import math
    from collections import defaultdict

    n = 16
    start_time = time.time()
    time_limit = 115.0

    def get_line_key(x1, y1, x2, y2):
        A = y2 - y1
        B = -(x2 - x1)
        C = x2*y1 - x1*y2
        g = math.gcd(A, math.gcd(B, C))
        if g != 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def compute_line_counts(points):
        """Compute pair counts per line"""
        line_counts = defaultdict(int)
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                key = get_line_key(points[i][0], points[i][1], points[j][0], points[j][1])
                line_counts[key] += 1
        return line_counts

    def get_score_from_counts(line_counts):
        """Return (is_valid, score) from line pair counts"""
        score = 0
        is_valid = True
        for key, count in line_counts.items():
            disc = 1 + 8 * count
            sqrt_disc = int(math.isqrt(disc))
            if sqrt_disc * sqrt_disc != disc:
                continue
            k = (1 + sqrt_disc) // 2
            if k * (k-1) // 2 != count:
                continue
            if k >= 5:
                is_valid = False
            if k == 4:
                score += 1
        return is_valid, score

    def get_score(pts):
        line_counts = compute_line_counts(pts)
        return get_score_from_counts(line_counts)

    def find_points_on_line(x1, y1, x2, y2, existing_set, max_t=30):
        """Find integer points on line, not in existing_set"""
        dx = x2 - x1
        dy = y2 - y1
        g = math.gcd(dx, dy)
        if g == 0:
            return []
        step_x = dx // g
        step_y = dy // g
        candidates = []
        for t in range(-max_t, max_t + 1):
            nx = x1 + t * step_x
            ny = y1 + t * step_y
            if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                pt = (nx, ny)
                if pt not in existing_set:
                    candidates.append(pt)
        return candidates

    # Start with 4x4 grid
    initial_points = [[x, y] for x in range(4) for y in range(4)]
    current_points = [p[:] for p in initial_points]
    current_line_counts = compute_line_counts(current_points)
    current_valid, current_score = get_score_from_counts(current_line_counts)

    best_points = [p[:] for p in current_points]
    best_score = current_score

    # Local search with smart moves
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995

    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        k = random.randint(0, n-1)
        old_pt = current_points[k][:]
        existing_set = set(tuple(p) for p in current_points)
        existing_set.discard(tuple(old_pt))

        move_type = random.random()
        new_pt = None

        if move_type < 0.5:
            # Try to place on a line with 2 or 3 other points to form a 4-point line
            others = [i for i in range(n) if i != k]
            i, j = random.sample(others, 2)
            x1, y1 = current_points[i]
            x2, y2 = current_points[j]
            key = get_line_key(x1, y1, x2, y2)
            cur_count = current_line_counts.get(key, 0)
            # Determine how many points are on this line
            disc = 1 + 8 * cur_count
            sqrt_disc = int(math.isqrt(disc))
            if sqrt_disc * sqrt_disc == disc:
                k_line = (1 + sqrt_disc) // 2
                if k_line * (k_line - 1) // 2 == cur_count:
                    if k_line == 3:
                        # Adding one more makes 4-point line - good!
                        candidates = find_points_on_line(x1, y1, x2, y2, existing_set, max_t=20)
                        if candidates:
                            new_pt = list(random.choice(candidates))
                    elif k_line == 2:
                        # Adding one more makes 3-point line - neutral
                        candidates = find_points_on_line(x1, y1, x2, y2, existing_set, max_t=20)
                        if candidates:
                            new_pt = list(random.choice(candidates))
        elif move_type < 0.8:
            # Small local move
            new_pt = [old_pt[0] + random.randint(-8, 8),
                      old_pt[1] + random.randint(-8, 8)]
        else:
            # Medium random move
            new_pt = [random.randint(-50, 50), random.randint(-50, 50)]

        if new_pt is None:
            new_pt = [old_pt[0] + random.randint(-5, 5),
                      old_pt[1] + random.randint(-5, 5)]

        new_pt[0] = max(-1000, min(1000, new_pt[0]))
        new_pt[1] = max(-1000, min(1000, new_pt[1]))

        # Check distinctness
        if tuple(new_pt) in existing_set:
            continue

        # Compute new line counts efficiently
        new_line_counts = dict(current_line_counts)
        # Remove old point's contributions
        for i in range(n):
            if i == k:
                continue
            key = get_line_key(old_pt[0], old_pt[1], current_points[i][0], current_points[i][1])
            new_line_counts[key] -= 1
            if new_line_counts[key] <= 0:
                del new_line_counts[key]
        # Add new point's contributions
        for i in range(n):
            if i == k:
                continue
            key = get_line_key(new_pt[0], new_pt[1], current_points[i][0], current_points[i][1])
            new_line_counts[key] = new_line_counts.get(key, 0) + 1

        new_valid, new_score = get_score_from_counts(new_line_counts)

        if not new_valid:
            continue

        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_points[k] = new_pt
            current_line_counts = new_line_counts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in current_points]

        T = max(T_min, T * cooling_rate)

    return best_points
# EVOLVE-BLOCK-END
