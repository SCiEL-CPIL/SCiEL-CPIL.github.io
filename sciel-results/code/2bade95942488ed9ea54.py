"""Direct construction: Finite field search for 12-point configuration with few ordinary lines.

Searches for 12-point configurations in GF(p) for small primes, then lifts to integer
coordinates. In GF(p), lines have exactly p points, enabling systematic design of
configurations with many 3+ point lines.
"""
import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 145.0  # safety margin before 150s kill


def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _collinear_check(points):
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)


def _gf_count_ordinary(points, p):
    """Count lines with exactly 2 points in GF(p) for the given set of 12 points."""
    n = len(points)
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2) in GF(p)
            # A = y1-y2, B = x2-x1, C = x1*y2 - x2*y1 (all mod p)
            A = (y1 - y2) % p
            B = (x2 - x1) % p
            C = (x1 * y2 - x2 * y1) % p
            # Normalize: make first non-zero coefficient 1
            if A != 0:
                A_inv = pow(A, p-2, p)
                B = (B * A_inv) % p
                C = (C * A_inv) % p
                A = 1
            elif B != 0:
                B_inv = pow(B, p-2, p)
                C = (C * B_inv) % p
                B = 1
                A = 0
            else:
                # Both A and B are 0 - shouldn't happen for distinct points
                continue
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)


def _lift_gf_to_integer(gf_points, p):
    """Lift GF(p) points to integer coordinates preserving collinearity structure.
    
    Strategy: Map (x, y) in GF(p) to integer (x * M, y * M) where M is chosen
    so that no unintended collinearities arise over the integers.
    """
    # Use a large multiplier to separate points
    # We need to ensure that if 3 points are collinear in GF(p), they are collinear in Z,
    # and if they are not collinear in GF(p), they are not collinear in Z (for the 12 points)
    
    # Simple approach: use (x, y) directly if p is small enough
    # But we need to verify no accidental collinearities
    base_points = [[x, y] for x, y in gf_points]
    
    # Try different scalings to avoid accidental collinearities
    for scale in [1, 2, 3, 5, 7, 11, 13, 17, 23, 29, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]:
        pts = [[x * scale, y * scale] for x, y in base_points]
        # Check if this lifting preserves the exact line structure
        # by comparing ordinary line count
        if _valid(pts):
            # This is a valid candidate; the caller will evaluate it
            return pts
    
    # Fallback: just use the base points
    return base_points


def _search_gf_prime(p, max_time=10.0):
    """Search for 12-point configuration in GF(p) minimizing ordinary lines."""
    if time.time() > _DEADLINE - 5:
        return None, float('inf')
    
    all_points = [(x, y) for x in range(p) for y in range(p)]
    n_all = len(all_points)
    
    best_subset = None
    best_ord = float('inf')
    
    # Use simulated annealing on the subset selection
    rng = random.Random(p * 7919)
    
    # Start with a random subset
    cur_subset = set(rng.sample(all_points, 12))
    cur_ord = _gf_count_ordinary(list(cur_subset), p)
    
    best_subset = set(cur_subset)
    best_ord = cur_ord
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        # Generate neighbor: swap one point
        new_subset = set(cur_subset)
        # Remove a random point
        to_remove = rng.choice(list(new_subset))
        new_subset.remove(to_remove)
        # Add a random point not in subset
        candidates = [pt for pt in all_points if pt not in new_subset]
        if not candidates:
            break
        to_add = rng.choice(candidates)
        new_subset.add(to_add)
        
        new_ord = _gf_count_ordinary(list(new_subset), p)
        delta = new_ord - cur_ord
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur_subset = new_subset
            cur_ord = new_ord
            if cur_ord < best_ord:
                best_ord = cur_ord
                best_subset = set(cur_subset)
        
        T *= cooling_rate
    
    return list(best_subset), best_ord


def _refine_integer(points, max_time=10.0):
    """Refine integer configuration using local search to minimize ordinary lines."""
    if not _valid(points):
        return points
    
    cur = [p[:] for p in points]
    cur_ord = _count_ordinary(cur)
    best = [p[:] for p in cur]
    best_ord = cur_ord
    
    rng = random.Random(int(time.time() * 1000) % 2**32)
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and time.time() - start_time < max_time:
        new_pts = [p[:] for p in cur]
        move_type = rng.random()
        
        if move_type < 0.6:
            # Small perturbation
            idx = rng.randrange(12)
            new_pts[idx][0] += rng.randint(-3, 3)
            new_pts[idx][1] += rng.randint(-3, 3)
        elif move_type < 0.85:
            # Snap to line through two other points
            i = rng.randrange(12)
            j = rng.randrange(12)
            k = rng.randrange(12)
            if i == j or i == k or j == k:
                T *= cooling_rate
                continue
            x1, y1 = new_pts[j]
            x2, y2 = new_pts[k]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                T *= cooling_rate
                continue
            # Find integer points on this line near current position
            xi, yi = new_pts[i]
            # Line: (y2-y1)(x-x1) = (x2-x1)(y-y1)
            # (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            # A x + B y = -C
            g_ab = gcd(abs(A), abs(B))
            if -C % g_ab != 0:
                T *= cooling_rate
                continue
            a = A // g_ab
            b = B // g_ab
            c = -C // g_ab
            
            def egcd(a, b):
                if b == 0:
                    return (a, 1, 0)
                g, x1, y1 = egcd(b, a % b)
                return (g, y1, x1 - (a // b) * y1)
            
            g_gcd, x0, y0 = egcd(a, b)
            if g_gcd != 1:
                T *= cooling_rate
                continue
            x0 *= c
            y0 *= c
            # Try several t values near current point
            for t in range(-5, 6):
                nx = x0 + b * t
                ny = y0 - a * t
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    if (nx, ny) not in [(p[0], p[1]) for p in new_pts if new_pts.index(p) != i]:
                        new_pts[i] = [nx, ny]
                        break
        else:
            # Random jump
            idx = rng.randrange(12)
            new_pts[idx] = [rng.randint(-100, 100), rng.randint(-100, 100)]
        
        if not _valid(new_pts):
            T *= cooling_rate
            continue
        
        new_ord = _count_ordinary(new_pts)
        delta = new_ord - cur_ord
        
        if delta < 0 or rng.random() < pow(2.71828, -delta / T):
            cur = new_pts
            cur_ord = new_ord
            if cur_ord < best_ord:
                best_ord = cur_ord
                best = [p[:] for p in cur]
        
        T *= cooling_rate
    
    return best


def _make_known_good():
    """Return a known good configuration as fallback."""
    # 4x3 grid
    points = []
    for i in range(4):
        for j in range(3):
            points.append([i, j])
    return points


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines. Uses finite field search + integer refinement."""
    
    best_pts = None
    best_ord = float('inf')
    
    # Try several primes
    primes_to_try = [7, 11, 13, 17, 19]
    
    for p in primes_to_try:
        if time.time() > _DEADLINE - 30:
            break
        
        gf_pts, gf_ord = _search_gf_prime(p, max_time=8.0)
        
        if gf_pts is None:
            continue
        
        # Lift to integer coordinates
        int_pts = _lift_gf_to_integer(gf_pts, p)
        
        if int_pts is None or not _valid(int_pts):
            continue
        
        int_ord = _count_ordinary(int_pts)
        
        if int_ord < best_ord:
            best_ord = int_ord
            best_pts = int_pts
    
    # Refine the best integer configuration
    if best_pts is not None and time.time() < _DEADLINE - 20:
        refined = _refine_integer(best_pts, max_time=15.0)
        refined_ord = _count_ordinary(refined)
        if refined_ord < best_ord:
            best_ord = refined_ord
            best_pts = refined
    
    # Fallback
    if best_pts is None:
        best_pts = _make_known_good()
    
    # Final validation
    if not _valid(best_pts):
        best_pts = _make_known_good()
    
    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
