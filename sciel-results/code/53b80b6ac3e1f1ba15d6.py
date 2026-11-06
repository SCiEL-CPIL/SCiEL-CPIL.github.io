"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict
from math import gcd

def _line_key(p1, p2):
    """Return a canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Reduce by gcd
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    # Canonical direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: ax + by = c where a=dy, b=-dx
    a = dy
    b = -dx
    c = a * x1 + b * y1
    # Normalize sign: make (a, b) canonical
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    # Reduce by gcd of (a, b, c)
    g2 = gcd(gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _count_exact_four(points):
    """Count lines containing exactly 4 of the given points. Returns (count, has_five)."""
    n = len(points)
    if n < 4:
        return 0, False
    
    line_to_indices = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_to_indices[key].add(i)
                line_to_indices[key].add(j)
    
    count = 0
    has_five = False
    for key, indices in line_to_indices.items():
        cnt = len(indices)
        if cnt == 4:
            count += 1
        elif cnt >= 5:
            has_five = True
    return count, has_five

def _is_valid(points):
    """Check: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, has_five = _count_exact_four(points)
    return not has_five

def _score(points):
    """Return (count, valid)."""
    if not _is_valid(points):
        return 0, False
    count, _ = _count_exact_four(points)
    return count, True

def _build_gf16_points():
    """
    Construct 16 points from GF(16) using a primitive element representation.
    
    GF(16) = GF(2)[x] / (x^4 + x + 1)
    Let alpha be a root of x^4 + x + 1 = 0, so alpha^4 = alpha + 1.
    
    Elements of GF(16) can be represented as polynomials in alpha of degree < 4.
    We use a 4x4 grid where each point (i,j) corresponds to an element.
    
    In the finite field, every line contains exactly 4 points (since 16 = 4^2).
    We map these to integer coordinates using a representation that preserves
    the collinearity structure as much as possible.
    
    Strategy: Use the polynomial representation. Each element a + b*alpha + c*alpha^2 + d*alpha^3
    can be mapped to integer coordinates. We'll use a clever mapping to create
    a configuration with many 4-point lines.
    """
    # GF(16) elements represented as 4-bit vectors (coefficients of 1, alpha, alpha^2, alpha^3)
    # Multiplication table for GF(16) with primitive polynomial x^4 + x + 1
    
    # Define multiplication in GF(16)
    def gf16_mult(a, b):
        """Multiply two elements in GF(16) represented as 4-bit integers."""
        result = 0
        a = int(a)
        b = int(b)
        for i in range(4):
            if b & (1 << i):
                # Multiply a by alpha^i
                temp = a
                for _ in range(i):
                    temp <<= 1
                    if temp & 0x10:  # if alpha^4 appears, reduce
                        temp ^= 0x11  # alpha^4 = alpha + 1, so x^4 = x + 1
                result ^= temp
        return result & 0x0F
    
    # Generate all 16 elements
    elements = list(range(16))
    
    # In GF(16), the 4x4 grid can be formed by:
    # Point (i, j) = (alpha^i, alpha^j) for i, j in {0, 1, 2, 3}
    # But we need to handle 0 separately.
    
    # Alternative: Use the fact that GF(16) has a Singer cycle.
    # The 16 points can be arranged as a 4x4 grid where:
    # - 4 rows (lines with slope 0)
    # - 4 columns (lines with infinite slope)  
    # - 8 diagonal lines (lines with other slopes)
    # Total: 16 lines, each containing exactly 4 points
    
    # Map GF(16) elements to integer coordinates:
    # We'll use a representation where:
    # Element e = a0 + a1*alpha + a2*alpha^2 + a3*alpha^3
    # Map to (x, y) = (a0 + 2*a1 + 4*a2, a3 + 2*a0 + 4*a1)
    # This is a linear transformation that should preserve some structure
    
    points = []
    for e in elements:
        a0 = e & 1
        a1 = (e >> 1) & 1
        a2 = (e >> 2) & 1
        a3 = (e >> 3) & 1
        x = a0 + 2 * a1 + 4 * a2
        y = a3 + 2 * a0 + 4 * a1
        points.append([x, y])
    
    # This gives points in [0,7] x [0,7], which is within bounds.
    # But we need to check if this preserves the 4-point line structure.
    
    # Let's try a different mapping that might work better:
    # Use the fact that GF(16) can be represented as a 4x4 grid over GF(4)
    # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
    
    # Actually, let's try a direct approach:
    # The 16 points of GF(16) as a 4x4 grid:
    # Point (i,j) for i,j in {0,1,2,3}
    # In the field, a line is determined by two points, and contains exactly 4 points.
    
    # Map (i,j) to integer coordinates using a nonlinear transformation
    # that might preserve collinearity patterns:
    
    # Try: x = i * 5 + j * 3, y = i * 7 + j * 2
    # This is an affine transformation, which preserves collinearity exactly.
    # But it will create a parallelogram grid, which has the same line structure as 4x4 grid.
    
    # The 4x4 integer grid has:
    # - 4 horizontal lines (y=0,1,2,3) each with 4 points
    # - 4 vertical lines (x=0,1,2,3) each with 4 points
    # - Diagonal lines: need to count
    # Total 4-point lines in 4x4 grid: Let's compute
    
    # Actually, for a 4x4 grid, the number of lines with exactly 4 points:
    # Horizontal: 4
    # Vertical: 4
    # Diagonals with slope 1: (0,0)-(1,1)-(2,2)-(3,3), (0,1)-(1,2)-(2,3), etc.
    # Only the main diagonal has 4 points: (0,0),(1,1),(2,2),(3,3)
    # Anti-diagonal: (0,3),(1,2),(2,1),(3,0) - also 4 points
    # Other diagonals have at most 3 points
    # So total 4-point lines in 4x4 grid: 4 + 4 + 2 = 10
    
    # But GF(16) has 16 lines with 4 points each!
    # The difference is that in GF(16), "lines" include lines that wrap around.
    # When we map to integers, we lose the wrap-around structure.
    
    # To get more 4-point lines, we need a different mapping.
    
    # Let's try using the finite field structure more directly.
    # In GF(16), the 16 points form a projective plane-like structure.
    
    # New approach: Use the fact that we can place points on a conic or other curve
    # in the integer plane that has many 4-point lines.
    
    # Actually, let's try a spiral or other non-grid arrangement.
    
    # For now, return the simple 4x4 grid and let local search improve it.
    return [[i, j] for i in range(4) for j in range(4)]

def _local_search(points, deadline):
    """Swap-based local search to improve the configuration."""
    best = list(points)
    best_score, _ = _score(best)
    
    n = 16
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single-point moves
        for i in range(n):
            if time.time() >= deadline:
                break
            x0, y0 = best[i]
            # Try nearby positions
            for dx in range(-5, 6):
                if time.time() >= deadline:
                    break
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    # Check if new position is already occupied
                    if [nx, ny] in best:
                        continue
                    trial = list(best)
                    trial[i] = [nx, ny]
                    s, valid = _score(trial)
                    if valid and s > best_score:
                        best = trial
                        best_score = s
                        improved = True
                        x0, y0 = nx, ny  # update for further moves from new pos
                        break
                if time.time() >= deadline:
                    break
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe within 150s limit
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best, best_score = _local_search(points, deadline)
    
    # Try a few different starting configurations
    # Configuration 2: shifted grid
    points2 = [[x + 1, y + 1] for x in range(4) for y in range(4)]
    b2, s2 = _local_search(points2, deadline)
    if s2 > best_score:
        best, best_score = b2, s2
    
    # Configuration 3: non-square grid (e.g., 4x4 but with wider spacing)
    points3 = [[x * 3, y * 3] for x in range(4) for y in range(4)]
    b3, s3 = _local_search(points3, deadline)
    if s3 > best_score:
        best, best_score = b3, s3
    
    # Configuration 4: 4x4 grid with one point moved
    points4 = [[x, y] for x in range(4) for y in range(4)]
    points4[0] = [5, 5]
    b4, s4 = _local_search(points4, deadline)
    if s4 > best_score:
        best, best_score = b4, s4
    
    # Configuration 5: Try a different arrangement
    # Use a 4x4 grid but with non-uniform spacing
    points5 = [[x * (x+1), y * (y+1)] for x in range(4) for y in range(4)]
    b5, s5 = _local_search(points5, deadline)
    if s5 > best_score:
        best, best_score = b5, s5
    
    # Final verification
    final_score, valid = _score(best)
    if not valid or final_score < best_score:
        # Fallback to original grid
        best = [[x, y] for x in range(4) for y in range(4)]
    
    return best
# EVOLVE-BLOCK-END
