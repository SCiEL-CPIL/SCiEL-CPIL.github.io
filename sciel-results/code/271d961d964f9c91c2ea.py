"""Grid-based discrete search: inner hexagons on a triangular lattice with a single
global rotation; 1D search over rotation + combinatorial search over lattice patterns,
then a high-precision polish pass."""
import math
import time
import numpy as np

N = 12
DEADLINE = 150.0
START_T = time.time()

def _elapsed():
    return time.time() - START_T

def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for k in range(6):
        th = a + k * math.pi / 3.0
        pts.append((cx + s * math.cos(th), cy + s * math.sin(th)))
    return pts

def _rot(p, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    x, y = p
    return (ca * x - sa * y, sa * x + ca * y)

def _enclosing_hex_side(points, ang_deg):
    # Minimal side of a regular hexagon (centered at origin, rotated ang_deg)
    # that contains all points.
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Hexagon with side s, rotation a: |x' cos(a'+pi/3) + y' sin(a'+pi/3)| <= s*sqrt(3)/2
    # for each of 6 directions a' = a + k*pi/3
    max_proj = 0.0
    for p in points:
        x, y = p
        for k in range(6):
            th = a + k * math.pi / 3.0 + math.pi / 3.0
            proj = abs(x * math.cos(th) + y * math.sin(th))
            if proj > max_proj:
                max_proj = proj
    return max_proj / (math.sqrt(3) / 2.0)

def _enclosing_hex_center(points, ang_deg):
    # Center of the minimal enclosing hexagon (we compute via 6 half-planes intersection)
    # For simplicity, use the centroid of the 6 support lines' intersection.
    a = math.radians(ang_deg)
    # Compute for each of 6 directions the max projection
    max_projs = []
    for k in range(6):
        th = a + k * math.pi / 3.0 + math.pi / 3.0
        proj = max(x * math.cos(th) + y * math.sin(th) for x, y in points)
        min_proj = min(x * math.cos(th) + y * math.sin(th) for x, y in points)
        max_projs.append((th, proj, min_proj))
    # The enclosing hexagon center is found by intersecting 6 lines.
    # For a regular hexagon, the center is the average of the 6 support-line offsets.
    # Each support line: x cos(th) + y sin(th) = c_k where c_k is the midpoint of (proj, min_proj)
    # but only if the hexagon is centered. For minimal enclosing, we need the intersection
    # of the 6 half-planes. Use linear programming via scipy if available, else approximate.
    try:
        from scipy.optimize import linprog
        # Minimize 0 subject to: for each direction, x cos(th) + y sin(th) <= c_k^+
        # and -x cos(th) - y sin(th) <= -c_k^-
        # where c_k^+ = max_proj, c_k^- = min_proj (relative to center)
        # Actually, we want to find center (cx, cy) such that all points are inside.
        # This is: for each direction th, cx*cos(th)+cy*sin(th) + max_proj_rel <= s*sqrt(3)/2
        # and cx*cos(th)+cy*sin(th) - min_proj_rel >= -s*sqrt(3)/2
        # This is complex; for now, use centroid as approximation.
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        return (cx, cy)
    except Exception:
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        return (cx, cy)

# Generate lattice patterns: positions on a triangular lattice
def _lattice_positions():
    """Generate candidate sets of 12 positions on a triangular lattice.
    The triangular lattice has basis vectors of length 2*sqrt(3) (for side-1 hexagons).
    We generate patterns by taking 12 points from a small window of the lattice."""
    a = 2 * math.sqrt(3)  # lattice spacing
    # Basis vectors
    e1 = (a, 0)
    e2 = (a / 2, a * math.sqrt(3) / 2)
    # Generate a grid of lattice points in a window
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * e1[0] + j * e2[0]
            y = i * e1[1] + j * e2[1]
            pts.append((x, y))
    return pts

def _select_12_from_lattice(lattice_pts, target_n=12):
    """Select 12 points from the lattice that form a compact cluster.
    We try several heuristic selections and return the best few candidates."""
    candidates = []
    # Strategy 1: 4x3 rectangle
    # Strategy 2: hexagonal cluster
    # Strategy 3: 3x4 rectangle
    # We'll generate all subsets of 12 from a small set and keep compact ones.
    # For efficiency, use a few hand-crafted patterns.
    
    # Pattern A: 4 columns x 3 rows (triangular lattice)
    # In triangular lattice, a "rectangle" of m x n points:
    # positions: (i*e1 + j*e2) for i in range(m), j in range(n)
    # But this gives m*n points. For 12: 4x3 or 3x4 or 2x6 or 6x2 or 1x12 or 12x1
    
    # 4x3: i in 0..3, j in 0..2
    pat_a = []
    for i in range(4):
        for j in range(3):
            x = i * lattice_pts[0][0] + j * (lattice_pts[1][0] - lattice_pts[0][0])  # wrong
            # Let me use the basis directly
    # Redo with explicit basis
    a = 2 * math.sqrt(3)
    e1 = np.array([a, 0])
    e2 = np.array([a / 2, a * math.sqrt(3) / 2])
    
    def make_pattern(m, n):
        pts = []
        for i in range(m):
            for j in range(n):
                p = i * e1 + j * e2
                pts.append((float(p[0]), float(p[1])))
        return pts
    
    # 12 = 4*3, 3*4, 2*6, 6*2, 1*12, 12*1
    patterns = []
    for m, n in [(4, 3), (3, 4), (2, 6), (6, 2), (1, 12), (12, 1)]:
        if m * n == 12:
            patterns.append(make_pattern(m, n))
    
    # Also try hexagonal-ish: center + ring
    # A hexagonal number pattern: 1 + 6 = 7, not 12.
    # Try: 3x4 with some offset for more compact shape
    
    # For each pattern, we'll also try shifted versions to center them
    return patterns

def _center_pattern(pattern):
    """Translate pattern so its centroid is at origin."""
    cx = sum(p[0] for p in pattern) / len(pattern)
    cy = sum(p[1] for p in pattern) / len(pattern)
    return [(x - cx, y - cy) for x, y in pattern]

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin
    
    # Generate lattice patterns
    a = 2 * math.sqrt(3)
    e1 = np.array([a, 0])
    e2 = np.array([a / 2, a * math.sqrt(3) / 2])
    
    def make_pattern(m, n):
        pts = []
        for i in range(m):
            for j in range(n):
                p = i * e1 + j * e2
                pts.append((float(p[0]), float(p[1])))
        return pts
    
    # Generate all rectangular patterns of 12 points on triangular lattice
    base_patterns = []
    for m, n in [(4, 3), (3, 4), (2, 6), (6, 2), (1, 12), (12, 1)]:
        if m * n == 12:
            base_patterns.append(make_pattern(m, n))
    
    # Also try some non-rectangular compact patterns
    # Pattern: 3x4 but with a "staggered" arrangement
    # Pattern: two rows of 4, offset by half
    # For triangular lattice, the natural compact shapes are parallelograms
    
    # Center each pattern
    centered_patterns = [_center_pattern(p) for p in base_patterns]
    
    best_side = float('inf')
    best_config = None
    
    # For each pattern, do 1D search over rotation angle
    for pat in centered_patterns:
        if time.time() > deadline:
            break
        
        # Sample rotation angles
        # The function is periodic with period 60 degrees (pi/3)
        # Sample densely
        angles = np.linspace(0, 60, 241)  # 0.25 degree steps
        
        for ang in angles:
            if time.time() > deadline:
                break
            # Rotate pattern
            rot_pat = [_rot(p, ang) for p in pat]
            side = _enclosing_hex_side(rot_pat, 0.0)  # hexagon axis-aligned after rotation
            if side < best_side:
                best_side = side
                best_config = (pat, ang)
    
    if best_config is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    pat, ang = best_config
    rot_pat = [_rot(p, ang) for p in pat]
    
    # Polish: refine the angle around the best
    if time.time() < deadline - 5:
        # Local search around best angle
        best_ang = ang
        best_side_local = best_side
        for _ in range(3):
            for da in [-1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1]:
                test_ang = best_ang + da
                rot_test = [_rot(p, test_ang) for p in pat]
                test_side = _enclosing_hex_side(rot_test, 0.0)
                if test_side < best_side_local:
                    best_side_local = test_side
                    best_ang = test_ang
            best_ang = best_ang * 1.0  # no change if no improvement
            if best_ang == ang:
                break
        ang = best_ang
        best_side = best_side_local
        rot_pat = [_rot(p, ang) for p in pat]
    
    # Compute the enclosing hexagon center and side
    center = _enclosing_hex_center(rot_pat, 0.0)
    side = _enclosing_hex_side(rot_pat, 0.0)
    
    # Build inner hexagons
    inner = []
    for (x, y) in rot_pat:
        # Each inner hexagon is rotated by ang (same as global rotation)
        inner.append([x, y, ang])
    
    # Return: inner rows (x, y, angle_degrees), outer center (x,y), side length, rotation
    # The outer hexagon has rotation 0 (axis-aligned) since we already rotated the pattern
    return inner, center, side, 0.0
