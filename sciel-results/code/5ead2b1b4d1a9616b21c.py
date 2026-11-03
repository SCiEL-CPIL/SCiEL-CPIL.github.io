"""Pack 12 unit regular hexagons inside a minimal regular hexagon."""
import math
import time
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon, side length `side`, centered at (cx,cy), rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        th = a + math.radians(60 * k)
        verts.append((cx + side * math.cos(th), cy + side * math.sin(th)))
    return np.array(verts)

def face_normals(angle_deg):
    """6 outward face normals of a regular hexagon rotated by angle_deg (degrees)."""
    a = math.radians(angle_deg)
    ns = []
    for k in range(6):
        th = a + math.radians(60 * k) + math.radians(30)  # face normal is 30 deg from vertex direction
        ns.append((math.cos(th), math.sin(th)))
    return np.array(ns)

def min_side_for_orientation(all_verts, ns):
    """Given all inner vertices (M,2) and 6 outward normals (6,2),
    find min side length s and center c such that all points are inside.
    For a fixed orientation, optimal center is midpoint of min/max support per axis.
    Returns (side_length, center)."""
    # project onto each normal
    proj = all_verts @ ns.T  # (M,6)
    # for each axis j, the hexagon half-width must be >= (max - min)/2, and center offset
    # The hexagon of side s has inradius r = s*sqrt(3)/2, so along normal j the half-extent is r.
    # We need r >= (max_j - min_j)/2 for all j, and center c satisfies (max_j+min_j)/2 - c . n_j = 0 ... 
    # Actually center must be at (max_j + min_j)/2 projected back. Let's compute:
    # For axis j: min_j <= (c . n_j) - r and max_j >= (c . n_j) + r
    # => c . n_j = (min_j + max_j)/2, and r >= (max_j - min_j)/2
    lo = proj.min(axis=0)  # (6,)
    hi = proj.max(axis=0)  # (6,)
    mid = (lo + hi) / 2.0  # (6,) = c . n_j for each j
    r_needed = (hi - lo) / 2.0  # (6,)
    r = r_needed.max()
    s = r * 2.0 / math.sqrt(3.0)
    # recover center: solve c . n_j = mid_j for all j (overdetermined but consistent if we use 2)
    # Use least squares with 2x6 matrix
    A = ns  # (6,2)
    c = np.linalg.lstsq(A, mid, rcond=None)[0]
    return s, (float(c[0]), float(c[1]))

def evaluate(inner, outer_angle_deg):
    """inner: list of (x,y,angle_deg). Returns (side, center)."""
    all_verts = []
    for (x, y, ang) in inner:
        all_verts.append(hex_vertices(x, y, ang))
    all_verts = np.vstack(all_verts)  # (72,2)
    ns = face_normals(outer_angle_deg)
    s, c = min_side_for_orientation(all_verts, ns)
    return s, c

def check_valid(inner, outer_center, side, outer_angle_deg):
    """Verify no inner hexagons overlap each other and all inside outer. Return (valid, min_gap)."""
    # Check pairwise overlap of inner hexagons via SAT
    polys = [hex_vertices(x, y, a) for (x, y, a) in inner]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys_overlap(polys[i], polys[j]):
                return False, 0.0
    # Check containment
    ns = face_normals(outer_angle_deg)
    r = side * math.sqrt(3.0) / 2.0
    cx, cy = outer_center
    for p in polys:
        for v in p:
            for n in ns:
                if (v[0] - cx) * n[0] + (v[1] - cy) * n[1] > r + 1e-9:
                    return False, 0.0
    return True, 1.0

def polys_overlap(A, B):
    """SAT overlap test for two convex polygons (numpy arrays of (x,y))."""
    polys = [A, B]
    for P in polys:
        for i in range(len(P)):
            p1 = P[i]
            p2 = P[(i + 1) % len(P)]
            # edge normal
            ex = p2[0] - p1[0]
            ey = p2[1] - p1[1]
            # normal (perpendicular)
            nx = -ey
            ny = ex
            norm = math.hypot(nx, ny)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            # project both polys
            projA = A @ np.array([nx, ny])
            projB = B @ np.array([nx, ny])
            if projA.max() < projB.min() or projB.max() < projA.min():
                return False  # separating axis found
    return True

def initial_layouts():
    """Generate several candidate initial layouts for the 12 hexagons."""
    layouts = []
    # 1) Hexagonal lattice: center + ring of 6 + partial ring of 5
    # Hex lattice spacing for non-overlap: distance >= 2 * inradius * sqrt(3)/... 
    # Two unit hexagons side by side: min center distance for no overlap is 2 * inradius = sqrt(3) ~ 1.732? 
    # Actually two regular hexagons of side 1: the minimal center distance for non-overlap depends on orientation.
    # For flat-top aligned, side-by-side: distance = 2 * inradius = sqrt(3) ~ 1.732.
    # Let's use spacing = 1.75 for safety.
    sp = 1.75
    # hex lattice positions (axial): 
    # center (0,0)
    # ring 1: 6 neighbors at distance sp in 60-deg directions (for pointy orientation) 
    # For hex lattice with flat-top hexagons, neighbors are at 0,60,120,180,240,300 at distance = 2*inradius = sqrt(3)
    pos = [(0.0, 0.0)]
    for k in range(6):
        th = math.radians(60 * k)
        pos.append((sp * math.cos(th), sp * math.sin(th)))
    # ring 2 partial: 5 more
    for k in range(6):
        th = math.radians(60 * k + 30)
        pos.append((2 * sp * math.cos(th), 2 * sp * math.sin(th)))
    pos = pos[:12]
    layouts.append([(x, y, 0.0) for (x, y) in pos])
    
    # 2) Rectangular 3x4 grid
    sp = 1.8
    pos = []
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * sp
            y = (r - 1.0) * sp
            pos.append((x, y))
    layouts.append([(x, y, 0.0) for (x, y) in pos])
    
    # 3) Two rows of 6
    sp = 1.8
    pos = []
    for r in range(2):
        for c in range(6):
            x = (c - 2.5) * sp
            y = (r - 0.5) * sp
            pos.append((x, y))
    layouts.append([(x, y, 0.0) for (x, y) in pos])
    
    # 4) Ring of 12 (no center)
    R = 2.0
    pos = []
    for k in range(12):
        th = math.radians(30 * k)
        pos.append((R * math.cos(th), R * math.sin(th)))
    layouts.append([(x, y, 0.0) for (x, y) in pos])
    
    # 5) Hexagonal lattice but pointy-top orientation for inner
    sp = 1.75
    pos = [(0.0, 0.0)]
    for k in range(6):
        th = math.radians(60 * k + 30)
        pos.append((sp * math.cos(th), sp * math.sin(th)))
    for k in range(6):
        th = math.radians(60 * k)
        pos.append((2 * sp * math.cos(th), 2 * sp * math.sin(th)))
    pos = pos[:12]
    layouts.append([(x, y, 30.0) for (x, y) in pos])
    
    return layouts

def anneal(inner, outer_angle_deg, time_budget):
    """Simulated annealing to minimize outer side length."""
    start = time.time()
    pos = np.array([[x, y] for (x, y, a) in inner], dtype=float)
    angles = np.array([a for (x, y, a) in inner], dtype=float)
    
    def objective():
        inner_list = [(float(pos[i,0]), float(pos[i,1]), float(angles[i])) for i in range(N)]
        s, _ = evaluate(inner_list, outer_angle_deg)
        # add penalty for overlaps
        polys = [hex_vertices(float(pos[i,0]), float(pos[i,1]), float(angles[i])) for i in range(N)]
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polys_overlap(polys[i], polys[j]):
                    overlap_pen += 1.0
        return s + 10.0 * overlap_pen
    
    best = objective()
    best_pos = pos.copy()
    best_ang = angles.copy()
    
    T = 1.0
    cool = 0.9995
    step = 0.5
    
    while time.time() - start < time_budget and T > 0.01:
        # propose move
        i = int(np.random.randint(N))
        kind = int(np.random.randint(3))
        new_pos = pos.copy()
        new_ang = angles.copy()
        if kind == 0:
            new_pos[i, 0] += np.random.randn() * step
            new_pos[i, 1] += np.random.randn() * step
        elif kind == 1:
            new_ang[i] += np.random.randn() * step * 0.5
        else:
            # move two
            j = int(np.random.randint(N))
            new_pos[i, 0] += np.random.randn() * step * 0.5
            new_pos[i, 1] += np.random.randn() * step * 0.5
            new_pos[j, 0] += np.random.randn() * step * 0.5
            new_pos[j, 1] += np.random.randn() * step * 0.5
        
        # evaluate
        new_inner = [(float(new_pos[k,0]), float(new_pos[k,1]), float(new_ang[k])) for k in range(N)]
        new_obj = objective_with(new_inner, outer_angle_deg)
        
        if new_obj < best or np.random.rand() < math.exp(-(new_obj - best) / T):
            pos = new_pos
            angles = new_ang
            if new_obj < best:
                best = new_obj
                best_pos = pos.copy()
                best_ang = angles.copy()
        T *= cool
        step *= 0.999
    
    inner_list = [(float(best_pos[i,0]), float(best_pos[i,1]), float(best_ang[i])) for i in range(N)]
    s, c = evaluate(inner_list, outer_angle_deg)
    return s, c, inner_list

def objective_with(inner_list, outer_angle_deg):
    s, _ = evaluate(inner_list, outer_angle_deg)
    polys = [hex_vertices(x, y, a) for (x, y, a) in inner_list]
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if polys_overlap(polys[i], polys[j]):
                pen += 1.0
    return s + 10.0 * pen

def construct():
    start = time.time()
    best_s = float('inf')
    best_result = None
    
    layouts = initial_layouts()
    
    # Time per layout
    n_layouts = len(layouts)
    # Also try a few outer angles
    outer_angles = [0.0, 30.0, 15.0]
    
    budget_per = (DEADLINE - (time.time() - start) - 5.0) / max(1, n_layouts * len(outer_angles))
    budget_per = max(1.0, min(budget_per, 25.0))
    
    for layout in layouts:
        for oa in outer_angles:
            if time.time() - start > DEADLINE - 10.0:
                break
            s, c, inner_list = anneal(layout, oa, budget_per)
            # validate
            valid, _ = check_valid(inner_list, c, s, oa)
            if valid and s < best_s:
                best_s = s
                best_result = (inner_list, c, s, oa)
        if time.time() - start > DEADLINE - 10.0:
            break
    
    # If nothing valid, fall back to a known-valid large packing
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result

# Provide a default so the module is importable
if __name__ == "__main__":
    r = construct()
    print(r[2])
