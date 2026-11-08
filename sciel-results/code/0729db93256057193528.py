import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
DEADLINE = 140.0  # safety margin under 150s kill

def _hex_vertices(cx, cy, s, ang_deg):
    # Unit regular hexagon side s, center (cx,cy), rotation ang_deg
    # Flat-top orientation at 0 deg: vertices at angles 0, 60, ...
    th = np.deg2rad(ang_deg) + np.arange(6) * np.pi / 3.0
    r = s
    return np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)

def _min_dist_to_hexagon_boundary(pts, verts):
    # pts: (M,2), verts: (6,2)
    # Compute min distance from each point to the hexagon boundary (edges)
    # For speed, use vectorized point-to-segment distance
    M = pts.shape[0]
    V = verts.shape[0]
    # edges: (V, 2, 2)
    e0 = verts
    e1 = np.roll(verts, -1, axis=0)
    # (V, M, 2) vectors from e0 to pts
    w = pts[None, :, :] - e0[:, None, :]
    ab = e1[:, None, :] - e0[:, None, :]  # (V, 1, 2)
    ab2 = np.sum(ab * ab, axis=2)  # (V, 1)
    t = np.sum(w * ab, axis=2) / np.maximum(ab2, 1e-12)  # (V, M)
    t = np.clip(t, 0.0, 1.0)
    proj = e0[:, None, :] + t[:, :, None] * ab  # (V, M, 2)
    d = np.linalg.norm(pts[None, :, :] - proj, axis=2)  # (V, M)
    return np.min(d, axis=0)  # (M,)

def _required_side_for_centers(cx, cy, angles_deg):
    # Given N centers and angles, find minimal S such that all hexagons
    # are disjoint and inside outer hexagon of side S (centered at origin, flat-top).
    # Use bisection on S.
    verts_list = [_hex_vertices(cx[i], cy[i], 1.0, angles_deg[i]) for i in range(N)]
    
    def feasible(S):
        # Check inner-inner disjointness: min distance between boundaries >= 0
        # Actually, disjoint means no overlap; distance between convex sets >= 0.
        # For hexagons, we can check if any vertex of one is inside the other,
        # or if edges intersect. A simpler robust check for small N:
        # Compute min distance between each pair of hexagons. If min_dist < 0, they overlap.
        # But distance between disjoint convex sets is positive. If they touch, 0.
        # We want them disjoint (non-overlapping interiors). So min_dist >= 0 is okay for touching.
        # However, the problem says "disjoint", which usually means non-overlapping.
        # Let's check for overlap by checking if any vertex of hex i is inside hex j.
        for i in range(N):
            for j in range(i + 1, N):
                # Check vertices of i inside j
                vi = verts_list[i]
                vj = verts_list[j]
                # Point in convex polygon test
                # Cross product signs
                cross = np.cross(vj - vj[0], vi - vj[0])
                if np.any(cross < -1e-9):
                    return False
                # Check vertices of j inside i
                cross2 = np.cross(vi - vi[0], vj - vi[0])
                if np.any(cross2 < -1e-9):
                    return False
        # Check inside outer hexagon
        # Outer hexagon: center (0,0), side S, flat-top.
        # Vertices at angles 0, 60, ... radius S.
        # Inradius = S * sqrt(3)/2
        # For a point (x,y), distance to boundary in flat-top hex:
        # Max of projections onto 3 normal directions.
        # Normals at 30, 90, 150 deg? No, flat-top normals are at 90, 150, 210, 270, 330, 30.
        # Let's use the standard support function.
        # d = max( |x| * cos(30) + |y| * sin(30) , |y| ) ? 
        # Actually, for flat-top hex of side S:
        # |x| <= S * sqrt(3)/2 * cos(0) ... no.
        # Let's just compute distance from each vertex of inner hexes to outer boundary.
        # If any vertex is outside, infeasible.
        # Outer hex vertices: S * (cos(k*pi/3), sin(k*pi/3))
        # Point in outer hex: 
        # For flat-top, the constraints are:
        # |y| <= S * sqrt(3)/2
        # |x| + |y|/sqrt(3) <= S * sqrt(3)/2  (for x>0) ... 
        # Simpler: use the distance to the 6 edges.
        # Edge normals: angles 90, 150, 210, 270, 330, 30 degrees.
        # n_k = (cos(a_k), sin(a_k))
        # Distance from origin to edge = S * sqrt(3)/2 (inradius)
        # For point p, distance to edge k = S*sqrt(3)/2 - dot(p, n_k)
        # If any distance < 0, outside.
        inradius = S * np.sqrt(3) / 2.0
        normals = np.stack([
            [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90))],
            [np.cos(np.deg2rad(150)), np.sin(np.deg2rad(150))],
            [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210))],
            [np.cos(np.deg2rad(270)), np.sin(np.deg2rad(270))],
            [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330))],
            [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
        ])
        for i in range(N):
            vi = verts_list[i]
            dists = inradius - vi @ normals.T
            if np.any(dists < -1e-9):
                return False
        return True

    # Bisection
    lo = 1.0
    hi = 10.0
    # Ensure hi is feasible
    while not feasible(hi):
        hi *= 2
        if hi > 100:
            break
    for _ in range(40):
        mid = (lo + hi) / 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi

def _objective(params):
    # params: [x0, y0, a0, x1, y1, a1, ...]
    cx = params[0::3]
    cy = params[1::3]
    angles = params[2::3]
    # Symmetry reduction: fix first hexagon at origin? 
    # The outer hexagon is centered at (0,0) in the return. 
    # The inner hexagons can be anywhere. 
    # To reduce symmetry, we can fix one hexagon's position, but it might constrain the search.
    # Given the small N and budget, let's just optimize all.
    return _required_side_for_centers(cx, cy, angles)

def construct():
    t0 = time.time()
    
    # Initial guess: hexagonal grid
    # For N=11, a 3x3 grid with center and 8 around? No, 1+6+... 
    # Let's use a simple cluster.
    # Centers at distance ~2 from each other.
    init = []
    # 1 center, 6 around, 4 more?
    # Let's just use a random init within a reasonable box.
    # Or use the grid from the seed as a starting point?
    # Seed: 4.0 * k for k in -5..5. That's a line. Bad for packing.
    
    # Let's try a hexagonal lattice init.
    # Distance between centers in hex lattice of side 1 is 2 * sqrt(3)/2 = sqrt(3) ~ 1.732?
    # No, if they touch, distance is 2 * inradius = sqrt(3).
    # Let's place them at distance 1.8.
    
    def hex_lattice_coords(n):
        coords = []
        # Generate hex lattice points
        # q, r coordinates
        # Max distance from origin
        # For N=11, radius 2 in hex lattice gives 1 + 6 + 12 = 19. Too many.
        # Radius 1 gives 7.
        # So we need a non-perfect lattice or slightly compressed.
        # Let's just use a random init.
        return coords

    # Random init
    np.random.seed(42)
    # Box of size 10x10
    init_params = np.random.uniform(-5, 5, size=(N * 3,))
    # Angles in [0, 60)
    init_params[2::3] = np.random.uniform(0, 60, size=N)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 60.0))
        
    best_result = None
    best_val = float('inf')
    
    # Differential Evolution
    # Use a small population to fit in time
    popsize = 15
    maxiter = 20
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=maxiter, 
            popsize=popsize,
            tol=1e-4,
            seed=42,
            polish=True
        )
        if res.success or res.fun < best_val:
            best_val = res.fun
            best_result = res.x
    except Exception as e:
        print("DE failed:", e)
        
    # If DE didn't finish or failed, try a few local optimizations or return best found
    # Check time
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 5:
        # Return best found or fallback
        if best_result is not None:
            cx = best_result[0::3]
            cy = best_result[1::3]
            angles = best_result[2::3]
            # Construct inner list
            inner = [[cx[i], cy[i], angles[i]] for i in range(N)]
            return inner, (0.0, 0.0), best_val, 0.0
        else:
            # Fallback to seed
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    # If we have time, we can try to refine with a different seed or method
    # But for now, return the DE result
    cx = best_result[0::3]
    cy = best_result[1::3]
    angles = best_result[2::3]
    inner = [[cx[i], cy[i], angles[i]] for i in range(N)]
    
    # Final check of the side length
    # _required_side_for_centers returns the minimal S.
    # We should return that S.
    S = _required_side_for_centers(cx, cy, angles)
    
    return inner, (0.0, 0.0), S, 0.0
