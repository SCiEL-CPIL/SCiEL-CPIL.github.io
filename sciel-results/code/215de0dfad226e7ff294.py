import math
import numpy as np
import time
from scipy.optimize import linprog, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side length, rotation."""
    verts = []
    for k in range(6):
        ang = math.radians(angle_deg) + k * math.pi / 3.0
        vx = cx + side * math.cos(ang)
        vy = cy + side * math.sin(ang)
        verts.append((vx, vy))
    return np.array(verts)

def _min_enclosing_hex_lp(pts, beta):
    """
    Find the minimum side-length regular hexagon (rotation beta) enclosing all points.
    Returns (S, cx, cy) or (inf, None, None).
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))

    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []

    for nx, ny in normals:
        for px, py in pts:
            # -nx*(cx) - ny*(cy) - D <= -(px*nx + py*ny)
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            # nx*(cx) + ny*(cy) - D <= px*nx + py*ny
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy)
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _S_from_params(params, beta_fixed=None):
    """
    params: flat array of [x0,y0, x1,y1, ..., x10,y10, cx, cy] (24+2 = 26 params)
    Or if beta_fixed is None, beta is optimized via LP for each candidate.
    Actually, we need beta as a variable too. Let's restructure.
    """
    pass

def _objective_with_beta(params, beta):
    """
    params: [x0,y0, x1,y1, ..., x10,y10] (22 params)
    beta: fixed rotation angle of outer hexagon
    Returns the minimum S such that a regular hexagon of rotation beta encloses all points.
    """
    pts = np.array(params).reshape(N, 2)
    S, center = _min_enclosing_hex_lp(pts, beta)
    return S

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate initial placements
    def hex_lattice_placement():
        """Standard hexagonal packing on triangular lattice."""
        pts = []
        # 3 rows: 2, 3, 3, 3, 2 = 13? No, we need exactly 11.
        # Try: row0: 2, row1: 3, row2: 3, row3: 3 = 11
        # Using axial coords
        axial = [
            (0, 0), (1, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3), (2, 3),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def compact_diamond():
        """A more compact diamond-shaped arrangement."""
        axial = [
            (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def compact_cluster():
        """A 6-around-1 plus extras."""
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def elongated_row():
        """Longer, narrower arrangement."""
        axial = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def staggered_rows():
        """Staggered rows for better packing."""
        axial = [
            (0, 0), (1, 0),
            (0, 1), (1, 1), (2, 1),
            (1, 2), (2, 2), (3, 2),
            (1, 3), (2, 3), (3, 3),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def tight_diamond2():
        axial = [
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def wide_diamond():
        axial = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def hex_ring_plus_center():
        """Center + 6 ring + 4 more."""
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (0, 2), (-2, 2), (0, -2),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def zigzag():
        axial = [
            (0, 0), (2, 0),
            (1, 1), (3, 1),
            (0, 2), (2, 2),
            (1, 3), (3, 3),
            (0, 4), (2, 4),
            (1, 5),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def compact_triangle():
        """Triangular number pattern: 1+2+3+4+1 = 11"""
        axial = [
            (0, 0),
            (0, 1), (1, 1),
            (0, 2), (1, 2), (2, 2),
            (0, 3), (1, 3), (2, 3), (3, 3),
            (0, 4),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    def parallelogram():
        axial = [
            (0, 0), (1, 0), (2, 0),
            (1, 1), (2, 1), (3, 1),
            (2, 2), (3, 2), (4, 2),
            (0, 3), (1, 3),
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    initial_placements = [
        hex_lattice_placement(),
        compact_diamond(),
        compact_cluster(),
        elongated_row(),
        staggered_rows(),
        tight_diamond2(),
        wide_diamond(),
        hex_ring_plus_center(),
        zigzag(),
        compact_triangle(),
        parallelogram(),
    ]

    # Add perturbed versions
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = initial_placements[rng.randint(len(initial_placements))].copy()
        perturbation = rng.normal(0, 0.3, size=(N, 2))
        initial_placements.append(base + perturbation)

    best_S = float('inf')
    best_params = None
    best_beta = 0.0
    best_center = None

    def full_optimize(pts0):
        """
        Optimize positions of N hexagons and outer hexagon center/rotation.
        Parameters: [x0,y0, ..., x10,y10, cx, cy, beta] = 22 + 2 + 1 = 25
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None

        def objective(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e10
            pts = params[:2*N].reshape(N, 2)
            cx = params[2*N]
            cy = params[2*N + 1]
            beta = params[2*N + 2]
            
            # Beta is periodic with period pi/6
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            # Compute max projection
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            
            pts_centered = pts - np.array([cx, cy])
            projs = pts_centered @ normals.T  # shape (N, 6)
            
            # For each normal direction, we need max(pts_proj) - min(pts_proj) <= 2*D
            # But since the hexagon is centered at (cx,cy), we need |proj| <= D for all
            # Actually the LP formulation handles this correctly.
            # For speed in the objective, use a smooth approximation:
            # D = max over all normals and points of |projection|
            # But this is the diameter in that direction, not the apothem distance.
            
            # Correct: for a regular hexagon centered at (cx,cy) with apothem A,
            # a point (px,py) is inside iff for all 6 normals n_k:
            #   n_k · ((px,cy) - (cx,cy)) <= A
            # So D = max_k max_i n_k · (p_i - c)
            # And S = (D + APOTHEM) / APOTHEM
            
            max_proj = np.max(np.abs(projs))
            S = (max_proj + APOTHEM) / APOTHEM
            return S

        # Initial parameters
        pts0 = np.array(pts0)
        # Center the points at origin for initialization
        mean_pt = np.mean(pts0, axis=0)
        pts0_centered = pts0 - mean_pt
        
        init_params = np.concatenate([
            pts0_centered.ravel(),
            [0.0, 0.0],  # cx, cy
            [0.0]  # beta
        ])
        
        # First pass: Nelder-Mead
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
                best_beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                best_center = (res.x[2*N], res.x[2*N+1])
        except Exception:
            pass

        # Second pass: polish with tighter tolerances
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN and best_params is not None:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res2.fun < best_S:
                    best_S = res2.fun
                    best_params = res2.x.copy()
                    best_beta = ((res2.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    best_center = (res2.x[2*N], res2.x[2*N+1])
            except Exception:
                pass

        return best_S, best_params, best_beta, best_center

    for i, pts0 in enumerate(initial_placements):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Quick evaluation before full optimization
        mean_pt = np.mean(pts0, axis=0)
        pts_centered = pts0 - mean_pt
        best_local_S = float('inf')
        for beta in np.linspace(0, math.pi/6, 13):
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            projs = pts_centered @ normals.T
            max_proj = np.max(np.abs(projs))
            S = (max_proj + APOTHEM) / APOTHEM
            if S < best_local_S:
                best_local_S = S
        
        if best_local_S < best_S + 0.5:  # Only optimize promising candidates
            full_optimize(pts0)

    # Final: verify with LP for exactness
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        pts = best_params[:2*N].reshape(N, 2)
        cx, cy = best_center
        beta = best_beta
        
        # Verify with LP
        S_lp, center_lp = _min_enclosing_hex_lp(pts, beta)
        if S_lp < float('inf') and S_lp < best_S:
            best_S = S_lp
            best_center = center_lp

    # Build output
    if best_params is not None:
        pts = best_params[:2*N].reshape(N, 2)
        angle_deg = math.degrees(best_beta)
        inner = []
        for x, y in pts:
            inner.append([float(x), float(y), float(angle_deg)])
        
        cx, cy = best_center
        center = [float(cx), float(cy)]
        side = float(best_S)
        rotation = float(angle_deg)
    else:
        # Fallback
        pts = hex_lattice_placement()
        mean_pt = np.mean(pts, axis=0)
        pts = pts - mean_pt
        best_beta_fb = 0.0
        best_S_fb = float('inf')
        for beta in np.linspace(0, math.pi/6, 37):
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            projs = pts @ normals.T
            max_proj = np.max(np.abs(projs))
            S = (max_proj + APOTHEM) / APOTHEM
            if S < best_S_fb:
                best_S_fb = S
                best_beta_fb = beta
        
        angle_deg = math.degrees(best_beta_fb)
        inner = []
        for x, y in pts:
            inner.append([float(x), float(y), float(angle_deg)])
        center = [0.0, 0.0]
        side = float(best_S_fb)
        rotation = float(angle_deg)

    return inner, center, side, rotation
