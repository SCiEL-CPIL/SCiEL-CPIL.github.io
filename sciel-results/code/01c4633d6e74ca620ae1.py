import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

# Hexagon geometry: for a unit hexagon with center (cx,cy) and rotation theta,
# the 6 vertices are at distance 1 from center.
# Two unit hexagons share an edge iff their centers are exactly 2*APOTHEM = sqrt(3) apart
# AND their orientations are the same (or differ by 60 degrees for the shared edge alignment).
# Actually, for two regular hexagons of side 1 to share an edge, their centers must be
# at distance sqrt(3) and the shared edge must be perpendicular to the line connecting centers.
# If both have the same orientation, the 6 neighbor directions (from center to center)
# are at angles: 30, 90, 150, 210, 270, 330 degrees (i.e., pi/6 + k*pi/3).

NEIGHBOR_ANGLES = [math.pi/6 + k * math.pi/3 for k in range(6)]
NEIGHBOR_VECTORS = np.array([[math.cos(a), math.sin(a)] for a in NEIGHBOR_ANGLES])
DIST = math.sqrt(3)  # distance between centers of edge-sharing hexagons

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple ring
        pts = np.array([[0.0, 0.0],
                        [DIST*math.cos(math.pi/6), DIST*math.sin(math.pi/6)],
                        [DIST*math.cos(math.pi/2), DIST*math.sin(math.pi/2)],
                        [DIST*math.cos(5*math.pi/6), DIST*math.sin(5*math.pi/6)],
                        [-DIST*math.cos(math.pi/6), -DIST*math.sin(math.pi/6)],
                        [-DIST*math.cos(math.pi/2), -DIST*math.sin(math.pi/2)],
                        [-DIST*math.cos(5*math.pi/6), -DIST*math.sin(5*math.pi/6)]])
        return [[float(x), float(y), 0.0] for x,y in pts], [0.0, 0.0], 3.0, 0.0

    # Generate all spanning trees on 11 nodes (labeled 0..10)
    # For each tree, assign node 0 at origin, then BFS to place all nodes.
    # Each edge in the tree corresponds to one of 6 neighbor directions.
    # Total trees: 11^(11-2) = 11^9 ≈ 2.36 billion — too many to enumerate all.
    # Instead, use a smarter approach: enumerate "compact" configurations.

    # Strategy: Build configurations by starting from a central hexagon and
    # adding neighbors in a BFS-like fashion, tracking which hexagons are placed.
    # We limit the search to configurations that are "connected" and "compact".

    # Better strategy: Use the fact that for a minimal packing, most hexagons
    # should be touching. Generate configurations by:
    # 1. Place hexagon 0 at origin.
    # 2. For each subsequent hexagon i, it must touch at least one already-placed hexagon.
    # 3. Enumerate all valid placements (touching existing, not overlapping any).
    # This is a backtracking search.

    # To keep it tractable, we'll limit the search and use symmetry breaking.

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    # Precompute: for a set of points, compute the minimal enclosing hexagon side
    def compute_min_enclosing_hex(pts):
        """Find minimal side length of a regular hexagon containing all points."""
        n = len(pts)
        if n == 0:
            return float('inf'), 0.0, (0.0, 0.0)

        # The enclosing hexagon has 6 sides. For a given rotation beta,
        # the constraint is that for each of the 6 normals, the projection
        # of all points onto that normal (centered at the hex center) must be
        # within [-D, D] where D is the apothem of the enclosing hex.
        # The side length S = (D + APOTHEM) / APOTHEM

        # For fixed beta, we can solve via LP:
        # Minimize D
        # Subject to: for each normal k, for each point i:
        #   -D <= (p_i - c) . n_k <= D
        # where c = (cx, cy) is the center.
        # Variables: cx, cy, D

        def solve_for_beta(beta):
            normals = np.array([[math.cos(beta + math.pi/6 + k*math.pi/3),
                                  math.sin(beta + math.pi/6 + k*math.pi/3)]
                                 for k in range(6)])

            # LP: minimize D
            # Variables: [cx, cy, D]
            c_obj = np.array([0.0, 0.0, 1.0])
            A_ub = []
            b_ub = []

            for nx, ny in normals:
                for px, py in pts:
                    # (px - cx)*nx + (py - cy)*ny <= D
                    # => -nx*cx - ny*cy - D <= -px*nx - py*ny
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(px*nx + py*ny))
                    # -(px - cx)*nx - (py - cy)*ny <= D
                    # => nx*cx + ny*cy - D <= px*nx + py*ny
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(px*nx + py*ny)

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]

            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    cx, cy = res.x[0], res.x[1]
                    S = (D + APOTHEM) / APOTHEM
                    return S, (cx, cy), D
                else:
                    return float('inf'), None, None
            except:
                return float('inf'), None, None

        # Coarse search over beta in [0, pi/6)
        best_S_beta = float('inf')
        best_beta_val = 0.0
        best_center_val = None

        # 72 grid points
        for i in range(72):
            beta = i * (math.pi/6) / 72
            S, center, D = solve_for_beta(beta)
            if S < best_S_beta:
                best_S_beta = S
                best_beta_val = beta
                best_center_val = center

        # Refine with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: solve_for_beta(b)[0],
                                     bounds=(0, math.pi/6), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-10})
                if res.fun < best_S_beta:
                    best_S_beta = res.fun
                    best_beta_val = res.x
                    _, best_center_val, _ = solve_for_beta(best_beta_val)
        except:
            pass

        return best_S_beta, best_beta_val, best_center_val

    # Backtracking search for configurations
    # State: list of placed hexagon centers, set of placed indices
    # We use symmetry breaking: hexagon 0 at origin.
    # For hexagon i, it must touch at least one placed hexagon j < i.
    # It must not overlap any placed hexagon.

    # Two hexagons overlap if their centers are at distance < sqrt(3)
    # (since they're both unit hexagons with same orientation, they're disjoint
    #  iff center distance >= sqrt(3), touching iff = sqrt(3))

    # Actually, for same-orientation unit hexagons, they are disjoint iff
    # center distance >= sqrt(3). If distance < sqrt(3), they overlap.

    placed = []  # list of (x, y)
    placed_set = set()
    all_configs = []
    max_configs = 5000  # limit to avoid explosion

    def backtrack(idx):
        nonlocal best_S, best_pts, best_beta, best_center, max_configs
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return

        if idx == N:
            # Evaluate this configuration
            pts = np.array(placed)
            S, beta, center = compute_min_enclosing_hex(pts)
            if S < best_S:
                best_S = S
                best_pts = pts.copy()
                best_beta = beta
                best_center = center
            max_configs -= 1
            return

        if max_configs <= 0:
            return

        # Try placing hexagon idx at each valid position
        # It must touch at least one already-placed hexagon
        # Possible positions: for each placed hexagon j, for each of 6 directions,
        # candidate position = placed[j] + DIST * neighbor_vector[dir]
        candidates = set()
        for j, (px, py) in enumerate(placed):
            for k in range(6):
                nx = px + DIST * math.cos(NEIGHBOR_ANGLES[k])
                ny = py + DIST * math.sin(NEIGHBOR_ANGLES[k])
                # Round to avoid floating point issues
                candidates.add((round(nx, 10), round(ny, 10)))

        # Symmetry breaking: for idx=1, only try directions 0,1,2 (half the hexagon)
        # Actually, let's be less aggressive to not miss good configs
        if idx == 1:
            # Only try 3 directions (by symmetry)
            valid_dirs = [0, 1, 2]
            new_candidates = set()
            for k in valid_dirs:
                nx = placed[0][0] + DIST * math.cos(NEIGHBOR_ANGLES[k])
                ny = placed[0][1] + DIST * math.sin(NEIGHBOR_ANGLES[k])
                new_candidates.add((round(nx, 10), round(ny, 10)))
            candidates = new_candidates

        for (nx, ny) in sorted(candidates):
            # Check no overlap with any placed hexagon
            valid = True
            for (px, py) in placed:
                d2 = (nx - px)**2 + (ny - py)**2
                if d2 < DIST**2 - 1e-8:
                    valid = False
                    break
            if not valid:
                continue

            # Place it
            placed.append((nx, ny))
            backtrack(idx + 1)
            placed.pop()

    # Start with hexagon 0 at origin
    placed.append((0.0, 0.0))
    backtrack(1)

    # If backtracking didn't find anything or timed out, use a fallback
    if best_pts is None:
        # Simple ring configuration
        pts = np.array([[0.0, 0.0]] +
                       [[DIST*math.cos(NEIGHBOR_ANGLES[k]), DIST*math.sin(NEIGHBOR_ANGLES[k])] for k in range(6)] +
                       [[DIST*2*math.cos(NEIGHBOR_ANGLES[0]), DIST*2*math.sin(NEIGHBOR_ANGLES[0])],
                        [DIST*2*math.cos(NEIGHBOR_ANGLES[2]), DIST*2*math.sin(NEIGHBOR_ANGLES[2])],
                        [DIST*2*math.cos(NEIGHBOR_ANGLES[4]), DIST*2*math.sin(NEIGHBOR_ANGLES[4])]])
        best_S, best_beta, best_center = compute_min_enclosing_hex(pts)
        best_pts = pts

    # Polish the best configuration
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            normals = np.array([[math.cos(beta + math.pi/6 + k*math.pi/3),
                                  math.sin(beta + math.pi/6 + k*math.pi/3)]
                                 for k in range(6)])
            pts_c = best_pts - np.array([cx, cy])
            projs = np.abs(pts_c @ normals.T)
            D = np.max(projs)
            return (D + APOTHEM) / APOTHEM

        cx0, cy0 = best_center
        best_S2 = objective([cx0, cy0, best_beta])
        best_params = [cx0, cy0, best_beta]

        # Multi-start Nelder-Mead
        starts = [[cx0, cy0, best_beta]]
        for db in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            for dc in [-0.03, 0.0, 0.03]:
                for dr in [-0.03, 0.0, 0.03]:
                    starts.append([cx0+dc, cy0+dr, best_beta+db])

        for init in starts:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S2:
                    best_S2 = res.fun
                    best_params = res.x
            except:
                pass

        if best_S2 < best_S:
            best_S = best_S2
            best_center = (best_params[0], best_params[1])
            best_beta = ((best_params[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
