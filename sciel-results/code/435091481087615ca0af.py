import math
import numpy as np
import time
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
CENTER_DIST = math.sqrt(3)  # distance between centers of edge-sharing unit hexagons
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Edge directions for hexagons (6 directions)
    # If hex A has hex B sharing its edge in direction k, B is at A + CENTER_DIST * (cos(angle_k), sin(angle_k))
    # For a regular hexagon with flat top, the 6 edge normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # But the center-to-center direction when sharing an edge is perpendicular to the edge
    # Let's define: for a hexagon, the 6 neighbors are at directions 0°, 60°, 120°, 180°, 240°, 300°
    # Actually, for pointy-top hexagons, neighbors are at 30°, 90°, 150°, 210°, 270°, 330°
    # Let's use: neighbor directions at angles (k * 60 + 30) degrees for k=0..5
    # This gives 30, 90, 150, 210, 270, 330 degrees
    
    edge_angles = [math.radians(30 + 60 * k) for k in range(6)]
    edge_vectors = np.array([[math.cos(a), math.sin(a)] for a in edge_angles])

    def compute_positions(adjacency):
        """
        Given adjacency as a list of (i, j, dir_k) meaning hex j is neighbor of hex i in direction k,
        compute positions. Hex 0 is at origin.
        Returns None if inconsistent, else array of positions.
        """
        n = N
        pos = [None] * n
        pos[0] = np.array([0.0, 0.0])
        
        # BFS/DFS to propagate positions
        visited = [False] * n
        visited[0] = True
        queue = [0]
        
        while queue:
            u = queue.pop(0)
            for (i, j, k) in adjacency:
                if i == u and not visited[j]:
                    pos[j] = pos[i] + CENTER_DIST * edge_vectors[k]
                    visited[j] = True
                    queue.append(j)
                elif j == u and not visited[i]:
                    pos[i] = pos[j] - CENTER_DIST * edge_vectors[k]
                    visited[i] = True
                    queue.append(i)
        
        if not all(visited):
            return None
        
        # Verify consistency for all edges
        for (i, j, k) in adjacency:
            if pos[i] is not None and pos[j] is not None:
                expected = pos[i] + CENTER_DIST * edge_vectors[k]
                if np.linalg.norm(pos[j] - expected) > 1e-9:
                    return None
        
        return np.array(pos)

    def min_enclosing_hexagon(pts):
        """
        Find minimum side length S of a regular hexagon containing all points.
        The hexagon can be at any center and rotation.
        For a given rotation beta, the problem is convex (LP).
        We optimize over beta in [0, pi/6) due to 6-fold symmetry.
        """
        n_pts = len(pts)
        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        # For a hexagon with rotation beta, the 6 edge normals are at beta + 30° + k*60°
        # A point p is inside if for all k: |dot(p - c, n_k)| <= D where D = S * APOTHEM
        # So S = (D + APOTHEM) / APOTHEM where D = max over all points and all k of |dot(p - c, n_k)|
        
        # For fixed beta, minimize over center c and D:
        # min D subject to: -D <= dot(p_i - c, n_k) <= D for all i, k
        # This is: min D subject to: dot(c, n_k) - D <= dot(p_i, n_k) and dot(c, n_k) + D >= dot(p_i, n_k)
        # i.e., dot(c, n_k) - dot(p_i, n_k) <= D and dot(p_i, n_k) - dot(c, n_k) <= D
        
        def calc_S_for_beta(beta):
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            n_constraints = 12 * n_pts
            A_ub = np.zeros((n_constraints, 3))
            b_ub = np.zeros(n_constraints)
            
            for k in range(6):
                nx, ny = normals[k]
                projs = pts[:, 0] * nx + pts[:, 1] * ny
                start = 2 * k * n_pts
                A_ub[start:start+n_pts, 0] = -nx
                A_ub[start:start+n_pts, 1] = -ny
                A_ub[start:start+n_pts, 2] = -1.0
                b_ub[start:start+n_pts] = -projs
                A_ub[start+n_pts:start+2*n_pts, 0] = nx
                A_ub[start+n_pts:start+2*n_pts, 1] = ny
                A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
                b_ub[start+n_pts:start+2*n_pts] = projs
            
            c_obj = np.array([0.0, 0.0, 1.0])
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                from scipy.optimize import linprog
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    D = res.x[2]
                    cx = res.x[0]
                    cy = res.x[1]
                    S = (D + APOTHEM) / APOTHEM
                    return S, (cx, cy), D
                else:
                    return float('inf'), None, None
            except Exception:
                return float('inf'), None, None

        # Grid search over beta
        for i in range(181):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / 180
            S, center, D = calc_S_for_beta(beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        # Refine with scalar optimization
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(lambda b: calc_S_for_beta(b)[0],
                                 bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 500, 'xatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    # Generate contact graphs
    # Strategy: start with a "core" of hexagons in a compact arrangement, then add others
    # We'll enumerate trees (spanning trees) of 11 nodes and assign edge directions
    
    # For efficiency, let's use a recursive approach to build clusters
    # Start with hex 0, add hexagons one by one, each attached to an existing hex
    
    def generate_clusters():
        """Generate diverse cluster configurations by building trees."""
        clusters = []
        
        # Predefined compact arrangements as adjacency lists
        # Each entry is a list of (parent, child, direction)
        
        # Hexagonal close packing patterns
        # Pattern 1: 7 hexagons in a hexagonal cluster + 4 more
        # Center at 0, ring of 6 at directions 0-5
        p1 = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5),
              (1, 7, 0), (2, 8, 1), (3, 9, 2), (4, 10, 3)]
        clusters.append(p1)
        
        # Pattern 2: 7 hex cluster with different outer attachments
        p2 = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5),
              (1, 7, 1), (2, 8, 2), (3, 9, 3), (4, 10, 4)]
        clusters.append(p2)
        
        # Pattern 3: 3x3 block + 2
        # Row 0: 0,1,2; Row 1: 3,4,5; Row 2: 6,7,8; plus 9,10
        # In hex coords: (0,0),(1,0),(2,0),(-1,1),(0,1),(1,1),(-2,2),(-1,2),(0,2),(2,0),(3,0)
        p3 = [(0, 1, 0), (0, 2, 1), (1, 3, 2), (1, 4, 0), (1, 5, 1),
              (2, 6, 2), (2, 7, 0), (2, 8, 1), (0, 9, 3), (0, 10, 4)]
        clusters.append(p3)
        
        # Pattern 4: Linear chain with branches
        p4 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0),
              (1, 6, 1), (2, 7, 2), (3, 8, 3), (4, 9, 4), (5, 10, 5)]
        clusters.append(p4)
        
        # Pattern 5: Star with center
        p5 = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5),
              (1, 7, 0), (2, 8, 1), (3, 9, 2), (4, 10, 3)]
        clusters.append(p5)
        
        # Pattern 6: Two rows of 4 + 3
        p6 = [(0, 1, 0), (1, 2, 0), (2, 3, 0),
              (1, 4, 1), (2, 5, 1), (3, 6, 1),
              (4, 7, 0), (5, 8, 0), (6, 9, 0),
              (7, 10, 1)]
        clusters.append(p6)
        
        # Pattern 7: Compact 4-wide
        p7 = [(0, 1, 0), (1, 2, 0), (2, 3, 0),
              (0, 4, 1), (1, 5, 1), (2, 6, 1), (3, 7, 1),
              (4, 8, 0), (5, 9, 0), (6, 10, 0)]
        clusters.append(p7)
        
        # Pattern 8: Diagonal arrangement
        p8 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
              (0, 5, 1), (1, 6, 1), (2, 7, 1), (3, 8, 1),
              (5, 9, 0), (6, 10, 0)]
        clusters.append(p8)
        
        # Pattern 9: T-shape
        p9 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
              (2, 5, 1), (2, 6, 2),
              (5, 7, 0), (6, 8, 3),
              (7, 9, 1), (8, 10, 4)]
        clusters.append(p9)
        
        # Pattern 10: L-shape
        p10 = [(0, 1, 0), (1, 2, 0), (2, 3, 0),
               (0, 4, 1), (4, 5, 1), (5, 6, 1), (6, 7, 1),
               (1, 8, 2), (2, 9, 2), (3, 10, 2)]
        clusters.append(p10)
        
        # Pattern 11: Dense cluster
        p11 = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5),
               (1, 7, 1), (2, 8, 2), (3, 9, 3), (4, 10, 4)]
        clusters.append(p11)
        
        # Pattern 12: Elongated with center
        p12 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0),
               (2, 6, 1), (3, 7, 2),
               (6, 8, 0), (7, 9, 3),
               (8, 10, 1)]
        clusters.append(p12)
        
        # Pattern 13: Zigzag
        p13 = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 1), (4, 5, 0),
               (1, 6, 2), (2, 7, 3), (3, 8, 4),
               (6, 9, 0), (7, 10, 1)]
        clusters.append(p13)
        
        # Pattern 14: 5x2 block
        p14 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
               (0, 5, 1), (1, 6, 1), (2, 7, 1), (3, 8, 1), (4, 9, 1),
               (5, 10, 0)]
        clusters.append(p14)
        
        # Pattern 15: Cross shape
        p15 = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5),
               (1, 7, 0), (3, 8, 2), (5, 9, 4),
               (7, 10, 0)]
        clusters.append(p15)
        
        # Pattern 16: Asymmetric
        p16 = [(0, 1, 0), (1, 2, 0), (2, 3, 0),
               (0, 4, 1), (1, 5, 1), (2, 6, 1),
               (4, 7, 2), (5, 8, 2),
               (7, 9, 0), (8, 10, 0)]
        clusters.append(p16)
        
        # Pattern 17: 3-4-4 arrangement
        p17 = [(0, 1, 0), (1, 2, 0),
               (0, 3, 1), (1, 4, 1), (2, 5, 1), (3, 6, 1),
               (3, 7, 0), (4, 8, 0), (5, 9, 0), (6, 10, 0)]
        clusters.append(p17)
        
        # Pattern 18: Wide base
        p18 = [(0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
               (1, 5, 1), (2, 6, 1), (3, 7, 1),
               (5, 8, 0), (6, 9, 0), (7, 10, 0)]
        clusters.append(p18)
        
        # Pattern 19: Tall narrow
        p19 = [(0, 1, 0), (1, 2, 0),
               (1, 3, 1), (2, 4, 1),
               (3, 5, 0), (4, 6, 0),
               (5, 7, 1), (6, 8, 1),
               (7, 9, 0), (8, 10, 0)]
        clusters.append(p19)
        
        # Pattern 20: Mixed
        p20 = [(0, 1, 0), (0, 2, 1), (0, 3, 2),
               (1, 4, 0), (2, 5, 1), (3, 6, 2),
               (4, 7, 1), (5, 8, 2), (6, 9, 3),
               (7, 10, 0)]
        clusters.append(p20)
        
        return clusters

    def quick_estimate(pts):
        """Quick estimate of enclosing hexagon size at beta=0."""
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Generate and evaluate clusters
    clusters = generate_clusters()
    
    # Also generate some random tree-based clusters
    import random
    random.seed(42)
    
    for _ in range(50):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Build a random tree
        adj = []
        for i in range(1, N):
            parent = random.randint(0, i-1)
            direction = random.randint(0, 5)
            adj.append((parent, i, direction))
        clusters.append(adj)
    
    # Sort by quick estimate
    cluster_data = []
    for adj in clusters:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = compute_positions(adj)
        if pts is not None:
            est = quick_estimate(pts)
            cluster_data.append((est, pts))
    
    cluster_data.sort(key=lambda x: x[0])
    
    # Process clusters
    for est, pts in cluster_data:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, beta, center = min_enclosing_hexagon(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        # Use a simple 7-hex cluster + 4 more
        adj = [(0, 1, 0), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4), (0, 6, 5),
               (1, 7, 0), (2, 8, 1), (3, 9, 2), (4, 10, 3)]
        best_global_pts = compute_positions(adj)
        if best_global_pts is None:
            best_global_pts = np.zeros((N, 2))
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = quick_estimate(best_global_pts)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scale slightly to ensure no overlap (tiny expansion)
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))

    pts_scaled = np.array(pts_scaled)

    # Recompute S with scaled points
    angles = best_global_beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    pts_c = pts_scaled - np.array([cx, cy])
    projs = pts_c @ normals.T
    D_scaled = np.max(np.abs(projs))
    S_scaled = (D_scaled + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)

    return inner, center, side, rotation
