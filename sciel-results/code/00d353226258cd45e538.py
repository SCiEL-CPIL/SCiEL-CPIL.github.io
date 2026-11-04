import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Leave 2 seconds for safety margin
    
    def hexagon_vertices(cx, cy, side, angle_rad):
        """Get vertices of a regular hexagon."""
        verts = np.empty((6, 2))
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            verts[i] = [cx + side * math.cos(theta), cy + side * math.sin(theta)]
        return verts
    
    def hexagon_edge_normals(cx, cy, side, angle_rad):
        """Get edge normals of a regular hexagon (outward facing)."""
        norms = np.empty((6, 2))
        for i in range(6):
            # Edge from vertex i to vertex i+1
            theta1 = angle_rad + i * math.pi / 3
            theta2 = angle_rad + (i + 1) * math.pi / 3
            v1 = np.array([cx + side * math.cos(theta1), cy + side * math.sin(theta1)])
            v2 = np.array([cx + side * math.cos(theta2), cy + side * math.sin(theta2)])
            edge = v2 - v1
            # Outward normal: perpendicular to edge, pointing away from center
            norm = np.array([-edge[1], edge[0]])
            norm = norm / np.linalg.norm(norm)
            # Ensure it points outward
            center = np.array([cx, cy])
            mid_edge = (v1 + v2) / 2
            if np.dot(norm, mid_edge - center) < 0:
                norm = -norm
            norms[i] = norm
        return norms
    
    def compute_outer_side(centers, angles_deg, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        otheta_rad = math.radians(otheta_deg)
        
        # The 6 outward normals of the outer hexagon
        outer_normals = []
        for k in range(6):
            theta = otheta_rad + k * math.pi / 3
            outer_normals.append(np.array([math.cos(theta), math.sin(theta)]))
        
        max_radius = 0.0
        
        # For each outer normal, find the maximum projection
        for on in outer_normals:
            max_proj = -float('inf')
            for i in range(N):
                cx, cy = centers[i]
                angle_rad = math.radians(angles_deg[i])
                # Vertices of inner hexagon i
                for v in range(6):
                    theta = angle_rad + v * math.pi / 3
                    vx = cx + math.cos(theta)
                    vy = cy + math.sin(theta)
                    proj = (vx - ocx) * on[0] + (vy - ocy) * on[1]
                    if proj > max_proj:
                        max_proj = proj
            
            if max_proj > max_radius:
                max_radius = max_proj
        
        # Side length of outer hexagon: for a regular hexagon, apothem = side * sqrt(3)/2
        # The projection from center to any side (apothem) must be >= max_radius
        # So apothem = max_radius, and side = 2 * apothem / sqrt(3)
        return 2.0 * max_radius / math.sqrt(3)
    
    def check_disjoint(centers, angles_deg, eps=1e-8):
        """Check if all hexagons are disjoint using SAT."""
        all_verts = []
        all_normals = []
        
        for i in range(N):
            cx, cy = centers[i]
            angle_rad = math.radians(angles_deg[i])
            verts = hexagon_vertices(cx, cy, 1.0, angle_rad)
            all_verts.append(verts)
            
            # Compute edge normals for SAT
            norms = np.empty((6, 2))
            for e in range(6):
                v1 = verts[e]
                v2 = verts[(e + 1) % 6]
                edge = v2 - v1
                norm = np.array([-edge[1], edge[0]])
                norm = norm / np.linalg.norm(norm)
                norms[e] = norm
            all_normals.append(norms)
        
        for i in range(N):
            for j in range(i + 1, N):
                v1, v2 = all_verts[i], all_verts[j]
                n1, n2 = all_normals[i], all_normals[j]
                
                # Check all axes
                disjoint = False
                for axis in list(n1) + list(n2):
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    if max1 < min2 + eps or max2 < min1 + eps:
                        disjoint = True
                        break
                
                if not disjoint:
                    return False
        
        return True
    
    def optimize_all(x0):
        """
        Full optimization using differential evolution-like approach + local polish.
        Variables: x1, y1, a1, ..., x12, y12, a12, ocx, ocy, otheta
        Total: 39 variables
        """
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            
            centers = [[x[i*3], x[i*3+1]] for i in range(N)]
            angles = [x[i*3+2] for i in range(N)]
            ocx, ocy, otheta = x[36], x[37], x[38]
            
            # Quick overlap check using distance heuristic
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance for non-overlapping unit hexagons is sqrt(3)
                    if dist < math.sqrt(3) - 0.01:
                        return 1e6 + dist  # Penalty for overlap
            
            return compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        bounds = []
        for i in range(N):
            bounds.extend([(-4, 4), (-4, 4), (0, 60)])
        bounds.extend([(-2, 2), (-2, 2), (0, 60)])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'adaptive': True})
            if res.fun < 1e6:
                return res.x
        except Exception:
            pass
        return x0
    
    def polish_solution(centers, angles, ocx, ocy, otheta):
        """High-precision polish of the solution."""
        if time.time() > deadline - 8:
            return centers, angles, ocx, ocy, otheta, compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # First: refine outer hexagon
        def obj_outer(x):
            return compute_outer_side(centers, angles, x[0], x[1], x[2])
        
        try:
            res = minimize(obj_outer, [ocx, ocy, otheta], method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500})
            if res.success:
                ocx, ocy, otheta = res.x
        except Exception:
            pass
        
        # Second: refine inner positions (fixing outer)
        def obj_inner(x):
            c = [[x[i*3], x[i*3+1]] for i in range(N)]
            a = [x[i*3+2] for i in range(N)]
            
            # Quick overlap check
            for i in range(N):
                for j in range(i+1, N):
                    dx = c[i][0] - c[j][0]
                    dy = c[i][1] - c[j][1]
                    if math.sqrt(dx*dx + dy*dy) < math.sqrt(3) - 0.005:
                        return 1e6
            return compute_outer_side(c, a, ocx, ocy, otheta)
        
        x_inner = []
        for i in range(N):
            x_inner.extend([centers[i][0], centers[i][1], angles[i]])
        
        bounds_inner = []
        for i in range(N):
            bounds_inner.extend([(-4, 4), (-4, 4), (0, 60)])
        
        try:
            res = minimize(obj_inner, x_inner, method='Nelder-Mead', bounds=bounds_inner,
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 100, 'adaptive': True})
            if res.fun < 1e6:
                centers = [[res.x[i*3], res.x[i*3+1]] for i in range(N)]
                angles = [res.x[i*3+2] for i in range(N)]
        except Exception:
            pass
        
        # Third: re-refine outer
        try:
            res = minimize(obj_outer, [ocx, ocy, otheta], method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 300})
            if res.success:
                ocx, ocy, otheta = res.x
        except Exception:
            pass
        
        # Grid search polish for outer
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        for scale in [0.001, 0.0001, 0.00001]:
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da in [-scale*10, 0, scale*10]:
                            s = compute_outer_side(centers, angles, ocx+dx, ocy+dy, otheta+da)
                            if s < best_side - 1e-12:
                                best_side = s
                                ocx, ocy, otheta = ocx+dx, ocy+dy, otheta+da
                                improved = True
        
        return centers, angles, ocx, ocy, otheta, best_side
    
    # Generate initial configurations
    random.seed(12345)
    
    def make_lattice_config(spacing_factor=1.0):
        """Create hexagonal lattice configuration."""
        d = math.sqrt(3) * spacing_factor + 0.01
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1 = [u, v, v-u, -u, -v, u-v]
        for vec in ring1:
            centers.append(vec.copy())
        
        # Add 5 more from ring 2
        ring2 = []
        basis = ring1
        for p1 in ring1:
            for b in basis:
                p2 = p1 + b
                if not np.allclose(p2, np.zeros(2)):
                    dup = False
                    for pc in ring2:
                        if np.allclose(p2, pc):
                            dup = True
                            break
                    if not dup:
                        ring2.append(p2)
        
        ring2.sort(key=lambda p: np.linalg.norm(p))
        for i in range(min(5, len(ring2))):
            centers.append(ring2[i].copy())
        
        return centers[:12]
    
    def make_ring_config():
        """Create ring configuration."""
        centers = [np.array([0.0, 0.0])]
        d = math.sqrt(3) + 0.01
        for i in range(6):
            theta = i * math.pi / 3
            centers.append(np.array([d * math.cos(theta), d * math.sin(theta)]))
        
        # Add 5 more at larger radius
        d2 = 2 * math.sqrt(3) + 0.01
        for i in range(5):
            theta = i * math.pi / 3 + math.pi / 6
            centers.append(np.array([d2 * math.cos(theta), d2 * math.sin(theta)]))
        
        return centers
    
    def make_random_config():
        """Create random configuration."""
        centers = []
        for _ in range(12):
            centers.append(np.array([random.uniform(-3, 3), random.uniform(-3, 3)]))
        return centers
    
    configs = []
    configs.append(make_lattice_config(1.0))
    configs.append(make_lattice_config(1.02))
    configs.append(make_ring_config())
    for _ in range(4):
        configs.append(make_random_config())
    
    best_result = None
    best_val = float('inf')
    
    for cfg in configs:
        if time.time() > deadline - 30:
            break
        
        centers = [[c[0], c[1]] for c in cfg]
        angles = [30.0] * N
        
        # Randomize angles slightly
        if random.random() < 0.5:
            angles = [random.uniform(0, 60) for _ in range(N)]
        
        x0 = []
        for i in range(N):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        x0.extend([0.0, 0.0, 0.0])
        
        # Optimize
        x_opt = optimize_all(x0)
        
        centers_opt = [[x_opt[i*3], x_opt[i*3+1]] for i in range(N)]
        angles_opt = [x_opt[i*3+2] for i in range(N)]
        ocx, ocy, otheta = x_opt[36], x_opt[37], x_opt[38]
        
        # Polish
        centers_p, angles_p, ocx_p, ocy_p, otheta_p, side_p = polish_solution(
            centers_opt, angles_opt, ocx, ocy, otheta
        )
        
        # Validate
        if check_disjoint(centers_p, angles_p):
            if side_p < best_val:
                best_val = side_p
                best_result = (centers_p, angles_p, ocx_p, ocy_p, otheta_p, side_p)
    
    if best_result is None:
        # Fallback
        centers = [[c[0], c[1]] for c in make_lattice_config()]
        angles = [30.0] * N
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        centers, angles, ocx, ocy, otheta, side = polish_solution(centers, angles, ocx, ocy, otheta)
        best_result = (centers, angles, ocx, ocy, otheta, side)
    
    centers, angles, ocx, ocy, otheta, side = best_result
    
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
