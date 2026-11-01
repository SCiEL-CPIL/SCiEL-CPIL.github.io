"""
Optimized packing of 11 unit hexagons in a regular hexagon.
Uses Nelder-Mead optimization on a smooth surrogate objective derived from
the geometric width of the point set in the hexagon's normal directions.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Pre-compute unit hexagon vertices at angle 0
    # Vertices at angles 0, 60, ..., 300
    base_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])

    def get_inner_vertices(params):
        """
        params: array of shape (3*N,)
        Returns: array of shape (N*6, 2) containing all vertices
        """
        vertices = []
        for i in range(N):
            x, y, angle_deg = params[3*i], params[3*i+1], params[3*i+2]
            angle_rad = np.radians(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            # Rotation matrix
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            verts = base_verts @ R.T + np.array([x, y])
            vertices.append(verts)
        return np.vstack(vertices)

    def objective(params):
        """
        Calculate the minimum enclosing hexagon side length for the given configuration.
        """
        verts = get_inner_vertices(params)
        # Try a few outer orientations. The optimal one is likely close to 0 or 30.
        # We sweep 0 to 60 in steps of 5 degrees.
        best_side = np.inf
        
        # Optimization: We can vectorize the projection calculation
        # normals for phi: n_k = [cos(phi + k*pi/3 + pi/6), sin(phi + k*pi/3 + pi/6)]
        # Wait, standard hexagon with "flat" top has vertices at 0, 60...
        # Normals are at 30, 90...
        # Let's define orientation phi such that one normal is at phi.
        # Then normals are at phi, phi+60, ..., phi+300.
        
        for phi_deg in np.arange(0, 60, 5.0):
            phi = np.radians(phi_deg)
            normals_angles = phi + np.linspace(0, 2 * np.pi, 7)[:-1]
            cos_n = np.cos(normals_angles)
            sin_n = np.sin(normals_angles)
            
            # Project all vertices onto the 6 normals
            # verts shape (M, 2), cos_n shape (6,)
            # projections shape (M, 6)
            projections = verts[:, 0] * cos_n[None, :] + verts[:, 1] * sin_n[None, :]
            
            # Widths
            max_proj = np.max(projections, axis=0)
            min_proj = np.min(projections, axis=0)
            widths = max_proj - min_proj
            
            # Min apothem R is half the max width
            R = 0.5 * np.max(widths)
            
            # Side length S = 2 * R / sqrt(3)
            side = 2.0 * R / np.sqrt(3.0)
            
            if side < best_side:
                best_side = side
                best_phi = phi_deg
                best_max_proj = max_proj
                best_min_proj = min_proj
                best_cos_n = cos_n
                best_sin_n = sin_n

        return best_side

    def find_center(verts, cos_n, sin_n, R):
        """
        Find a valid center (cx, cy) for the outer hexagon with given normals and apothem R.
        Constraints:
        cos_n[k] * cx + sin_n[k] * cy <= max_proj[k] + R  (Wait, n . (v-c) <= R => n.v - n.c <= R => n.c >= n.v - R)
        Let's re-derive:
        n_k . (v - c) <= R  =>  n_k . v - n_k . c <= R  =>  -n_k . c <= R - n_k . v  =>  n_k . c >= n_k . v - R
        Also n_k . (v - c) >= -R => n_k . v - n_k . c >= -R => n_k . c <= n_k . v + R
        
        So for each k:
        min_v (n_k . v) - R <= n_k . c <= max_v (n_k . v) + R
        
        We want to find any (cx, cy) satisfying these 12 inequalities.
        Variables: cx, cy.
        A_ub * [cx, cy] <= b_ub
        """
        A_ub = np.zeros((12, 2))
        b_ub = np.zeros(12)
        
        # We need max_proj and min_proj. Let's recompute or pass them.
        # For simplicity, recompute in this helper or pass.
        # Let's pass max_proj and min_proj.
        
        # Actually, I'll just implement a simple LP solver call.
        # But we need max_proj/min_proj.
        # Let's modify the objective to store the best details, or just recompute here.
        # Recalculating is fast enough.
        
        projections = verts[:, 0] * cos_n[None, :] + verts[:, 1] * sin_n[None, :]
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        
        for k in range(6):
            # Lower bound: n_k . c >= min_proj[k] - R
            # -n_k . c <= R - min_proj[k]
            A_ub[2*k, 0] = -cos_n[k]
            A_ub[2*k, 1] = -sin_n[k]
            b_ub[2*k] = R - min_proj[k]
            
            # Upper bound: n_k . c <= max_proj[k] + R
            A_ub[2*k+1, 0] = cos_n[k]
            A_ub[2*k+1, 1] = sin_n[k]
            b_ub[2*k+1] = max_proj[k] + R
            
        try:
            res = linprog(c=np.zeros(2), A_ub=A_ub, b_ub=b_ub, method='highs')
            if res.success:
                return res.x[0], res.x[1]
        except Exception:
            pass
            
        # Fallback: average of midpoints
        cx = 0
        cy = 0
        for k in range(6):
            mid = (max_proj[k] + min_proj[k]) / 2.0
            # Project midpoint back? No, just accumulate.
            # This is a heuristic.
            pass
        
        # Better fallback: Center of mass of the vertices
        return np.mean(verts[:, 0]), np.mean(verts[:, 1])

    # Initial configurations
    def hex_grid_config():
        """3x3 grid + 2 extras"""
        # Standard hex grid spacing: sqrt(3) horizontal, 1.5 vertical
        pos = []
        # Row 0
        for c in range(3):
            pos.append((c * np.sqrt(3), 0))
        # Row 1
        for c in range(3):
            pos.append((c * np.sqrt(3) + np.sqrt(3)/2, 1.5))
        # Row 2
        for c in range(3):
            pos.append((c * np.sqrt(3), 3.0))
        # Add 2 more: attach to the side
        pos.append((3 * np.sqrt(3), 1.5)) # Right of middle
        pos.append((-np.sqrt(3)/2, 1.5)) # Left of middle
        
        # Center it
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        
        params = []
        for x, y in pos:
            params.extend([x - cx, y - cy, 0.0])
        return np.array(params)

    def random_config(rng):
        # Random positions in a box
        params = []
        for _ in range(N):
            x = rng.uniform(-3, 3)
            y = rng.uniform(-3, 3)
            a = rng.uniform(0, 60)
            params.extend([x, y, a])
        return np.array(params)

    best_params = None
    best_score = np.inf
    best_center = (0, 0)
    best_angle = 0
    best_side = np.inf

    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Start 1: Hex Grid
    initial_guesses = [hex_grid_config()]
    
    # Start 2-5: Random perturbations of grid
    for _ in range(4):
        p = hex_grid_config()
        noise = rng.normal(0, 0.5, size=p.shape)
        initial_guesses.append(p + noise)
        
    # Start 6-10: Pure random
    for _ in range(5):
        initial_guesses.append(random_config(rng))

    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7, 'disp': False})
            
            if res.success or res.fun < best_score:
                score = res.fun
                if score < best_score:
                    best_score = score
                    best_params = res.x
                    
                    # Find the best orientation and center for this result
                    verts = get_inner_vertices(best_params)
                    
                    # Re-evaluate to find best phi and center
                    best_phi_local = 0
                    best_side_local = np.inf
                    best_cos_n_local = None
                    best_sin_n_local = None
                    best_R_local = 0
                    
                    for phi_deg in np.arange(0, 60, 1.0): # Finer search for final
                        phi = np.radians(phi_deg)
                        normals_angles = phi + np.linspace(0, 2 * np.pi, 7)[:-1]
                        cos_n = np.cos(normals_angles)
                        sin_n = np.sin(normals_angles)
                        
                        projections = verts[:, 0] * cos_n[None, :] + verts[:, 1] * sin_n[None, :]
                        max_proj = np.max(projections, axis=0)
                        min_proj = np.min(projections, axis=0)
                        widths = max_proj - min_proj
                        R = 0.5 * np.max(widths)
                        side = 2.0 * R / np.sqrt(3.0)
                        
                        if side < best_side_local:
                            best_side_local = side
                            best_phi_local = phi_deg
                            best_cos_n_local = cos_n
                            best_sin_n_local = sin_n
                            best_R_local = R
                            best_max_proj_local = max_proj
                            best_min_proj_local = min_proj

                    # Find center
                    cx, cy = find_center(verts, best_cos_n_local, best_sin_n_local, best_R_local)
                    
                    best_center = (cx, cy)
                    best_angle = best_phi_local
                    best_side = best_side_local
                    
        except Exception as e:
            pass

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct final output
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])

    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
