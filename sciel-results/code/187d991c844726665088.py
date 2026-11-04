"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def get_axes(verts):
        """Get the 6 normal axes for a hexagon."""
        axes = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            axes.append(normal)
        return np.array(axes)

    def compute_overlap_depth(verts1, verts2):
        """
        Compute the penetration depth using SAT.
        Returns positive value if overlapping, 0 if disjoint.
        """
        axes1 = get_axes(verts1)
        axes2 = get_axes(verts2)
        all_axes = np.vstack([axes1, axes2])
        
        min_sep = np.inf
        for axis in all_axes:
            proj1 = verts1 @ axis
            proj2 = verts2 @ axis
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Separation along this axis
            s = min(min2 - max1, min1 - max2)
            if s > 0:
                # Separated on this axis
                if s < min_sep:
                    min_sep = s
            else:
                # Overlapping on this axis
                # The depth of overlap on this axis is -s
                # We track the minimum separation (which is negative if overlapping)
                if s < min_sep:
                    min_sep = s
        
        if min_sep > 0:
            return 0.0
        else:
            return -min_sep

    def compute_S_exact(vertices_list):
        """Compute the minimum side length S of the outer hexagon (centered at 0,0, rot 0) to contain all vertices."""
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = v[0] * np.cos(theta_n) + v[1] * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def soft_max(x, beta=10.0):
        """Smooth max approximation."""
        # Shift for numerical stability
        x_max = np.max(x)
        if x_max > 0:
            x_shifted = x - x_max
            return x_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * x_shifted)))
        else:
            return (1.0 / beta) * np.log(np.sum(np.exp(beta * x)))

    def objective(params):
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        # Compute S using soft max for differentiability
        projs = []
        for verts in vertices_list:
            for v in verts:
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = v[0] * np.cos(theta_n) + v[1] * np.sin(theta_n)
                    projs.append(abs(proj))
        
        S_val = soft_max(np.array(projs)) / (np.sqrt(3) / 2)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 4.0: # Loose bound to skip distant pairs
                    overlap = compute_overlap_depth(vertices_list[i], vertices_list[j])
                    # Smooth penalty: softplus(overlap * scale)
                    # softplus(x) = log(1 + exp(x))
                    # Use logaddexp for stability
                    penalty += np.logaddexp(0.0, 10.0 * overlap)
        
        return S_val + penalty

    # Initial guess: hexagonal lattice
    params = np.zeros(3 * N)
    params[0:3] = [0, 0, 0]
    idx = 1
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1
    for k in range(4):
        theta = k * np.pi / 3
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1

    # Use L-BFGS-B for gradient-based optimization
    best_params = params
    best_score = objective(params)
    
    for run in range(10):
        if run == 0:
            p = params.copy()
        else:
            p = params.copy()
            p += np.random.normal(0, 0.5, size=p.size)
            
        try:
            res = minimize(objective, p, method='L-BFGS-B', options={'maxiter': 500, 'maxfev': 1000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue

    opt_params = best_params
    inner = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = np.degrees(opt_params[3 * i + 2])
        inner.append([cx, cy, angle])
    
    # Compute the final S using the exact max (not soft max)
    vertices_list = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = opt_params[3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
    S = compute_S_exact(vertices_list)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
