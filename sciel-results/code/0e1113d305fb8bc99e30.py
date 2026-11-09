import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    # Outer hexagon normals (flat-top, side L, centered at 0,0)
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def get_vertices(cx, cy, theta):
        angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        return np.column_stack((vx, vy))
        
    def get_edge_normals(vertices):
        edges = np.roll(vertices, -1, axis=0) - vertices
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1.0
        normals = normals / norms[:, np.newaxis]
        return normals

    def calculate_overlap(v1, v2):
        """
        Returns the minimum penetration depth (positive if overlapping, 0 if disjoint).
        Uses SAT.
        """
        n1 = get_edge_normals(v1)
        n2 = get_edge_normals(v2)
        all_normals = np.vstack([n1, n2])
        
        max_overlap = 0.0
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_overlap:
                max_overlap = overlap
                
        return max_overlap

    def calculate_boundary_violation(verts, L):
        """
        Returns max(0, max_proj - L*sqrt3/2)
        """
        max_violation = 0.0
        limit = L * (sqrt3 / 2.0)
        for j in range(6):
            max_p = np.max(verts[:, 0] * normal_cos[j] + verts[:, 1] * normal_sin[j])
            if max_p > limit:
                violation = max_p - limit
                if violation > max_violation:
                    max_violation = violation
        return max_violation

    def penalty_objective(x):
        """
        Objective: L + lambda * (sum of overlaps^2 + sum of boundary violations^2)
        """
        L = x[-1]
        penalty = 0.0
        
        # Boundary penalties
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i + 1]
            theta_i = x[3*i + 2]
            verts = get_vertices(xi, yi, theta_i)
            viol = calculate_boundary_violation(verts, L)
            penalty += viol * viol
            
        # Overlap penalties
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i + 1]
            theta_i = x[3*i + 2]
            verts_i = get_vertices(xi, yi, theta_i)
            for j in range(i + 1, N):
                xj = x[3*j]
                yj = x[3*j + 1]
                theta_j = x[3*j + 2]
                verts_j = get_vertices(xj, yj, theta_j)
                overlap = calculate_overlap(verts_i, verts_j)
                if overlap > 0:
                    penalty += overlap * overlap
                    
        # Large weight for penalty
        weight = 1000.0
        return L + weight * penalty

    def check_validity(x, tol=1e-6):
        L = x[-1]
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i + 1]
            theta_i = x[3*i + 2]
            verts = get_vertices(xi, yi, theta_i)
            if calculate_boundary_violation(verts, L) > tol:
                return False
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i + 1]
            theta_i = x[3*i + 2]
            verts_i = get_vertices(xi, yi, theta_i)
            for j in range(i + 1, N):
                xj = x[3*j]
                yj = x[3*j + 1]
                theta_j = x[3*j + 2]
                verts_j = get_vertices(xj, yj, theta_j)
                if calculate_overlap(verts_i, verts_j) > tol:
                    return False
        return True

    # Initial guess: Heuristic valid packing
    initial_x = []
    initial_x.extend([0.0, 0.0, 0.0])
    for k in range(6):
        angle = math.radians(60 * k)
        dist = sqrt3
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        initial_x.extend([x, y, 0.0])
    for k in range(5):
        angle = math.radians(60 * k)
        dist = 2.0 * sqrt3
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        initial_x.extend([x, y, 0.0])
    
    # Estimate L for initial guess
    max_proj = 0
    for i in range(N):
        xi = initial_x[3*i]
        yi = initial_x[3*i + 1]
        theta_i = initial_x[3*i + 2]
        verts = get_vertices(xi, yi, theta_i)
        for j in range(6):
            p = np.max(verts[:, 0] * normal_cos[j] + verts[:, 1] * normal_sin[j])
            if p > max_proj:
                max_proj = p
    L_init = max_proj / (sqrt3 / 2.0) + 0.5
    initial_x.append(L_init)
    
    x0 = np.array(initial_x)
    
    # Ensure initial guess is valid, if not, expand L
    # The heuristic above should be valid for L ~ 6.5
    
    best_x = x0.copy()
    best_L = L_init
    
    # Try optimization
    try:
        res = minimize(penalty_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.x[-1] < best_L:
            # Check if the new solution is valid
            if check_validity(res.x):
                best_x = res.x
                best_L = res.x[-1]
            else:
                # If invalid, try to repair by increasing L slightly? 
                # Or just keep the best valid found. 
                # Since penalty method drives to valid, if it's invalid, penalty was not 0.
                # We can try a few more iterations with higher penalty weight?
                # For now, stick with the best valid.
                pass
                
    except Exception as e:
        pass

    # If optimization didn't improve or failed, use the initial heuristic
    # We should verify the initial heuristic is valid
    if not check_validity(best_x):
        # Force validity by increasing L
        # Find max violation
        # ... simple fix: just increase L until valid
        L = best_L
        while not check_validity(np.concatenate([best_x[:-1], [L]])) and L < 10:
            L += 0.1
        best_L = L
        best_x[-1] = L

    inner = []
    for i in range(N):
        x = best_x[3*i]
        y = best_x[3*i + 1]
        theta = best_x[3*i + 2]
        inner.append([x, y, math.degrees(theta)])
        
    return inner, (0.0, 0.0), best_L, 0.0
