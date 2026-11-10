import numpy as np
from scipy.optimize import minimize

def hexagon_packing_11():
    # Initial configuration: Honeycomb
    centers = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, np.sqrt(3)],
        [-1.0, np.sqrt(3)],
        [-2.0, 0.0],
        [-1.0, -np.sqrt(3)],
        [1.0, -np.sqrt(3)],
        # Ring 2 (4 hexes)
        [3.0, np.sqrt(3)],      # 30 deg
        [-3.0, np.sqrt(3)],     # 150 deg
        [-3.0, -np.sqrt(3)],    # 210 deg
        [3.0, -np.sqrt(3)]      # 330 deg
    ])
    
    # Variables: x0, y0, ..., x10, y10, theta
    x0 = centers[:, 0].copy()
    y0 = centers[:, 1].copy()
    theta0 = 0.0
    
    v0 = np.concatenate([x0, y0, [theta0]])
    
    # Normals for outer hexagon (angle 0 relative to container frame)
    # The container angle is theta.
    # Normals in world frame: theta + 30, theta + 90, etc.
    
    def objective(v):
        xs = v[:11]
        ys = v[11:22]
        theta = v[22]
        
        centers_curr = np.column_stack((xs, ys))
        
        # Calculate normals
        base_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        normals_angles = base_angles + theta
        normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
        
        # Projections: (11, 6)
        projections = centers_curr @ normals.T
        max_proj = np.max(projections)
        return max_proj

    def objective_grad(v):
        # Subgradient of max is the normal corresponding to the max projection
        xs = v[:11]
        ys = v[11:22]
        theta = v[22]
        
        centers_curr = np.column_stack((xs, ys))
        base_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        normals_angles = base_angles + theta
        normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
        
        projections = centers_curr @ normals.T
        max_idx = np.unravel_index(np.argmax(projections), projections.shape)
        
        # Gradient wrt x, y, theta
        # d(max_proj)/d(x_i) = n_jx if i=max_idx[0], j=max_idx[1]
        # d(max_proj)/d(y_i) = n_jy
        # d(max_proj)/d(theta) = - (x_i sin(a_j) + y_i cos(a_j))
        
        grad = np.zeros_like(v)
        i, j = max_idx
        n_j = normals[j]
        grad[i] = n_j[0]
        grad[11+i] = n_j[1]
        
        # Derivative wrt theta
        # proj = x cos(a+t) + y sin(a+t)
        # d/dt = -x sin(a+t) + y cos(a+t)
        a_t = normals_angles[j]
        grad[22] = -xs[i] * np.sin(a_t) + ys[i] * np.cos(a_t)
        
        return grad

    # Constraints: d_ik >= 2
    constraints = []
    for i in range(11):
        for k in range(i+1, 11):
            def make_constraint(i, k):
                def constraint(v):
                    dx = v[i] - v[k]
                    dy = v[11+i] - v[11+k]
                    return dx*dx + dy*dy - 4.0
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, k)})
    
    # Run optimizer
    # Use SLSQP
    res = minimize(objective, v0, method='SLSQP', jac=objective_grad, 
                   constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
    
    if res.success:
        v_opt = res.x
    else:
        # Fallback to initial if failed
        v_opt = v0
        
    xs_opt = v_opt[:11]
    ys_opt = v_opt[11:22]
    theta_opt = v_opt[22]
    
    centers_opt = np.column_stack((xs_opt, ys_opt))
    
    # Calculate final S
    base_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals_angles = base_angles + theta_opt
    normals = np.column_stack((np.cos(normals_angles), np.sin(normals_angles)))
    projections = centers_opt @ normals.T
    max_proj = np.max(projections)
    
    S = 1.0 + (2.0 / np.sqrt(3.0)) * max_proj
    
    # Construct return values
    inner_data = np.zeros((11, 3))
    inner_data[:, 0] = xs_opt
    inner_data[:, 1] = ys_opt
    inner_data[:, 2] = 0.0 # Inner hexagons keep angle 0 relative to world? 
    # Wait, the problem says "inner ... rows (x, y, angle_deg)".
    # If I rotated the cluster, the inner hexagons are still at angle 0 in the world frame?
    # The "angle" in the tuple is the orientation of the hexagon itself.
    # In the current code, it was 0.
    # If I moved the centers, the hexagons are still "pointy-top" (angle 0).
    # So inner angle is 0.
    
    outer_data = np.array([0.0, 0.0, np.rad2deg(theta_opt)])
    
    return inner_data, outer_data, S

