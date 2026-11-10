import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600  # seconds
    
    # Define the 6 normals for the outer hexagon
    # Angles 30, 90, 150, 210, 270, 330 degrees
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Support of unit hexagon in these normal directions is sqrt(3)/2
    h_inner = np.sqrt(3) / 2
    
    # Initial positions: Honeycomb cluster
    # Axial coordinates
    points_axial = [
        (0,0),
        (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
        (2,0), (2,-1), (1,-2), (-1,-2), (-2,-1)
    ]
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return np.array([x, y])
    
    centers = np.array([axial_to_cartesian(q, r) for q, r in points_axial])
    
    # Identify contacts
    contacts = []
    for i in range(12):
        for j in range(i+1, 12):
            q1, r1 = points_axial[i]
            q2, r2 = points_axial[j]
            dq = q2 - q1
            dr = r2 - r1
            if max(abs(dq), abs(dr), abs(dq+dr)) == 1:
                d = centers[j] - centers[i]
                norm_d = np.linalg.norm(d)
                if norm_d > 1e-6:
                    contacts.append((i, j, d / norm_d))
                
    # Objective and constraints
    def objective(vars):
        return vars[24]
        
    def constraints(vars):
        c = vars[:24].reshape(12, 2)
        S = vars[24]
        cons = []
        
        # Outer containment
        for i in range(12):
            for k in range(6):
                n = normals[k]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, k=k: (v[24] - h_inner) - (v[2*i] * normals[k][0] + v[2*i+1] * normals[k][1])
                })
                
        # Non-overlap
        h_contact = np.sqrt(3)
        for i, j, n in contacts:
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j, n=n: (v[2*i] * n[0] + v[2*i+1] * n[1]) - (v[2*j] * n[0] + v[2*j+1] * n[1]) - h_contact
            })
            
        return cons

    # Initial guess
    x0 = np.zeros(25)
    for i in range(12):
        x0[2*i] = centers[i][0]
        x0[2*i+1] = centers[i][1]
    max_proj = np.max([np.dot(normals[k], centers[i]) for i in range(12) for k in range(6)])
    x0[24] = max_proj + h_inner + 0.1
    
    # Optimize
    best_s = x0[24]
    best_c = centers.copy()
    
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
            if res.success and res.x[24] < best_s:
                best_s = res.x[24]
                best_c = res.x[:24].reshape(12, 2)
        except Exception:
            pass

    # Construct return values
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_c[i, 0]
        inner_hex_data[i, 1] = best_c[i, 1]
        inner_hex_data[i, 2] = 0.0
        
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, best_s
