import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # Define apothem directions for container (angle 0)
    # Apothems are at 30, 90, 150, 210, 270, 330 degrees
    angles_apothem = np.deg2rad([30, 90, 150, 210, 270, 330])
    u_vectors = np.array([[np.cos(a), np.sin(a)] for a in angles_apothem])
    
    # Initial guess: The 4.464 configuration
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 3 in gaps (angles 30, 150, 270, 30? No, gaps are at 30, 90...? No)
    # Layer 1 angles: 0, 60, 120, 180, 240, 300
    # Gaps are at 30, 90, 150, 210, 270, 330
    # We pick 4 gaps, e.g., 30, 90, 270, 330? Or symmetric 30, 150, 210, 330?
    # Let's pick 30, 90, 210, 270 (top-right, top-left? No)
    # Let's just use the lattice positions for initialization and let the solver adjust.
    
    # Lattice points for 11 hexes
    # Center (0,0)
    # Layer 1: 6 points
    # Layer 2: 4 points
    
    # Helper to generate lattice points
    def lattice_points():
        pts = [(0,0)]
        # Layer 1
        for k in range(6):
            ang = np.deg2rad(60*k)
            pts.append((np.sqrt(3)*np.cos(ang), np.sqrt(3)*np.sin(ang)))
        # Layer 2 (4 points)
        # Gaps at 30, 90, 150, 210, 270, 330
        # Pick 4: 30, 150, 210, 330
        gap_angles = [30, 150, 210, 330]
        for ang_deg in gap_angles:
            ang = np.deg2rad(ang_deg)
            pts.append((3.0*np.cos(ang), 3.0*np.sin(ang)))
        return np.array(pts)
    
    x0 = lattice_points().flatten()
    
    def objective(x):
        centers = x.reshape(-1, 2)
        # Calculate max projection
        projs = centers @ u_vectors.T
        max_abs_proj = np.max(np.abs(projs))
        A = max_abs_proj + np.sqrt(3)/2
        S = 2 * A / np.sqrt(3)
        return S
    
    def constraints(x):
        centers = x.reshape(-1, 2)
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                # d >= sqrt(3)
                cons.append({'type': 'ineq', 'fun': lambda d=d: d - np.sqrt(3)})
        return cons
    
    res = minimize(objective, x0, method='SLSQP', constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    
    centers = res.x.reshape(-1, 2)
    S = objective(res.x)
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
