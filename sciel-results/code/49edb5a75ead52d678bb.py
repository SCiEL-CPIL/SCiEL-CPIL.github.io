import numpy as np

def construct():
    N = 11
    coords = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1),
              (2,0), (2,-1), (1,-2), (0,-2)]
    
    points = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        points.append((x, y))
    points = np.array(points)
    
    A_inner = np.sqrt(3) / 2.0
    
    best_S = float('inf')
    best_theta = 0.0
    best_center = np.mean(points, axis=0)
    
    for theta_deg in np.linspace(0, 60, 121):
        theta = np.deg2rad(theta_deg)
        angles = [theta, theta + np.pi/3, theta + 2*np.pi/3]
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Initial center guess
        c = np.mean(points, axis=0)
        
        # Simple gradient-free optimization for center
        # Try shifting center in small steps
        # Or just use the centroid, it's often close enough.
        # Let's stick to centroid for speed and reliability.
        
        max_required_A = 0.0
        for n in normals:
            rel = points - c
            proj = rel @ n
            max_abs = np.max(np.abs(proj))
            required_A = max_abs + A_inner
            if required_A > max_required_A:
                max_required_A = required_A
        
        S = 2.0 / np.sqrt(3) * max_required_A
        if S < best_S:
            best_S = S
            best_theta = theta_deg
            best_center = c
            
    inner = [[float(x), float(y), 0.0] for x, y in points]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_S), float(best_theta)
