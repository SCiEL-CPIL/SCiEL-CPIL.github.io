import numpy as np
from scipy.optimize import minimize_scalar
import time

N = 12

def construct():
    start_time = time.time()
    
    # 1. Define the 3x4 hexagonal lattice cluster
    a = np.sqrt(3.0)
    
    points = []
    for i in range(3):
        for j in range(4):
            x = i * a + (j % 2) * (a / 2.0)
            y = j * 1.5
            points.append([x, y])
            
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 2. Objective function
    def objective(theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        p_rot = points_centered @ R.T
        
        widths = []
        for k in range(6):
            angle = theta + k * np.pi / 3.0
            n = np.array([np.cos(angle), np.sin(angle)])
            proj = p_rot @ n
            w = np.max(proj) - np.min(proj)
            widths.append(w)
            
        max_width = np.max(widths)
        return max_width / np.sqrt(3.0)

    # 3. Optimize
    res = minimize_scalar(objective, bounds=(0, np.pi/3.0), method='bounded', 
                          options={'xatol': 1e-6})
    
    if res.success:
        best_theta = res.x
        best_S = res.fun
    else:
        thetas = np.linspace(0, np.pi/3.0, 200)
        scores = [objective(t) for t in thetas]
        idx = np.argmin(scores)
        best_theta = thetas[idx]
        best_S = scores[idx]
        
    # 4. Finalize
    c, s = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    final_points = points_centered @ R.T
    
    inner = []
    angle_deg = np.degrees(best_theta)
    for p in final_points:
        inner.append([float(p[0]), float(p[1]), angle_deg])
        
    center = [0.0, 0.0]
    side = float(best_S)
    angle = angle_deg
    
    return inner, center, side, angle
