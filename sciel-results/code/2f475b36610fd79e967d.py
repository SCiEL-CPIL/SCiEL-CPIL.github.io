import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    def calc_S(pos):
        # pos: (N, 2)
        # Normals for outer hexagon (angle 0)
        normals = np.array([
            [np.cos(np.pi/3*i), np.sin(np.pi/3*i)] for i in range(6)
        ])
        
        max_proj = 0
        for i in range(N):
            # Vertices of unit hexagon, angle 0
            # Vertices at angles 0, 60, 120, 180, 240, 300
            vert_angles = np.pi/3 * np.arange(6)
            verts = pos[i] + np.array([np.cos(vert_angles), np.sin(vert_angles)]).T
            
            # Project all vertices onto all normals
            dots = np.abs(verts @ normals.T)
            max_proj = max(max_proj, np.max(dots))
            
        return (2.0 / np.sqrt(3)) * max_proj

    def check_overlap(pos):
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist - 1e-9:
                    return False
        return True

    def objective(params):
        pos = params.reshape(N, 2)
        S = calc_S(pos)
        
        penalty = 0
        min_dist = np.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    penalty += (min_dist - d)**2
                    
        return S + 1000 * penalty

    # Initial configurations
    inits = []
    
    # Config 1: 1 center, 6 neighbors, 4 in second ring
    p1 = [[0.0, 0.0]]
    for i in range(6):
        a = np.pi/3 * i
        p1.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a)])
    for i in range(4):
        a = np.pi/3 * (i + 0.5)
        p1.append([2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a)])
    inits.append(np.array(p1).flatten())
    
    # Config 2: 1 center, 6 neighbors, 3 in second ring, 1 in third?
    # Just random
    np.random.seed(42)
    p2 = np.random.uniform(-3, 3, (N, 2))
    inits.append(p2.flatten())
    
    best_S = float('inf')
    best_pos = None
    
    for x0 in inits:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        pos = res.x.reshape(N, 2)
        
        if check_overlap(pos):
            S = calc_S(pos)
            if S < best_S:
                best_S = S
                best_pos = pos.copy()
                
    if best_pos is None:
        # Fallback to initial config 1 if no valid solution found
        best_pos = np.array(inits[0]).reshape(N, 2)
        best_S = calc_S(best_pos)
        
    inner = [[best_pos[i][0], best_pos[i][1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), best_S, 0.0
