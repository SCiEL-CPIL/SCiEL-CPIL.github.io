import numpy as np

N = 12

def construct():
    from scipy.optimize import minimize
    
    sqrt3 = np.sqrt(3)
    
    def compute_L(pos, angs):
        max_proj = 0.0
        for i in range(len(pos)):
            cx, cy = pos[i]
            a = np.radians(angs[i])
            for k in range(6):
                theta = a + k * np.pi / 3
                vx = cx + np.cos(theta)
                vy = cy + np.sin(theta)
                for m in range(6):
                    phi = m * np.pi / 3
                    proj = abs(vx * np.cos(phi) + vy * np.sin(phi))
                    if proj > max_proj:
                        max_proj = proj
        return max_proj / (sqrt3 / 2)
    
    # Better initial arrangement: hexagonal cluster
    # Center + 6 ring + 5 outer
    positions = [[0, 0]]
    # Ring 1: 6 hexagons at distance sqrt3, angles 0,60,120,180,240,300
    for k in range(6):
        theta = k * np.pi / 3
        positions.append([sqrt3 * np.cos(theta), sqrt3 * np.sin(theta)])
    # Ring 2: 5 hexagons at distance 2*sqrt3, skip one direction
    for k in range(5):
        theta = k * np.pi / 3  # 0,60,120,180,240
        positions.append([2*sqrt3 * np.cos(theta), 2*sqrt3 * np.sin(theta)])
    
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)
    
    angles = np.zeros(12)
    
    def objective(params):
        pos = params[:24].reshape(12, 2)
        angs = params[24:]
        return compute_L(pos, angs)
    
    x0 = np.concatenate([positions.flatten(), angles])
    
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    if res.x is not None and np.all(np.isfinite(res.x)):
        pos = res.x[:24].reshape(12, 2)
        angs = res.x[24:]
    else:
        pos = positions
        angs = angles
    
    L = compute_L(pos, angs)
    angs_deg = np.degrees(angs)
    
    inner = [[pos[i,0], pos[i,1], angs_deg[i]] for i in range(12)]
    
    return inner, (0.0, 0.0), L, 0.0

