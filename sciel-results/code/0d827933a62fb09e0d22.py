import numpy as np
from scipy.optimize import minimize

N = 11

def cost(params):
    n = N
    # Inner hexagons
    inner_verts = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        a_rad = np.radians(a)
        for k in range(6):
            theta = a_rad + k * np.pi / 3
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            inner_verts.append([vx, vy])
    inner_verts = np.array(inner_verts)
    
    cx, cy = params[3*n], params[3*n+1]
    oangle = params[3*n+2]
    
    oangle_rad = np.radians(oangle)
    normals = []
    for k in range(6):
        theta = oangle_rad + k * np.pi / 3
        normals.append([np.cos(theta), np.sin(theta)])
    normals = np.array(normals)
    
    # For each vertex and each normal, compute dot product
    # dot(v - c, n)
    diff = inner_verts - np.array([cx, cy])  # (6n, 2)
    dots = np.dot(diff, normals.T)  # (6n, 6)
    
    max_dot = np.max(dots)
    side = max_dot / (np.sqrt(3) / 2)
    
    if side < 0:
        side = 0
    
    return side

def construct():
    # Initial hexagonal arrangement
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        theta = k * np.pi / 3
        r = np.sqrt(3)
        positions.append([r * np.cos(theta), r * np.sin(theta)])
        angles.append(0.0)
    
    for k in range(4):
        theta = (k + 0.5) * np.pi / 3
        r = 2 * np.sqrt(3)
        positions.append([r * np.cos(theta), r * np.sin(theta)])
        angles.append(0.0)
    
    cx0, cy0 = 0.0, 0.0
    oangle0 = 0.0
    
    params0 = []
    for i in range(N):
        params0.extend([positions[i][0], positions[i][1], angles[i]])
    params0.extend([cx0, cy0, oangle0])
    
    result = minimize(cost, params0, method='Nelder-Mead', 
                      options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    xopt = result.x
    inner = []
    for i in range(N):
        inner.append([xopt[3*i], xopt[3*i+1], xopt[3*i+2]])
    cx, cy, oangle = xopt[3*N], xopt[3*N+1], xopt[3*N+2]
    side = cost(xopt)
    
    return inner, (cx, cy), side, oangle
