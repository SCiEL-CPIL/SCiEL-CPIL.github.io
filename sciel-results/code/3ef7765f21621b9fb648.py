import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    def fibonacci_sphere(n):
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            points.append([np.cos(theta) * radius, y, np.sin(theta) * radius])
        return np.array(points)
    
    unit_vectors = fibonacci_sphere(N)
    
    def objective(params):
        thetas = params[:N]
        phis = params[N:]
        
        sin_theta = np.sin(thetas)
        cos_theta = np.cos(thetas)
        cos_phi = np.cos(phis)
        sin_phi = np.sin(phis)
        
        x = sin_theta * cos_phi
        y = sin_theta * sin_phi
        z = cos_theta
        
        uvecs = np.column_stack((x, y, z))
        
        abs_uvecs = np.abs(uvecs)
        M = np.max(abs_uvecs)
        
        dot_products = uvecs @ uvecs.T
        dot_products[np.diag_indices_from(dot_products)] = np.inf
        min_dot = np.min(dot_products)
        D_sq = 2.0 - 2.0 * min_dot
        if D_sq < 0: D_sq = 0.0
        D = np.sqrt(D_sq)
        
        if M + D/2.0 == 0:
            return 0.0
            
        val = D / (2.0 * M + D)
        return -val
    
    z = unit_vectors[:, 2]
    thetas = np.arccos(np.clip(z, -1.0, 1.0))
    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    phis = np.arctan2(y, x)
    
    x0 = np.concatenate((thetas, phis))
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14})
        thetas_opt = res.x[:N]
        phis_opt = res.x[N:]
    except Exception:
        thetas_opt = thetas
        phis_opt = phis

    sin_theta = np.sin(thetas_opt)
    cos_theta = np.cos(thetas_opt)
    cos_phi = np.cos(phis_opt)
    sin_phi = np.sin(phis_opt)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    uvecs_opt = np.column_stack((x, y, z))
    
    abs_uvecs = np.abs(uvecs_opt)
    M = np.max(abs_uvecs)
    
    dot_products = uvecs_opt @ uvecs_opt.T
    dot_products[np.diag_indices_from(dot_products)] = np.inf
    min_dot = np.min(dot_products)
    D_sq = 2.0 - 2.0 * min_dot
    D = np.sqrt(max(0.0, D_sq))
    
    R_opt = 0.5 / (M + D/2.0)
    
    centers = R_opt * uvecs_opt
    
    return centers
