import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    # Function to solve for optimal radii given centers
    def solve_radii(centers):
        # centers: (N, 2)
        # Variables: r_0 ... r_{N-1}
        
        # Objective: maximize sum(r) -> minimize -sum(r)
        c = -np.ones(N)
        
        # Inequality constraints: A_ub @ r <= b_ub
        A_ub = []
        b_ub = []
        
        # 1. Pairwise non-overlap: r_i + r_j <= d_ij
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                # r_i + r_j <= dist
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(dist)
                
        # 2. Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        # r_i - min_dist <= 0  =>  r_i <= min_dist
        # This is just a bound on the variable, but linprog handles bounds separately.
        # However, the "min_dist" depends on the center position, which is fixed in this inner call.
        # So we can just use bounds.
        
        bounds = []
        for i in range(N):
            x, y = centers[i]
            max_r = min(x, 1 - x, y, 1 - y)
            if max_r < 0:
                max_r = 0
            bounds.append((0, max_r))
            
        # Solve LP
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
        
        if res.success:
            radii = res.x
            return -res.fun, radii # res.fun is min of -sum(r), so -res.fun is max sum(r)
        else:
            # Fallback: very small radii
            radii = np.zeros(N)
            return 0.0, radii

    # Objective function for outer optimizer (centers)
    def objective(centers_flat):
        centers = centers_flat.reshape(N, 2)
        val, _ = solve_radii(centers)
        return -val # Minimize negative sum of radii

    # Initial guess: 5x5 grid minus corners
    points = []
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            if (i in [0, 4] and j in [0, 4]):
                continue
            points.append([x, y])
            
    centers0 = np.array(points)
    
    # Optimize centers
    # Use Powell or Nelder-Mead for non-smooth objective
    # Powell is robust
    result = minimize(
        objective, 
        centers0.flatten(), 
        method='Powell',
        bounds=[(0, 1)] * (N * 2),
        options={'maxiter': 2000, 'xtol': 1e-6, 'ftol': 1e-6}
    )
    
    if result.success:
        centers_opt = result.x.reshape(N, 2)
    else:
        centers_opt = centers0
        
    # Get final radii
    _, radii_opt = solve_radii(centers_opt)
    
    # Construct output array
    circles = np.zeros((N, 3))
    circles[:, 0] = centers_opt[:, 0]
    circles[:, 1] = centers_opt[:, 1]
    circles[:, 2] = radii_opt
    
    return circles
