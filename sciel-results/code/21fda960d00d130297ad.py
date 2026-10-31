import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def pairwise_dists(pts):
        # pts: (N, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        return d[iu]
    
    def objective(params):
        pts = params.reshape(N, D)
        d = pairwise_dists(pts)
        # Smooth max and min using log-sum-exp
        t = 50.0  # temperature parameter for smoothness
        # Clip to avoid overflow
        d_clipped = np.clip(d, 1e-10, 1e10)
        soft_max = np.log(np.sum(np.exp(t * d_clipped))) / t
        soft_min = -np.log(np.sum(np.exp(-t * d_clipped))) / t
        ratio = soft_max / soft_min
        return ratio ** 2
    
    def true_score(pts):
        d = pairwise_dists(pts)
        return (d.max() / d.min()) ** 2
    
    best_params = None
    best_score = np.inf
    
    # Multiple initial configurations
    inits = []
    
    # 1. Hexagonal lattice arrangement
    # Arrange 16 points in a hexagonal pattern
    # Rows of 4,4,4,4 in hexagonal offset
    hex_pts = []
    row_offsets = [0, 0.5, 0, 0.5]  # alternating offsets for hexagonal
    row_spacing = np.sqrt(3) / 2
    for r in range(4):
        for c in range(4):
            x = c + row_offsets[r]
            y = r * row_spacing
            hex_pts.append([x, y])
    inits.append(np.array(hex_pts, dtype=float))
    
    # 2. Circle arrangement: 1 center + 15 on circle? No, let's do 2 rings
    # Inner ring: 1 point at center
    # Middle ring: 5 points
    # Outer ring: 10 points
    # Actually, let's do: center + 5 inner + 10 outer
    circle_pts = []
    circle_pts.append([0, 0])
    for i in range(5):
        angle = 2 * np.pi * i / 5
        circle_pts.append([np.cos(angle), np.sin(angle)])
    for i in range(10):
        angle = 2 * np.pi * i / 10
        circle_pts.append([2 * np.cos(angle), 2 * np.sin(angle)])
    inits.append(np.array(circle_pts, dtype=float))
    
    # 3. 4x4 grid
    grid_pts = []
    for r in range(4):
        for c in range(4):
            grid_pts.append([c, r])
    inits.append(np.array(grid_pts, dtype=float))
    
    # 4. Random
    np.random.seed(42)
    for _ in range(5):
        random_pts = np.random.randn(N, D)
        inits.append(random_pts)
    
    # 5. Square with rounded corners / other symmetric
    # 3x3 grid + 1 center offset? No, let's try a different symmetric config
    # Points on vertices and edges of a square, plus interior
    sq_pts = []
    # 4 corners
    sq_pts.extend([[0,0],[1,0],[1,1],[0,1]])
    # 4 edge midpoints
    sq_pts.extend([[0.5,0],[1,0.5],[0.5,1],[0,0.5]])
    # 4 interior in a smaller square
    sq_pts.extend([[0.3,0.3],[0.7,0.3],[0.7,0.7],[0.3,0.7]])
    # 4 more interior
    sq_pts.extend([[0.5,0.2],[0.8,0.5],[0.5,0.8],[0.2,0.5]])
    inits.append(np.array(sq_pts, dtype=float))
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        init = init.copy()
        # Center the points
        init -= init.mean(axis=0)
        
        result = minimize(
            objective,
            init.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_score:
            cand = result.x.reshape(N, D)
            cand_score = true_score(cand)
            if cand_score < best_score:
                best_score = cand_score
                best_params = cand.copy()
        
        if time.time() > deadline - 10:
            break
    
    # If we still have time, do a polish pass on the best
    if best_params is not None and time.time() < deadline - 5:
        result = minimize(
            objective,
            best_params.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if result.fun < best_score:
            cand = result.x.reshape(N, D)
            cand_score = true_score(cand)
            if cand_score < best_score:
                best_score = cand_score
                best_params = cand.copy()
    
    if best_params is None:
        # Fallback to grid
        best_params = np.array([[c, r] for r in range(4) for c in range(4)], dtype=float)
    
    return best_params
