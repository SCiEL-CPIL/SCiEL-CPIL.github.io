import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Initial guess: Hexagonal lattice packing
    # 3 rows of 4
    positions = []
    for row in range(3):
        y = row * 1.5
        for col in range(4):
            x = col * sqrt3
            if row % 2 == 1:
                x += sqrt3 / 2
            positions.append([x, y])
    
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)
    
    def calc_min_hexagon(pos):
        # pos: (12, 2)
        # Assume container is flat-to-flat (theta=0) and centered at centroid
        cx = np.mean(pos[:, 0])
        cy = np.mean(pos[:, 1])
        
        x = pos[:, 0] - cx
        y = pos[:, 1] - cy
        
        # Constraints for theta=0 hexagon
        s1 = np.max(np.abs(x))
        s2 = np.max(np.abs(y)) / (sqrt3 / 2)
        s3 = np.max(np.abs(x + y / sqrt3))
        s4 = np.max(np.abs(x - y / sqrt3))
        
        return max(s1, s2, s3, s4)
        
    def objective(vars_):
        pos = vars_.reshape(12, 2)
        
        # Constraint: distance between centers >= sqrt(3)
        # Using a simple loop for N=12 is fast
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < sqrt3 - 1e-5:
                    return 1000.0 + (sqrt3 - dist) * 100
        
        return calc_min_hexagon(pos)
        
    x0 = positions.flatten()
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
        opt_pos = result.x.reshape(12, 2)
    except Exception:
        opt_pos = positions
        
    # Re-center
    opt_pos -= np.mean(opt_pos, axis=0)
    
    # Calculate final S
    S = calc_min_hexagon(opt_pos)
    
    # Construct inner hexagons
    inner = []
    for i in range(12):
        inner.append([opt_pos[i, 0], opt_pos[i, 1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
