"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    np.random.seed(42)
    
    # Initialize with a perturbed grid for diversity
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= N:
                break
            x = (i + 0.5) / side + np.random.uniform(-0.1, 0.1)
            y = (j + 0.5) / side + np.random.uniform(-0.1, 0.1)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append([x, y])
    
    positions = np.array(positions[:N])
    # Initial radii: small, will grow
    radii = np.full(N, 0.02)
    
    def check_validity(pos, rad):
        """Check if all circles are inside unit square and non-overlapping."""
        # Check boundaries
        if np.any(pos[:, 0] - rad < 0) or np.any(pos[:, 0] + rad > 1):
            return False
        if np.any(pos[:, 1] - rad < 0) or np.any(pos[:, 1] + rad > 1):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < rad[i] + rad[j] - 1e-9:
                    return False
        return True
    
    def objective(params):
        """Negative sum of radii, with penalty for invalid configurations."""
        pos = params[:2*N].reshape(N, 2)
        rad = params[2*N:]
        
        # Penalty for going out of bounds
        penalty = 0
        for i in range(N):
            if pos[i,0] - rad[i] < 0:
                penalty += (pos[i,0] - rad[i])**2 * 100
            if pos[i,0] + rad[i] > 1:
                penalty += (pos[i,0] + rad[i] - 1)**2 * 100
            if pos[i,1] - rad[i] < 0:
                penalty += (pos[i,1] - rad[i])**2 * 100
            if pos[i,1] + rad[i] > 1:
                penalty += (pos[i,1] + rad[i] - 1)**2 * 100
        
        # Penalty for overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                overlap = rad[i] + rad[j] - dist
                if overlap > 0:
                    penalty += overlap**2 * 100
        
        return -np.sum(rad) + penalty
    
    # Iterative optimization
    best_params = np.concatenate([positions.flatten(), radii])
    best_score = -np.sum(radii)
    
    # Try multiple random restarts
    for restart in range(5):
        if restart > 0:
            # Perturb initial positions
            pos = positions + np.random.uniform(-0.2, 0.2, size=(N, 2))
            pos = np.clip(pos, 0.05, 0.95)
            rad = np.full(N, np.random.uniform(0.02, 0.05))
        else:
            pos = positions
            rad = radii
        
        params = np.concatenate([pos.flatten(), rad])
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0.001, 0.5))  # radius
        
        result = minimize(objective, params, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
    
    # Extract best solution
    best_pos = best_params[:2*N].reshape(N, 2)
    best_rad = best_params[2*N:]
    
    # Final validity check and fix if needed
    if not check_validity(best_pos, best_rad):
        # Fallback to a valid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([best_pos, best_rad])
# EVOLVE-BLOCK-END
