"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    
    best_result = None
    best_score = -np.inf
    
    def unpack(z):
        """Convert flat vector to (x, y, r) arrays."""
        n = len(z) // 3
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r
    
    def evaluate(x, y, r):
        """Check validity and compute score."""
        # Boundary: circles must be inside [0,1]^2
        if np.any(r < 0) or np.any(r > 0.5):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        # Non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)
    
    def objective(z):
        """Negative sum of radii with penalty for constraint violations."""
        x, y, r = unpack(z)
        
        # Sum of radii
        score = np.sum(r)
        
        # Boundary penalties
        penalty = 0.0
        for i in range(N):
            # Left boundary
            viol = r[i] - x[i]
            if viol > 0:
                penalty += viol * viol
            # Right boundary
            viol = r[i] - (1.0 - x[i])
            if viol > 0:
                penalty += viol * viol
            # Bottom boundary
            viol = r[i] - y[i]
            if viol > 0:
                penalty += viol * viol
            # Top boundary
            viol = r[i] - (1.0 - y[i])
            if viol > 0:
                penalty += viol * viol
        
        # Non-overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap * overlap
        
        return -score + penalty
    
    def gradient(z):
        """Compute gradient of objective."""
        x, y, r = unpack(z)
        grad = np.zeros_like(z)
        
        # Gradient of -sum(r)
        grad[2::3] = -1.0
        
        # Boundary penalty gradients
        for i in range(N):
            # Left: viol = r - x, d/dx = -2*viol, d/dr = 2*viol
            viol = r[i] - x[i]
            if viol > 0:
                grad[3*i] += -2 * viol       # d/dx
                grad[3*i+2] += 2 * viol      # d/dr
            
            # Right: viol = r - (1-x) = r - 1 + x, d/dx = 2*viol, d/dr = 2*viol
            viol = r[i] - (1.0 - x[i])
            if viol > 0:
                grad[3*i] += 2 * viol
                grad[3*i+2] += 2 * viol
            
            # Bottom: viol = r - y, d/dy = -2*viol, d/dr = 2*viol
            viol = r[i] - y[i]
            if viol > 0:
                grad[3*i+1] += -2 * viol
                grad[3*i+2] += 2 * viol
            
            # Top: viol = r - (1-y) = r - 1 + y, d/dy = 2*viol, d/dr = 2*viol
            viol = r[i] - (1.0 - y[i])
            if viol > 0:
                grad[3*i+1] += 2 * viol
                grad[3*i+2] += 2 * viol
        
        # Non-overlap penalty gradients
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy + 1e-12)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
                    # d(overlap^2)/dx_i = 2*overlap * (-dx/dist)
                    factor = 2 * overlap / dist
                    grad[3*i] += -factor * dx
                    grad[3*i+1] += -factor * dy
                    grad[3*j] += factor * dx
                    grad[3*j+1] += factor * dy
                    # d(overlap)/dr_i = 1, d(overlap^2)/dr_i = 2*overlap
                    grad[3*i+2] += 2 * overlap
                    grad[3*j+2] += 2 * overlap
        
        return grad
    
    # Initial grid configuration
    side = int(np.ceil(np.sqrt(N)))
    base_cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side) 
                  for i in range(side) for j in range(side)]
    
    # Try multiple restarts with perturbations
    rng = np.random.default_rng(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        # Create initial configuration with perturbation
        z0 = np.zeros(3 * N)
        for i in range(N):
            x0, y0, r0 = base_cells[i]
            # Add random perturbation
            px = (rng.random() - 0.5) * 0.1
            py = (rng.random() - 0.5) * 0.1
            pr = (rng.random() - 0.5) * 0.02
            z0[3*i] = np.clip(x0 + px, 0.01, 0.99)
            z0[3*i+1] = np.clip(y0 + py, 0.01, 0.99)
            z0[3*i+2] = np.clip(r0 + pr, 0.01, 0.4)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective,
                z0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(0.001, 0.499)] * (3 * N),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                x, y, r = unpack(result.x)
                score = evaluate(x, y, r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])
                    
        except Exception:
            continue
    
    # If no valid result found, return the trivial seed
    if best_result is None:
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
