"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay


def _hexagonal_init(n):
    """Generate a hexagonal close-packed initial configuration."""
    # Hexagonal lattice parameters
    r0 = 0.05
    dx = 2 * r0
    dy = dx * np.sqrt(3) / 2
    
    centers = []
    # Generate enough points in a hexagonal grid
    for row in range(10):
        for col in range(10):
            x = (col + 0.5) * dx + (0.5 if row % 2 else 0) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append((x, y))
    
    # Shuffle and take first n
    np.random.seed(42)
    np.random.shuffle(centers)
    if len(centers) < n:
        # Fallback to grid if not enough
        side = int(np.ceil(np.sqrt(n)))
        centers = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    
    return np.array(centers[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    # Initial configuration
    centers = _hexagonal_init(N)
    radii = np.full(N, 0.05)
    
    best_score = -np.inf
    best_x = np.hstack([centers.flatten(), radii])
    
    def _objective(x):
        """Minimize negative sum of radii (i.e., maximize sum of radii)."""
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        return -np.sum(radii)
    
    def _constraints_func(x):
        """Return constraint values where positive means satisfied."""
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        
        cons = []
        
        # Boundary constraints: circle must be inside [0,1]^2
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        cons.append(centers[:, 0] - radii)  # x - r >= 0
        cons.append(1 - centers[:, 0] - radii)  # 1 - x - r >= 0
        cons.append(centers[:, 1] - radii)  # y - r >= 0
        cons.append(1 - centers[:, 1] - radii)  # 1 - y - r >= 0
        
        # Pairwise non-overlap: ||c_i - c_j|| >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                cons.append(d - radii[i] - radii[j])
        
        return np.concatenate(cons)
    
    def _solve_slsqp(x0, maxiter=500, tol=1e-8):
        """Solve using SLSQP."""
        constraints = []
        
        # Add boundary constraints as separate inequality constraints
        # x_i - r_i >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        # 1 - x_i - r_i >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*N+i]})
        # y_i - r_i >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        # 1 - y_i - r_i >= 0
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*N+i]})
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - x[2*N+i] - x[2*N+j]
                })
        
        bounds = [(0, 1)] * (2*N) + [(0.001, 0.5)] * N
        
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': maxiter, 'ftol': tol})
        return res
    
    # Try SLSQP from initial configuration
    x0 = np.hstack([centers.flatten(), radii])
    
    # Multi-start with different random initializations
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if s == 0:
            x0 = np.hstack([centers.flatten(), radii])
        else:
            # Perturb initial configuration
            np.random.seed(s * 100)
            c = centers + np.random.randn(N, 2) * 0.02
            c = np.clip(c, 0.05, 0.95)
            r = np.full(N, 0.04) + np.random.rand(N) * 0.02
            x0 = np.hstack([c.flatten(), r])
        
        try:
            res = _solve_slsqp(x0, maxiter=300, tol=1e-7)
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Polish with tighter tolerance if time permits
    if time.time() < deadline - 20 and best_score > -np.inf:
        try:
            res = _solve_slsqp(best_x, maxiter=500, tol=1e-10)
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Extract final result
    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    
    # Ensure validity: clip radii and centers
    radii = np.clip(radii, 0.001, 0.5)
    centers = np.clip(centers, radii, 1 - radii)
    
    # Fix any remaining overlaps by reducing radii
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j]:
                # Reduce both radii
                excess = (radii[i] + radii[j] - d) / 2
                radii[i] = max(0.001, radii[i] - excess)
                radii[j] = max(0.001, radii[j] - excess)
                # Re-clip centers
                centers[i] = np.clip(centers[i], radii[i], 1 - radii[i])
                centers[j] = np.clip(centers[j], radii[j], 1 - radii[j])
    
    # Build result array
    result = np.column_stack([centers[:, 0], centers[:, 1], radii])
    
    return result
# EVOLVE-BLOCK-END
