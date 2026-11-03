"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate initial hexagonal lattice
    def get_hex_init():
        best_points = None
        best_r = 0
        # Try a range of spacings
        for s in np.linspace(0.20, 0.28, 20):
            k = int(1.0 / s) + 2
            rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            # Filter points that can fit a circle of radius s/2
            mask = radius <= 1.0 - s / 2.0
            points = points[mask]
            if len(points) >= N:
                # Sort by radius to keep central ones
                idx = np.argsort(radius)
                points = points[idx[:N]]
                r = s / 2.0
                # Check validity
                valid = True
                for i in range(N):
                    if np.linalg.norm(points[i]) + r > 1.0 + 1e-9:
                        valid = False
                        break
                    for j in range(i+1, N):
                        if np.linalg.norm(points[i] - points[j]) < 2*r - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and r > best_r:
                    best_r = r
                    best_points = points.copy()
        if best_points is None:
            # Fallback to random
            best_points = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_points, axis=1)
            mask = norms > 0.9
            best_points[mask] *= 0.9 / norms[mask, np.newaxis]
            best_r = 0.05
        return best_points, best_r

    X0, r0 = get_hex_init()
    
    # 2. Define energy and gradient for a fixed r
    def energy_and_grad(X_flat, r):
        X = X_flat.reshape(N, 2)
        E = 0.0
        G = np.zeros_like(X)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = X[i] - X[j]
                d = np.linalg.norm(d_vec)
                if d < 2 * r:
                    overlap = 2 * r - d
                    if d > 1e-8:
                        grad_vec = 2 * overlap * (d_vec / d)
                    else:
                        grad_vec = 2 * overlap * np.random.rand(2)
                    G[i] += grad_vec
                    G[j] -= grad_vec
                    E += overlap * overlap
        
        # Boundary repulsion
        for i in range(N):
            norm = np.linalg.norm(X[i])
            if norm > 1 - r:
                overlap = norm - (1 - r)
                if norm > 1e-8:
                    grad_vec = 2 * overlap * (X[i] / norm)
                else:
                    grad_vec = 2 * overlap * np.random.rand(2)
                G[i] += grad_vec
                E += overlap * overlap
                
        return E, G

    # 3. Binary search on r
    lo = r0 * 0.9
    hi = 0.126
    
    best_X = X0.copy()
    best_r = r0
    
    def optimize_for_r(r, X_init):
        X_flat = X_init.flatten()
        # Use L-BFGS-B
        res = minimize(
            lambda x: energy_and_grad(x, r)[0],
            X_flat,
            jac=lambda x: energy_and_grad(x, r)[1],
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.success:
            return res.x.reshape(N, 2), res.fun
        else:
            # If it fails, return the initial guess and high energy
            return X_init, np.inf

    # Perform binary search
    for _ in range(25):
        if time.time() > deadline:
            break
        mid = (lo + hi) / 2.0
        X_opt, E_val = optimize_for_r(mid, best_X)
        
        # Check if solution is valid (energy close to 0)
        if E_val < 1e-8:
            lo = mid
            best_X = X_opt
            best_r = mid
        else:
            hi = mid
            
    # 4. Final polish
    # Try to improve best_r slightly
    if time.time() < deadline - 5:
        for delta in [0.00005, -0.00005, 0.0001, -0.0001]:
            r_try = best_r + delta
            if r_try > hi or r_try < lo:
                continue
            X_opt, E_val = optimize_for_r(r_try, best_X)
            if E_val < 1e-9:
                best_X = X_opt
                best_r = r_try

    return best_X

# EVOLVE-BLOCK-END
