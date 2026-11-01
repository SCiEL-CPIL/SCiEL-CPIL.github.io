import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def lj_energy(pos):
    """Calculate Lennard-Jones energy and gradient."""
    # pos is (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    # Avoid division by zero
    dist_sq[dist_sq == 0] = 1e-12
    dist = np.sqrt(dist_sq)
    
    # Mask upper triangle to avoid double counting and self-interaction
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d = dist[mask]
    
    inv_d2 = 1.0 / d**2
    inv_d6 = inv_d2**3
    inv_d12 = inv_d6**2
    
    energy = 4.0 * np.sum(inv_d12 - inv_d6)
    
    # Gradient
    # Force on i due to j: 24 * (2*inv_d12 - inv_d6) / d^2 * (r_i - r_j)
    # dE/dx_i = sum_j 24 * (2*inv_d12_ij - inv_d6_ij) / d_ij^2 * (x_i - x_j)
    # Using vectorization
    # f = 24 * (2*inv_d12 - inv_d6) / dist_sq
    f = 24.0 * (2.0 * inv_d12 - inv_d6) / dist_sq[mask]
    
    # Construct full force matrix
    F = np.zeros((N, N))
    F[mask] = f
    
    # Gradient is sum of forces
    grad = np.sum(F * diff, axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: Symmetric Icosahedral Seed
    # 1 center, 12 icosahedron vertices, 42 outer points
    # The 42 outer points in Mackay icosahedron M(2) are:
    # 20 face centers of the icosahedron
    # 12 vertices of the icosahedron? No.
    # Actually, let's just use a Fibonacci sphere for the outer 54 points? No.
    
    # Let's use a known good symmetric seed:
    # 1 center
    # 12 atoms at icosahedron vertices (radius r1)
    # 42 atoms: 
    #   20 at face centers (radius r2)
    #   12 at ...?
    #   10 at ...?
    
    # Simpler: Use 55 points on a Fibonacci sphere, then relax.
    # To improve symmetry, we can try multiple random rotations or radii.
    
    def fibonacci_sphere(n):
        points = np.zeros((n, 3))
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        points[:, 0] = np.sin(phi) * np.cos(theta)
        points[:, 1] = np.sin(phi) * np.sin(theta)
        points[:, 2] = np.cos(phi)
        return points
    
    # Initial radius guess for N=55 is around 2.5-3.0 sigma
    # We will try a few radii and random perturbations
    
    attempts = 0
    max_attempts = 20
    
    while time.time() - start_time < timeout and attempts < max_attempts:
        attempts += 1
        
        # Generate initial positions
        # 1 atom at center, 54 on a sphere?
        # Or all 55 on a sphere?
        # The Mackay icosahedron has 1 center, 12 inner, 42 outer.
        # Let's try: 1 center, 12 at r=1.12, 42 at r=2.24 (approx)
        
        # Generate icosahedron vertices
        phi = (1 + 5**0.5) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts[0])
        
        # Generate 42 outer points
        # Use Fibonacci sphere for 42 points
        outer_pts = fibonacci_sphere(42)
        
        # Combine
        r1 = 1.12
        r2 = 2.24 + 0.1 * np.random.randn() # Perturb radius
        
        pos = np.vstack([
            np.array([[0, 0, 0]]),
            verts * r1,
            outer_pts * r2
        ])
        
        # Add small random perturbation to break symmetry and avoid saddle points
        pos += np.random.randn(N, 3) * 0.01
        
        # Relax
        try:
            res = minimize(lj_energy, pos, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000})
            if res.success or res.fun < best_energy:
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x
        except Exception:
            pass
        
        if time.time() - start_time > timeout - 10:
            break

    if best_pos is None:
        # Fallback: Fibonacci sphere
        pos = fibonacci_sphere(55) * 2.5
        res = minimize(lj_energy, pos, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        best_pos = res.x
        best_energy = res.fun

    return best_pos
# EVOLVE-BLOCK-END
