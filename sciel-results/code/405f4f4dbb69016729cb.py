import numpy as np
import time

N = 38

def lj_energy_and_grad(pos):
    """
    Compute Lennard-Jones energy and gradient.
    pos: (N, 3) array
    """
    # Pairwise distances
    # r_ij = |pos_i - pos_j|
    # We need sum over i < j of 4 * (r^-12 - r^-6)
    
    # Using broadcasting for clarity, though O(N^2) memory is fine for N=38
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2**3
    r12 = r6**2
    
    # Energy term for pair (i,j) is 4 * (r^-12 - r^-6)
    # Since we sum over all i,j, we double count, so divide by 2 at the end.
    # Or just sum over upper triangle.
    # Let's use the full matrix and divide by 2.
    pair_energy = 4.0 * (1.0/r12 - 1.0/r6)
    total_energy = 0.5 * np.sum(pair_energy)
    
    # Gradient
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7) = 24 * (r^-7 - 2 r^-13)
    # dE/dpos_i = sum_{j != i} dE/dr * (pos_i - pos_j) / r
    # Let's compute vector gradient
    # grad_r = 24 * (r^-7 - 2 r^-13)
    # Note: r = sqrt(r2)
    # r^-7 = r2^-3.5
    # r^-13 = r2^-6.5
    
    r = np.sqrt(r2)
    # Avoid zero division again, though diagonal is handled
    np.fill_diagonal(r, 1.0)
    
    # dE/dr for each pair
    dEdr = 24.0 * (r**-7 - 2.0 * r**-13)
    
    # Gradient with respect to position
    # grad_pos_i = sum_j dEdr_ij * (pos_i - pos_j) / r_ij
    # diff is pos_i - pos_j
    # So we need dEdr * diff / r
    # dEdr / r = 24 * (r^-8 - 2 r^-14)
    
    # Let's just use dEdr / r
    dEdr_over_r = dEdr / r
    np.fill_diagonal(dEdr_over_r, 0.0)
    
    # grad = sum_j dEdr_over_r_ij * (pos_i - pos_j)
    # This is equivalent to dEdr_over_r @ diff
    # diff shape (N, N, 3)
    # dEdr_over_r shape (N, N)
    
    grad = np.sum(dEdr_over_r[:, :, np.newaxis] * diff, axis=1)
    
    return total_energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    # 1. Generate FCC lattice points
    # FCC lattice: points (x,y,z) where x+y+z is even.
    # We need 38 points closest to origin.
    # Let's generate points in a box [-4, 4]^3
    coords = []
    for x in range(-4, 5):
        for y in range(-4, 5):
            for z in range(-4, 5):
                if (x + y + z) % 2 == 0:
                    coords.append([x, y, z])
    
    coords = np.array(coords, dtype=float)
    
    # 2. Sort by distance to origin
    dists = np.sum(coords**2, axis=1)
    sorted_indices = np.argsort(dists)
    init_coords = coords[sorted_indices[:N]]
    
    # 3. Scale to LJ equilibrium
    # Find minimum distance between any pair in init_coords
    diff = init_coords[:, np.newaxis, :] - init_coords[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_dist = np.sqrt(np.min(r2))
    
    # Target min distance is 2^(1/6)
    target_dist = 2.0 ** (1.0/6.0)
    scale_factor = target_dist / min_dist
    
    pos = init_coords * scale_factor
    
    # 4. Local Minimization
    # Use L-BFGS-B
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            pos_2d = x.reshape(-1, 3)
            # Enforce minimum distance constraint? 
            # L-BFGS-B handles bounds, but not non-linear constraints easily without penalty.
            # The prompt says atoms must be at least 0.5 apart.
            # Our initial guess is scaled to 2^(1/6) ~ 1.12, so we are safe.
            # During minimization, distances might shrink.
            # We can add a penalty or just hope it stays valid.
            # Given the start is very good, it should stay in the basin.
            
            e, g = lj_energy_and_grad(pos_2d)
            return e, g.flatten()
        
        res = minimize(objective, pos.flatten(), method='L-BFGS-B', 
                       jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time.time() < deadline:
            final_pos = res.x.reshape(-1, 3)
            
            # Verify validity
            # Check min distance
            diff_f = final_pos[:, np.newaxis, :] - final_pos[np.newaxis, :, :]
            r2_f = np.sum(diff_f**2, axis=-1)
            np.fill_diagonal(r2_f, np.inf)
            if np.min(r2_f) < 0.25: # 0.5^2
                # If invalid, fall back to initial guess
                final_pos = pos
            else:
                pos = final_pos
                
    except Exception:
        pass # Keep initial guess if minimization fails
    
    return pos
