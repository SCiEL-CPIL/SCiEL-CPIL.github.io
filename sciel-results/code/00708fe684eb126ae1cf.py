import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 55

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # 1. Generate Icosahedral Lattice Points
    phi = (1 + np.sqrt(5)) / 2
    # Basis vectors for icosahedral lattice
    a1 = np.array([0, 1, phi])
    a2 = np.array([phi, 0, 1])
    a3 = np.array([1, phi, 0])
    
    # Generate points with |n1| + |n2| + |n3| <= 2
    points = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            for n3 in range(-2, 3):
                if abs(n1) + abs(n2) + abs(n3) <= 2:
                    p = n1 * a1 + n2 * a2 + n3 * a3
                    points.append(p)
    
    points = np.array(points)
    # We expect 49 points here
    # Add 6 points to reach 55.
    # The missing points in the Mackay icosahedron construction often correspond to the (1,1,1) type or similar.
    # Let's add 6 points at the vertices of an octahedron: (1,0,0), (-1,0,0), etc., scaled.
    # Actually, the 55-atom Mackay icosahedron is exactly the set of lattice points with |n1|+|n2|+|n3| <= 2 
    # IF the basis is defined slightly differently or if I missed a symmetry.
    # Let's check the count again.
    # 0: 1
    # 1: 6
    # 2: 42
    # Total 49.
    
    # To get 55, we add 6 points.
    # A common completion is the 6 points (1,1,1) type in the dual lattice?
    # Let's just add 6 points at (s,0,0), (-s,0,0), (0,s,0), etc.
    # And optimize s.
    
    octa_points = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    # Total 55 points
    # We will scale the lattice points and the octahedral points separately.
    
    def energy_and_grad(scales):
        s_latt = scales[0]
        s_octa = scales[1]
        
        pos_latt = points * s_latt
        pos_octa = octa_points * s_octa
        pos = np.vstack([pos_latt, pos_octa])
        
        # Calculate LJ Energy and Gradient
        n = len(pos)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(r2, 1.0) # Avoid division by zero
        r = np.sqrt(r2)
        
        # LJ Potential: 4 * (r^-12 - r^-6)
        r6 = r**-6
        r12 = r6**2
        U = 4 * (r12 - r6)
        np.fill_diagonal(U, 0.0)
        energy = np.sum(U)
        
        # Gradient
        # dU/dr = 4 * (-12*r^-13 + 6*r^-7) = 24 * (r^-7 - 2*r^-13)
        # dU/dpos_i = sum_j dU/dr_ij * d r_ij / d pos_i
        # d r_ij / d pos_i = (pos_i - pos_j) / r_ij
        # So dU/dpos_i = sum_j 24 * (r^-7 - 2*r^-13) * (pos_i - pos_j) / r
        # = sum_j 24 * (r^-8 - 2*r^-14) * (pos_i - pos_j)
        
        coeff = 24 * (r**-8 - 2 * r**-14)
        np.fill_diagonal(coeff, 0.0)
        
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Gradient with respect to scales
        grad_s_latt = np.sum(grad[:49] * points)
        grad_s_octa = np.sum(grad[49:] * octa_points)
        
        return energy, np.array([grad_s_latt, grad_s_octa])
    
    # Initial guess for scales
    # Typical LJ minimum distance is 2^(1/6) ~ 1.122
    # Lattice spacing is roughly 1.
    x0 = np.array([1.0, 2.0])
    
    # Optimize scales
    res = minimize(energy_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-8})
    
    if res.success:
        s_latt, s_octa = res.x
    else:
        s_latt, s_octa = x0
        
    pos_latt = points * s_latt
    pos_octa = octa_points * s_octa
    pos = np.vstack([pos_latt, pos_octa])
    
    # Final Relaxation
    def full_energy_and_grad(x):
        pos = x.reshape(N, 3)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(r2, 1.0)
        r = np.sqrt(r2)
        
        r6 = r**-6
        r12 = r6**2
        U = 4 * (r12 - r6)
        np.fill_diagonal(U, 0.0)
        energy = np.sum(U)
        
        coeff = 24 * (r**-8 - 2 * r**-14)
        np.fill_diagonal(coeff, 0.0)
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return energy, grad.flatten()
        
    x_flat = pos.flatten()
    
    # Check time
    if time.time() - start_time < deadline - 5:
        res2 = minimize(full_energy_and_grad, x_flat, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-10})
        if res2.success:
            pos = res2.x.reshape(N, 3)
            
    return pos
# EVOLVE-BLOCK-END
