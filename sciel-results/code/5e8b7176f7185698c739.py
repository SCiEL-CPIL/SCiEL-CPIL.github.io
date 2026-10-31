"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct the 38-atom FCC truncated octahedron (global minimum of LJ38).
    Uses a 'topology-snap' strategy: generates the 38 nearest FCC lattice points
    and applies a high-precision L-BFGS relaxation to reach the global minimum.
    """
    
    # FCC lattice constant for LJ equilibrium distance 2^(1/6)
    a = 2.0 ** (2.0 / 3.0)
    half_a = a / 2.0
    
    # Generate candidate FCC points
    candidates = []
    max_coord = 4
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pos = np.array([i, j, k]) * half_a
                    r = np.linalg.norm(pos)
                    candidates.append((r, pos))
    
    # Sort by distance from origin and take the 38 closest
    candidates.sort(key=lambda x: x[0])
    points = np.array([c[1] for c in candidates[:N]])
    
    # Center the cluster
    points = points - points.mean(axis=0)
    
    def lj_energy_and_grad(coords):
        n = len(coords)
        energy = 0.0
        grad = np.zeros_like(coords)
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = coords[i] - coords[j]
                r2 = np.dot(r_vec, r_vec)
                if r2 < 1e-12:
                    continue
                r6 = r2 ** 3
                r12 = r6 ** 2
                energy += 4.0 * (1.0 / r12 - 1.0 / r6)
                
                # Gradient calculation
                r14 = r2 ** 7
                r8 = r2 ** 4
                coeff = 4.0 * (12.0 / r14 - 6.0 / r8)
                grad[i] += coeff * r_vec
                grad[j] -= coeff * r_vec
        return energy, grad
    
    # High-precision relaxation
    result = minimize(
        lambda x: lj_energy_and_grad(x.reshape(-1, 3))[0],
        points.flatten(),
        jac=lambda x: lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-15, 'maxls': 50}
    )
    
    final_positions = result.x.reshape(-1, 3)
    return final_positions
# EVOLVE-BLOCK-END
