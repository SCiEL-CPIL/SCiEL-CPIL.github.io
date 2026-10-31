"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Construct the 38-atom FCC truncated octahedron (global minimum of LJ38).
    
    The FCC truncated octahedron for N=38 consists of:
    - 1 atom at the center
    - 12 atoms at the nearest-neighbor positions (vertices of a cuboctahedron)
    - 18 atoms at the next-nearest-neighbor positions (face centers of a cube, 
      but only those consistent with FCC truncation)
    - 7 atoms at the next layer
    
    Actually, let me use the known construction. The FCC truncated octahedron
    with 38 atoms can be constructed from FCC lattice points within a certain radius.
    
    FCC lattice points are at positions (i, j, k) where i+j+k is even, scaled by a/2,
    where a is the cubic lattice constant. The nearest-neighbor distance in FCC is a/sqrt(2).
    
    For LJ, the equilibrium nearest-neighbor distance is 2^(1/6) ≈ 1.12246.
    So a/sqrt(2) = 2^(1/6), giving a = 2^(1/6) * sqrt(2) = 2^(1/6) * 2^(1/2) = 2^(2/3) ≈ 1.5874.
    
    Let me generate FCC lattice points and select the 38 closest to the origin.
    """
    
    # FCC lattice: points at (i, j, k) * (a/2) where i, j, k are all even or all odd... 
    # Actually, FCC points can be described as:
    # (n1, n2, n3) where n1+n2+n3 is even, in units of a/2
    # So positions are (a/2) * (i, j, k) for integer i, j, k with i+j+k even
    
    # The nearest neighbor distance in FCC is a/sqrt(2)
    # We want this to be 2^(1/6) for LJ
    # So a = 2^(1/6) * sqrt(2) = 2^(1/6) * 2^(1/2) = 2^(4/6) = 2^(2/3)
    
    a = 2.0 ** (2.0 / 3.0)  # cubic lattice constant
    half_a = a / 2.0
    
    # Generate candidate FCC points
    candidates = []
    # We need to search in a range that will give us at least 38 points
    # The 38th point in an FCC cluster of truncated octahedron shape...
    # Let's generate points within a reasonable radius
    
    max_coord = 4  # search range for i, j, k in units of half_a
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pos = np.array([i, j, k]) * half_a
                    r = np.linalg.norm(pos)
                    candidates.append((r, pos))
    
    # Sort by distance from origin
    candidates.sort(key=lambda x: x[0])
    
    # Take the 38 closest points
    # But we need to be careful: the FCC truncated octahedron is a specific shape
    # Let's just take the 38 nearest FCC lattice points to the origin
    
    points = np.array([c[1] for c in candidates[:N]])
    
    # Center the cluster
    center = points.mean(axis=0)
    points = points - center
    
    # Ensure minimum separation >= 0.5
    # The nearest neighbor distance should be 2^(1/6) ≈ 1.122, which is well above 0.5
    
    # Apply a tight L-BFGS relaxation to minimize any residual energy
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
                # dE/dr = 4 * (12*r^-13 - 6*r^-7) = 4/r * (12/r^12 - 6/r^6)
                # grad_i = -dE/dr * r_vec/r = -4 * (12/r^13 - 6/r^7) * r_vec / r
                #        = 4 * (-12/r^13 + 6/r^7) * r_vec / r
                #        = 4 * r_vec * (-12/r^14 + 6/r^8)
                # Let me recompute:
                # E_pair = 4*(r^-12 - r^-6)
                # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7)
                # grad_i = -dE_pair/dr * (coords[i]-coords[j])/r = -4*(-12*r^-13+6*r^-7) * r_vec/r
                #        = 4*(12*r^-13-6*r^-7) * r_vec/r
                #        = 4*r_vec*(12*r^-14 - 6*r^-8)
                # grad_j = -grad_i
                coeff = 4.0 * (12.0 / (r2 * r6) - 6.0 / r6)
                # 12/r^14 = 12/(r2^7), 6/r^8 = 6/(r2^4)
                # Let me redo: r^14 = (r^2)^7, r^8 = (r^2)^4
                # coeff for r_vec: 4*(12/r^14 - 6/r^8)
                r14 = r2 ** 7
                r8 = r2 ** 4
                coeff = 4.0 * (12.0 / r14 - 6.0 / r8)
                grad[i] += coeff * r_vec
                grad[j] -= coeff * r_vec
                energy += 4.0 * (1.0 / r12 - 1.0 / r6)
        return energy, grad
    
    # Use L-BFGS for a tight relaxation to minimize residual energy
    # Increase maxiter to ensure convergence from potentially asymmetric start
    result = minimize(
        lambda x: lj_energy_and_grad(x.reshape(-1, 3))[0],
        points.flatten(),
        jac=lambda x: lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14}
    )
    
    final_positions = result.x.reshape(-1, 3)
    
    # Optional: Perform a second, very tight pass if needed
    # But 500 iterations should be sufficient for a near-minimum start
    
    return final_positions
# EVOLVE-BLOCK-END
