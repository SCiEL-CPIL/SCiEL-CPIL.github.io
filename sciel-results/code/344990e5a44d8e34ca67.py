"""Construct LJ38 global minimum: FCC truncated octahedron with optimized scale."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar


def _fcc_truncated_octahedron():
    """
    Generate 38 FCC lattice points forming a truncated octahedron
    with nearest-neighbor distance = 1.
    
    Shells: 1 (center), 12 (nearest neighbors at d=1), 
           14 (at d=sqrt(2)), 11 (at d=sqrt(3))
    Total: 38
    """
    points = []
    
    # Shell 0: center
    points.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 nearest neighbors at distance 1
    # These are the FCC nearest neighbors: permutations of (±1, ±1, 0)/sqrt(2) * sqrt(2) = (±1, ±1, 0)
    # Wait, in FCC the nearest neighbors are at distance a where a is the NN distance.
    # For FCC with NN distance = 1, the basis vectors are (1,0,0), (0,1,0), (0,0,1) 
    # but actually FCC nearest neighbors are at (±0.5, ±0.5, 0) type positions times 2...
    # Let me think more carefully.
    # 
    # In an FCC lattice with lattice constant a, the nearest neighbor distance is a/sqrt(2).
    # So if NN distance = 1, then a = sqrt(2).
    # The FCC lattice points are: (i,j,k) where i+j+k is even, scaled by a.
    # 
    # Actually, let me just use the standard FCC construction.
    # FCC primitive vectors: a1 = (0, a/2, a/2), a2 = (a/2, 0, a/2), a3 = (a/2, a/2, 0)
    # with a = sqrt(2) so NN distance = 1.
    # 
    # But it's easier to think in terms of the conventional cubic cell.
    # FCC atoms in conventional cell of side a: corners + face centers.
    # For a truncated octahedron of 38 atoms, we take a specific set of FCC points.
    # 
    # Let me use the known structure: the 38 atoms are at FCC lattice positions
    # (in units where NN distance = 1) forming a truncated octahedron.
    # 
    # I'll generate all FCC points within a certain radius and select the 38 closest to origin.
    
    a = np.sqrt(2)  # lattice constant so that NN distance = 1
    
    # Generate FCC points: all (i*a, j*a, k*a) where i,j,k are integers and i+j+k is even
    # plus the face-center offsets... actually FCC is: points where all coords are integers 
    # or all are half-integers with same parity... 
    # 
    # Simpler: FCC = union of simple cubic lattice at (0,0,0) and at (a/2, a/2, a/2) etc.
    # Standard FCC: points (i*a, j*a, k*a) for i,j,k in Z, plus (a/2)(i,j,k) where i,j,k 
    # are all odd... no.
    # 
    # FCC conventional cell: atoms at (0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)
    # So the full lattice is: (i*a + dx, j*a + dy, k*a + dz) where (dx,dy,dz) in 
    # {(0,0,0), (0,a/2,a/2), (a/2,0,a/2), (a/2,a/2,0)}
    
    max_coord = 3  # search up to this coordinate range
    
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for offset in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    x = i * a + offset[0]
                    y = j * a + offset[1]
                    z = k * a + offset[2]
                    fcc_points.append([x, y, z])
    
    fcc_points = np.array(fcc_points)
    
    # Compute distances from origin
    dists = np.linalg.norm(fcc_points, axis=1)
    
    # Sort by distance and take the 38 closest
    idx = np.argsort(dists)
    points = fcc_points[idx[:N]]
    
    return points


def _lj_energy(positions):
    """Compute total LJ energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    # Avoid division by zero
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Sum over all pairs (each pair counted twice, so divide by 2)
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6) / 2.0
    return energy


def construct():
    # Get the FCC truncated octahedron with NN distance = 1
    base_points = _fcc_truncated_octahedron()
    
    # Optimize the scale factor to minimize LJ energy
    # Energy as function of scale s: E(s) = LJ_energy(base_points * s)
    
    def energy_fn(s):
        return _lj_energy(base_points * s)
    
    # Use bounded scalar optimization
    # The optimal scale is near 1.0 (NN distance ~1.12 in LJ units where sigma=1)
    # Actually for LJ, the minimum of 4*(r^-12 - r^-6) is at r = 2^(1/6) ≈ 1.1225
    # So we expect the optimal NN distance to be around 1.1225
    result = minimize_scalar(energy_fn, bounds=(0.9, 1.3), method='bounded', 
                             options={'xatol': 1e-10, 'maxiter': 1000})
    
    optimal_scale = result.x
    positions = base_points * optimal_scale
    
    # Verify minimum separation >= 0.5
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        # Fallback: use the seed program's approach
        side = int(np.ceil(N ** (1 / 3)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        positions = np.array(pts[:N], dtype=float) * 1.12
    
    return positions
# EVOLVE-BLOCK-END
