"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    return 4.0 * np.sum(r12 - r6)

def _fcc_truncated_octahedron():
    """Construct LJ38 truncated octahedron from FCC lattice."""
    # FCC lattice points: all (i,j,k) where i+j+k is even, within a cube
    # For a truncated octahedron, we take points from FCC lattice
    # The structure has: 1 center + 12 (1,0,0) type + 14 (1,1,0) type + 11 (1,1,1) type
    # Actually, let's use the known coordinates for LJ38 truncated octahedron
    
    # Standard construction: FCC lattice with cutoff
    # Generate FCC points and select 38 closest to center
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC condition: i+j+k even
                if (i + j + k) % 2 == 0:
                    points.append((float(i), float(j), float(k)))
    
    points = np.array(points)
    # Sort by distance from origin
    dists = np.sum(points * points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Center the structure
    points -= points.mean(axis=0)
    
    # Scale to optimal spacing (approximately 1.0 in LJ units for the first shell)
    # The first shell distance in FCC is 1.0 (for lattice parameter a, nearest neighbor is a/sqrt(2))
    # For LJ minimum at r=2^(1/6) ≈ 1.1225, we want nearest neighbors at that distance
    # In our FCC construction, nearest neighbors are at distance 1.0, so scale by 2^(1/6)
    scale = 2.0 ** (1.0 / 6.0)
    points *= scale
    
    return points

def _try_relax(pos, max_time=30.0):
    """Try to relax the structure using science_chemistry.relax."""
    try:
        import science_chemistry
        relaxed = science_chemistry.relax(pos.copy())
        if relaxed is not None and len(relaxed) == N:
            # Check minimum distance constraint
            rij = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            r2 = np.sum(rij * rij, axis=-1)
            np.fill_diagonal(r2, np.inf)
            min_r2 = np.min(r2)
            if min_r2 >= 0.25:  # 0.5^2
                return relaxed
    except Exception:
        pass
    return None

def construct():
    """Construct LJ38 cluster with minimum energy."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with truncated octahedron
    pos = _fcc_truncated_octahedron()
    best_pos = pos.copy()
    best_energy = _lj_energy(best_pos)
    
    # Try relaxation
    t0 = time.time()
    relaxed = _try_relax(pos, max_time=40.0)
    if relaxed is not None:
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
    
    # Check if we have time for additional attempts with perturbations
    elapsed = time.time() - start_time
    if elapsed < deadline - 30.0:
        # Try a few perturbed versions
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            # Small perturbation
            perturbed = best_pos + rng.normal(0, 0.05, size=best_pos.shape)
            t0 = time.time()
            relaxed = _try_relax(perturbed, max_time=20.0)
            if relaxed is not None:
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed
    
    # Final check: ensure minimum distance constraint
    rij = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    if min_r2 < 0.25:
        # If constraint violated, fall back to scaled version
        best_pos = _fcc_truncated_octahedron()
    
    return best_pos
# EVOLVE-BLOCK-END
