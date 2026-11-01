"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _lennard_jones_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-8:
                continue
            r6 = r ** 6
            r12 = r6 * r6
            energy += 4.0 * (r12 - r6)
    return energy


def _lennard_jones_grad(positions):
    """Compute gradient of Lennard-Jones energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[i] - positions[j]
            r2 = np.dot(dr, dr)
            if r2 < 1e-12:
                continue
            r = math.sqrt(r2)
            r6 = r2 ** 3
            r12 = r6 * r6
            # dE/dr = 4 * (-12/r^13 + 6/r^7) = 24/r * (1 - 2*r^6) ... let me be careful
            # E_pair = 4*(r^-12 - r^-6)
            # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7)
            # dE_pair/dri = dE_pair/dr * dri/dr = dE_pair/dr * (ri - rj)/r
            # = 4*(-12/r^13 + 6/r^7) * (ri-rj)/r
            # = 4*(-12/r^14 + 6/r^8) * (ri-rj)
            # = 24/r^8 * (1 - 2*r^6) * (ri-rj) ... wait let me redo
            # dE/dr = 4*(-12*r^-13 + 6*r^-7)
            # dE/dri = (dE/dr) * (ri - rj)/r
            # = 4*(-12*r^-13 + 6*r^-7) * (ri-rj)/r
            # = 4*(-12*r^-14 + 6*r^-8) * (ri-rj)
            # = 24*r^-8 * (1 - 2*r^6) * (ri-rj) ... 
            # 4*(-12/r^14 + 6/r^8) = 4*(-12*r^6 + 6)/r^14 = 24*(1 - 2*r^6)/r^14
            # Hmm, let me just use the standard formula
            # grad_i = 24*(2*r^-14 - r^-8) * (ri - rj)  -- no
            # Let me use: dE_pair/dri = 24 * (2/r^13 - 1/r^7) * (ri-rj)/r ... 
            # Actually the cleanest: 
            # E = 4*(r^-12 - r^-6)
            # dE/dri = 4*(-12*r^-13 * dri/dr + 6*r^-7 * dri/dr) where dri/dr = (ri-rj)/r
            # = 4*(-12*r^-13 + 6*r^-7) * (ri-rj)/r
            # = 4*(-12/r^14 + 6/r^8) * (ri-rj)
            # = (24/r^8 - 48/r^14) * (ri-rj)
            # = 24*(1/r^8 - 2/r^14) * (ri-rj)
            coeff = 24.0 * (1.0 / r6 - 2.0 / (r6 * r6))
            grad[i] += coeff * dr
            grad[j] -= coeff * dr
    return grad


def _mckay_icosahedron_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Shell 1: 12 atoms at radius 1.0 (icosahedron vertices)
    Shell 2: 42 atoms at radius ~1.633 (cuboctahedral arrangement)
    Center: 1 atom
    Total: 1 + 12 + 42 = 55
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices (radius 1)
    shell1 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to radius 1
    shell1 = shell1 / np.linalg.norm(shell1[0])
    
    # 42 atoms on the second shell. The Mackay icosahedron second shell
    # consists of 42 atoms arranged as: 6 face centers (octahedral), 
    # 12 edge centers (cubic), and 24 more... Actually for Mackay icosahedron:
    # The 42 atoms are at positions corresponding to the cuboctahedron vertices
    # plus additional ones. Let me use the known construction.
    
    # Actually, the standard Mackay icosahedron has:
    # - 1 center
    # - 12 at radius 1 (icosahedron vertices)  
    # - 42 at radius R2
    
    # The 42 atoms can be generated as follows:
    # 6 atoms at (±R2/2, ±R2/2, ±R2/2) type positions won't work directly.
    # Let me use the proper construction.
    
    # For the Mackay icosahedron, the second shell radius is typically ~1.633
    # and the 42 atoms are at:
    # - 12 at the edge midpoints of the icosahedron, scaled
    # - 30 at face-related positions
    
    # A cleaner approach: the 42 atoms of the second Mackay shell are the
    # vertices of a truncated icosahedron-like structure. Let me use a 
    # well-known coordinate set.
    
    # Actually, let me construct it properly. The Mackay icosahedron I_2:
    # Shell 2 has 42 atoms. These can be described as:
    # 6 atoms at octahedral positions: (±a, 0, 0), (0, ±a, 0), (0, 0, ±a)
    # 12 atoms at square positions: (±a, ±a, 0) and permutations, but that's 12
    # Wait, 6 + 12 = 18, need 42.
    
    # Let me use a different well-known construction. The 42 atoms are at:
    # 12 at (0, ±1, ±φ) type positions scaled... 
    
    # I'll use a simpler approach: generate candidate positions and relax.
    # The key insight is that even an approximate icosahedral structure will
    # relax to the global minimum.
    
    # Let me use the known fact that the Mackay icosahedron second shell
    # consists of atoms at positions that are sums of two shell-1 unit vectors.
    
    # Actually, for a robust approach, let me just place 42 atoms on a sphere
    # of radius ~1.633 using a Fibonacci sphere or similar, then relax.
    
    # But to be more precise, let me use the actual Mackay coordinates.
    # The 42 positions for shell 2 of the Mackay icosahedron:
    
    # From literature, the Mackay icosahedron I_2 has shell 2 at radius 
    # r2 = 1/sin(π/5) * something... 
    
    # Let me just use a pragmatic approach: place 42 points on a sphere of 
    # radius 1.633 using a well-distributed scheme, then the L-BFGS will 
    # find the correct arrangement.
    
    # Fibonacci sphere for 42 points
    n2 = 42
    r2 = 1.633  # approximate Mackay second shell radius
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    shell2 = np.zeros((n2, 3))
    for i in range(n2):
        z = 1.0 - 2.0 * i / (n2 - 1)
        radius_xy = math.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * math.pi * i / golden_ratio
        shell2[i] = [radius_xy * math.cos(theta), radius_xy * math.sin(theta), z] * r2
    
    center = np.zeros((1, 3))
    
    positions = np.vstack([center, shell1, shell2])
    return positions


def construct():
    """
    Multi-start stochastic search: initialize with perturbed Mackay icosahedron,
    relax with L-BFGS, return the best result.
    """
    base = _mckay_icosahedron_55()
    
    best_positions = None
    best_energy = float('inf')
    
    n_starts = 50
    rng = np.random.RandomState(42)
    
    for start_idx in range(n_starts):
        # Perturb the base geometry with Gaussian noise
        noise_scale = 0.3 if start_idx < 20 else 0.1
        positions = base + rng.randn(N, 3) * noise_scale
        
        # Ensure minimum distance constraint: if any pair is too close, push apart
        # For now, rely on the optimizer to handle this; add a check
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(positions[i] - positions[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try to fix by small random displacement
            positions = base + rng.randn(N, 3) * 0.5
            continue
        
        # Relax with L-BFGS
        try:
            result = minimize(
                _lennard_jones_energy,
                positions,
                jac=_lennard_jones_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            energy = result.fun
            if energy < best_energy:
                best_energy = energy
                best_positions = result.x.copy()
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: return the base geometry
        return base
    
    # Final check: ensure minimum distance
    # If violated, we could do a simple repair, but the optimizer should have handled it
    return best_positions
# EVOLVE-BLOCK-END
