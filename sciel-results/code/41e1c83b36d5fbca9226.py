"""
Strategy: Basin Hopping with Hybrid Local Optimization on FCC Truncated Octahedron.
Rationale: 
1. The global minimum of LJ38 is the FCC truncated octahedron. 
2. We construct this structure analytically to start in the correct basin.
3. Instead of Simulated Annealing (which uses single-step Metropolis criteria), 
   we use Basin Hopping. In each step, we perturb the structure and then fully 
   minimize it using a robust local optimizer (L-BFGS). 
4. This ensures that after each perturbation, the system is at a local minimum, 
   making the energy landscape comparison more meaningful and helping to escape 
   shallow local minima (like the icosahedral funnel) by jumping to the bottom 
   of new basins.
5. We use a fixed seed for reproducibility and a strict time deadline.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # 5s safety margin before 150s kill
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Compute LJ energy efficiently using vectorized operations."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron.
    This is the putative global minimum structure.
    We generate points in an FCC lattice and select the 38 closest to the center.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    # Sort by distance from center
    points.sort(key=lambda x: x[3])
    
    # Take the 38 closest
    selected = [p[:3] for p in points[:38]]
    
    # Center the cluster
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords

def local_minimize(positions):
    """
    Perform local minimization using L-BFGS-B.
    Returns the minimized positions and the energy.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    try:
        result = minimize(
            energy_func, 
            positions.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-10}
        )
        final_pos = result.x.reshape(N, 3)
        final_energy = result.fun
        return final_pos, final_energy
    except Exception:
        return positions, lj_energy_vectorized(positions)

def basin_hopping(initial_positions, rng, max_steps=200, T=1.0):
    """
    Perform Basin Hopping optimization.
    """
    current_pos = initial_positions.copy()
    current_energy = lj_energy_vectorized(current_pos)
    
    # Initial local minimization
    current_pos, current_energy = local_minimize(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() - START_TIME > DEADLINE - 15:
            break
            
        # Perturb the structure
        # Random displacement
        delta = rng.normal(0, 0.5, size=(N, 3))
        new_pos = current_pos + delta
        
        # Check validity before expensive minimization
        if not check_validity(new_pos):
            continue
            
        # Local minimization of the perturbed structure
        new_pos, new_energy = local_minimize(new_pos)
        
        # Check validity after minimization
        if not check_validity(new_pos):
            continue
            
        # Metropolis criterion
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            delta_E = new_energy - current_energy
            if delta_E < 50.0: # Cap the exponent to avoid underflow
                prob = math.exp(-delta_E / T)
            else:
                prob = 0.0
                
            if rng.uniform() < prob:
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                    
    return best_pos, best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Generate initial FCC structure
    try:
        initial_pos = make_fcc_truncated_octahedron()
        if not check_validity(initial_pos):
            # Fallback to random if FCC generation fails validity
            rng = np.random.default_rng(0)
            initial_pos = rng.uniform(-2, 2, size=(N, 3))
    except Exception:
        rng = np.random.default_rng(0)
        initial_pos = rng.uniform(-2, 2, size=(N, 3))
    
    # 2. Run Basin Hopping
    # We use a fixed seed for reproducibility, but could vary it if time permits.
    # For now, one robust run is preferred.
    
    rng = np.random.default_rng(42)
    
    best_pos_bh = None
    best_energy_bh = float('inf')
    
    try:
        # Parameters:
        # max_steps: 200 is usually enough for BH to converge if starting near global min.
        # T: 1.0 is a standard temperature for LJ BH.
        best_pos_bh, best_energy_bh = basin_hopping(initial_pos, rng, max_steps=200, T=1.0)
    except Exception:
        best_pos_bh = initial_pos
        best_energy_bh = lj_energy_vectorized(initial_pos)
        
    # 3. Final Polish with L-BFGS on the best result
    final_pos = best_pos_bh
    try:
        if time.time() - START_TIME < DEADLINE - 5:
            # One more local minimization to ensure we are at the bottom of the basin
            final_pos, final_energy = local_minimize(best_pos_bh)
            if check_validity(final_pos):
                # Accept if valid
                pass
            else:
                final_pos = best_pos_bh
    except Exception:
        pass
        
    # 4. Final validity check
    if not check_validity(final_pos):
        # Fallback to initial FCC if something went wrong
        final_pos = initial_pos
        if not check_validity(final_pos):
            # Absolute fallback: random valid points
            rng = np.random.default_rng(123)
            pts = []
            for _ in range(N):
                while True:
                    p = rng.uniform(-2, 2, size=3)
                    if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                        pts.append(p)
                        break
            final_pos = np.array(pts)
            
    return final_pos

# EVOLVE-BLOCK-END
