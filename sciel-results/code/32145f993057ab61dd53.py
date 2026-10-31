"""
Strategy: Stochastic Basin Hopping with Adaptive Perturbation and Safety-Net Templates.
Rationale:
1. The previous strategy relied on generating idealized templates (FCC, Icosahedral, BCC) 
   and locally minimizing them. This is a deterministic template-matching approach.
2. This new strategy uses Basin Hopping (BH), a stochastic global optimization algorithm.
   It starts from the provided seed (cubic lattice fragment) and performs a series of 
   random perturbations followed by local minimization.
3. BH is effective for escaping local minima (like the icosahedral funnel) because 
   the random steps can jump over energy barriers, and the local minimization ensures 
   we land at the bottom of new basins.
4. We use an adaptive step size: start with small perturbations to refine, then increase 
   them to explore, and finally decrease them for final polishing.
5. To ensure robustness, we also include the FCC truncated octahedron template as a 
   "safety net" initial structure. If the BH process finds a lower energy than the 
   FCC template, we keep it. If not, we keep the FCC result. This combines the 
   stochastic exploration power of BH with the deterministic reliability of the 
   known good template.
6. The algorithm respects the 120s deadline by tracking elapsed time and returning 
   the best result found so far.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin: 5s before 120s limit
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

def local_minimize(positions, maxiter=200):
    """
    Perform local minimization using L-BFGS-B.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    try:
        result = minimize(
            energy_func, 
            positions.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        final_pos = result.x.reshape(N, 3)
        final_energy = result.fun
        return final_pos, final_energy
    except Exception:
        return positions, lj_energy_vectorized(positions)

def basin_hopping(start_positions, rng, max_steps=50, step_scale=0.5):
    """
    Perform basin hopping optimization.
    """
    current_pos = start_positions.copy()
    current_energy, _ = local_minimize(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Adaptive step size: start small, increase, then decrease
    # We'll use a fixed schedule for simplicity and determinism
    for i in range(max_steps):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Determine step size based on iteration phase
        if i < max_steps // 3:
            step_size = step_scale * 0.5  # Small steps to refine
        elif i < 2 * max_steps // 3:
            step_size = step_scale * 1.5  # Large steps to explore
        else:
            step_size = step_size = step_scale * 0.5  # Small steps again
        
        # Generate random perturbation
        noise = rng.normal(0, step_size, size=(N, 3))
        perturbed_pos = current_pos + noise
        
        # Local minimize the perturbed structure
        new_pos, new_energy = local_minimize(perturbed_pos)
        
        # Accept if energy is lower (Metropolis criterion with T=0 for simplicity, 
        # or we could use a temperature schedule)
        # For global minimum search, we typically only accept if energy decreases,
        # but that can get stuck. Let's use a simple Metropolis acceptance with 
        # a decaying temperature to allow some uphill moves early on.
        T = 0.5 * (1 - i / max_steps)  # Decaying temperature
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_pos = new_pos.copy()
                best_energy = new_energy
        else:
            # Metropolis acceptance
            delta_E = new_energy - current_energy
            if T > 0:
                accept_prob = math.exp(-delta_E / T)
                if rng.random() < accept_prob:
                    current_pos = new_pos
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_pos = new_pos.copy()
                        best_energy = new_energy
    
    return best_pos, best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_pos = None
    best_energy = float('inf')
    
    # 1. Safety Net: FCC Template
    try:
        fcc_pos = make_fcc_truncated_octahedron()
        fcc_pos_min, fcc_energy = local_minimize(fcc_pos)
        if check_validity(fcc_pos_min):
            best_pos = fcc_pos_min
            best_energy = fcc_energy
    except Exception:
        pass
    
    # 2. Basin Hopping from Seed (Cubic Lattice Fragment)
    # The seed is described as a "cubic lattice fragment". 
    # Let's construct a simple cubic cluster of 38 atoms as the start for BH.
    # A 3x3x5 cubic block has 45 atoms. We can take a subset or use a smaller block.
    # Let's create a 3x3x4 block (36 atoms) and add 2 more, or just use a random 
    # compact cluster as the start for BH, since the seed is fixed but we don't 
    # have its exact coordinates. The task says "the seed is a cubic lattice fragment".
    # We'll generate a 3x3x5 cubic lattice and take the first 38 atoms in a 
    # space-filling curve order to ensure compactness.
    
    rng = np.random.default_rng(42)
    
    # Generate cubic lattice points
    cubic_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                cubic_points.append([i, j, k])
    
    # Sort by distance from center to get compact cluster
    cubic_points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    seed_pos = np.array(cubic_points[:N], dtype=float)
    seed_pos -= np.mean(seed_pos, axis=0)
    
    # Run Basin Hopping from the seed
    try:
        bh_pos, bh_energy = basin_hopping(seed_pos, rng, max_steps=30, step_scale=0.3)
        if check_validity(bh_pos):
            if bh_energy < best_energy:
                best_pos = bh_pos
                best_energy = bh_energy
    except Exception:
        pass
    
    # 3. Additional BH Start from Random Compact Cluster
    # Generate a random compact cluster to diversify the search
    random_pos = []
    for _ in range(N):
        for _ in range(50):
            p = rng.uniform(-1.5, 1.5, size=3)
            if all(np.linalg.norm(p - q) > 0.6 for q in random_pos):
                random_pos.append(p)
                break
        if len(random_pos) < N:
            random_pos.append(rng.uniform(-1.5, 1.5, size=3))
    
    random_pos = np.array(random_pos, dtype=float)
    random_pos -= np.mean(random_pos, axis=0)
    
    try:
        bh_pos2, bh_energy2 = basin_hopping(random_pos, rng, max_steps=30, step_scale=0.3)
        if check_validity(bh_pos2):
            if bh_energy2 < best_energy:
                best_pos = bh_pos2
                best_energy = bh_energy2
    except Exception:
        pass
    
    # Final Polish on the best result
    if best_pos is not None:
        try:
            if time.time() - START_TIME < DEADLINE - 5:
                final_pos, final_energy = local_minimize(best_pos, maxiter=500)
                if check_validity(final_pos) and final_energy < best_energy:
                    best_pos = final_pos
                    best_energy = final_energy
        except Exception:
            pass
            
    # Fallback
    if best_pos is None:
        # Absolute fallback: Random valid points
        rng_fallback = np.random.default_rng(123)
        pts = []
        for _ in range(N):
            while True:
                p = rng_fallback.uniform(-2, 2, size=3)
                if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                    pts.append(p)
                    break
        best_pos = np.array(pts)
        
    return best_pos

# EVOLVE-BLOCK-END
