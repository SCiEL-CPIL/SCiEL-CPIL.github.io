"""LJ38 cluster optimization using Basin Hopping in Cartesian space."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 145.0  # Safety margin before 150s kill
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    # positions: (N, 3)
    # Compute pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    FCC lattice constant a = 1/sqrt(2) for sigma=1.
    We generate points in the FCC lattice and select the 38 closest to origin.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)

def make_icosahedral():
    """
    Create an icosahedral cluster (13 atoms) plus a shell.
    For N=38, a common structure is 13 (inner) + 12 (icosahedron vertices) + 13 (next shell?) 
    Actually, the icosahedral funnel for N=38 usually involves a 13-atom icosahedron core 
    and a 25-atom outer shell? No, 13+12=25. 
    Let's just generate a reasonable icosahedral-like seed.
    Core 13: 1 center + 12 vertices of icosahedron.
    Shell 1: 12 vertices of a larger icosahedron?
    Let's use a simpler heuristic: random points near an icosahedral template.
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2.0
    # Icosahedron vertices
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    # Normalize to radius ~1.2 (typical LJ shell)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * 1.2
    
    # Add center
    center = np.array([[0, 0, 0]])
    
    # We need 38 atoms. 13 is the core.
    # Let's add a second shell of 12 at radius ~2.0
    shell2 = verts * (2.0 / 1.2)
    
    # And a third shell? 13 + 12 = 25. Need 13 more.
    # Let's just add random points in a shell of radius 2.5
    rng = np.random.default_rng(42)
    remaining = 38 - 25
    # Generate random points on sphere
    z = rng.uniform(-1, 1, remaining)
    r = np.sqrt(1 - z**2)
    phi_angle = rng.uniform(0, 2*np.pi, remaining)
    shell3 = np.zeros((remaining, 3))
    shell3[:, 0] = r * np.cos(phi_angle) * 2.5
    shell3[:, 1] = r * np.sin(phi_angle) * 2.5
    shell3[:, 2] = z * 2.5
    
    positions = np.vstack([center, verts, shell2, shell3])
    return positions

def make_random_cluster():
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng()
    positions = np.zeros((N, 3))
    # Start with first atom at origin
    positions[0] = 0
    
    # Add atoms one by one, ensuring distance > 0.8
    min_dist = 0.8
    for i in range(1, N):
        while True:
            p = rng.uniform(-3, 3, size=3)
            # Check distance to existing atoms
            diff = positions[:i] - p
            r2 = np.sum(diff*diff, axis=1)
            if np.min(r2) > min_dist*min_dist:
                positions[i] = p
                break
    return positions

def relax_ljfg(positions):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        # The relax function might expect a specific format or handle constraints.
        # Assuming it takes (N,3) and returns (N,3)
        relaxed = relax(positions)
        if relaxed is not None and len(relaxed) == N:
            return relaxed
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to local L-BFGS
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
        
    try:
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 50, 'gtol': 1e-5})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, T0=1.0, T1=0.1, n_iter=100, step_size=0.5, rng=None):
    """
    Basin hopping algorithm.
    positions: initial (N,3) array
    """
    if rng is None:
        rng = np.random.default_rng()
        
    current_pos = positions.copy()
    current_energy = lj_energy_vectorized(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T = T0
    cooling_rate = (T1 / T0) ** (1.0 / max(1, n_iter))
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 5:
            break
            
        # 1. Local minimization
        min_pos = relax_ljfg(current_pos)
        min_energy = lj_energy_vectorized(min_pos)
        
        if min_energy < best_energy:
            best_energy = min_energy
            best_pos = min_pos.copy()
            
        # 2. Random perturbation
        noise = rng.normal(0, step_size, size=min_pos.shape)
        perturbed_pos = min_pos + noise
        
        # Ensure validity (min distance)
        if not check_validity(perturbed_pos):
            # Try to fix or reject
            # For simplicity, reject if invalid
            perturbed_energy = 1e20
        else:
            perturbed_energy = lj_energy_vectorized(perturbed_pos)
            
        # 3. Metropolis acceptance
        delta_E = perturbed_energy - min_energy
        if delta_E < 0:
            accept = True
        else:
            if T > 0:
                prob = math.exp(-delta_E / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            current_pos = perturbed_pos
        else:
            # Keep current position (the minimized one)
            current_pos = min_pos
            
        T *= cooling_rate
        
    return best_pos, best_energy

def construct():
    """Main entry point: Multi-start Basin Hopping."""
    global START_TIME
    START_TIME = time.time()
    
    # List of seeds
    seeds = []
    
    # 1. FCC Truncated Octahedron (Known global minimum candidate)
    fcc_pos = make_fcc_truncated_octahedron()
    if check_validity(fcc_pos):
        seeds.append(fcc_pos)
        
    # 2. Icosahedral (Local minimum trap, but good for comparison)
    ico_pos = make_icosahedral()
    if check_validity(ico_pos):
        seeds.append(ico_pos)
        
    # 3. Random clusters
    rng_global = np.random.default_rng(123)
    for _ in range(3):
        rand_pos = make_random_cluster()
        if check_validity(rand_pos):
            seeds.append(rand_pos)
            
    best_pos = None
    best_energy = float('inf')
    
    # Iterate through seeds
    # We have ~150s. Relaxation takes maybe 0.1-1s. Basin hopping takes the rest.
    # Let's allocate time per seed.
    n_seeds = len(seeds)
    time_per_seed = (DEADLINE - 10) / n_seeds
    
    for i, seed in enumerate(seeds):
        if time.time() - START_TIME > DEADLINE - 5:
            break
            
        # First, relax the seed to get into a basin
        relaxed_seed = relax_ljfg(seed)
        if not check_validity(relaxed_seed):
            relaxed_seed = seed
            
        seed_energy = lj_energy_vectorized(relaxed_seed)
        
        if seed_energy < best_energy:
            best_energy = seed_energy
            best_pos = relaxed_seed.copy()
            
        # Run basin hopping
        # Adjust iterations based on remaining time
        # Assume ~10ms per BH iteration (relax + energy)
        # If 10s left, we can do ~1000 iterations? No, relax is expensive.
        # Let's do a fixed number of iterations, but check time.
        n_iter = 50
        # For the first seed (FCC), we might want to do more if it's already good?
        # Actually, if FCC is the global min, we just need to confirm it.
        # But BH can escape if T is high.
        
        # Let's run BH for a short duration to see if we can improve
        # Since FCC is likely the global min, improving it is hard.
        # Improving random seeds is easier.
        
        bh_pos, bh_energy = basin_hopping(
            relaxed_seed, 
            T0=2.0, 
            T1=0.5, 
            n_iter=n_iter, 
            step_size=0.3, 
            rng=np.random.default_rng(i)
        )
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
            
    # Final relaxation of the best found
    if best_pos is not None:
        final_relaxed = relax_ljfg(best_pos)
        if check_validity(final_relaxed):
            final_energy = lj_energy_vectorized(final_relaxed)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_relaxed
    
    # Fallback if something went wrong
    if best_pos is None or not check_validity(best_pos):
        best_pos = make_fcc_truncated_octahedron()
        # Relax it one last time
        try:
            best_pos = relax_ljfg(best_pos)
        except:
            pass
    
    return best_pos

# EVOLVE-BLOCK-END
