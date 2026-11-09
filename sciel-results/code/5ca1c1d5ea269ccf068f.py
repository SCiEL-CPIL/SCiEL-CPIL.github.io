"""
Strategy: Parallel Basin Hopping with Polytetrahedral Seeding and L-BFGS-B Local Minimization.
The Dzugutov potential is designed to favor PT order (icosahedral shells).
Instead of relying on a single, potentially slow or trapped `relax` call from a few starts,
we implement a robust search:
1. Generate diverse initial structures (PT shells, random compact, perturbed PT).
2. Use a parallel pool (up to 4 threads) to run independent basin hopping chains.
3. Each chain uses L-BFGS-B (via SciPy) for fast, accurate local minimization,
   combined with random perturbations to escape local minima.
4. Keep the global best energy and positions found across all threads.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 320.0  # Safety margin before 330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy_gradient(positions):
    """
    Compute energy and gradient for the Dzugutov potential.
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    n = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    # Precompute pairwise distances and differences
    # For n up to 60, O(N^2) is fine.
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            
            if r < 1e-10:
                continue
                
            # Force direction
            r_vec = diff / r
            
            force_mag = 0.0
            
            if r < a:
                # V(r) = A * (r^-m - B) * exp(c / (r - a))
                # dV/dr = A * [ -m * r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2) ]
                # dV/dr = A * exp(c/(r-a)) * [ -m * r^(-m-1) - c * (r^-m - B) / (r-a)^2 ]
                
                term1 = -m * r ** (-m - 1)
                term2 = -c * (r ** (-m) - B) / ((r - a) ** 2)
                force_mag = A * np.exp(c / (r - a)) * (term1 + term2)
                
            elif r < b:
                # V(r) = B * exp(d / (r - b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                
                force_mag = B * np.exp(d / (r - b)) * (-d / ((r - b) ** 2))
            
            else:
                # Force is zero for r >= b
                # But wait, the potential definition says:
                # V(r) = A(...) [r < a] + B(...) [r < b]
                # Is it piecewise or sum?
                # "V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]"
                # Usually this means:
                # If r < a: V = A(...)
                # If a <= r < b: V = B(...)
                # If r >= b: V = 0
                # Let's assume piecewise based on standard Dzugutov definition.
                # My code above calculates force_mag for r < a in the first if.
                # If r is between a and b, it falls into elif.
                # If r >= b, force_mag remains 0.
                # This matches the piecewise interpretation.
                pass

            # Update energy
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
            
            # Update gradient
            # Force on i is +F * r_vec, on j is -F * r_vec
            # Gradient of E wrt pos_i is -Force_on_i = -F * r_vec
            # Wait, Force = -dV/dr * r_vec.
            # My force_mag is dV/dr.
            # So Force vector on i is - (dV/dr) * (pos_j - pos_i)/r = - (dV/dr) * (-r_vec) = (dV/dr) * r_vec?
            # Let's be careful.
            # diff = pos_i - pos_j. r_vec = diff/r.
            # Potential V(r).
            # dE/dpos_i = dV/dr * dr/dpos_i = dV/dr * (pos_i - pos_j)/r = dV/dr * r_vec.
            # dE/dpos_j = dV/dr * dr/dpos_j = dV/dr * (pos_j - pos_i)/r = -dV/dr * r_vec.
            
            gradient[i] += force_mag * r_vec
            gradient[j] -= force_mag * r_vec

    return energy, gradient

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)] # Center
    
    # Shell 1: 12 atoms
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # Shell 2: Add remaining atoms on a sphere with radius r2
    # For PT order, the next shell is often larger.
    # Let's use r2 = 1.9 * 1.7 ~ 3.23
    r2 = 1.9 * 1.7
    num_outer = n_atoms - 13
    if num_outer > 0:
        # Generate num_outer points on sphere r2
        # Use random points for simplicity, or try to distribute
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.1):
    """
    Generate a random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        # Fallback: simple grid or random without strict check
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def _local_minimize(positions, rng):
    """
    Perform local minimization using L-BFGS-B.
    """
    # Add small noise to avoid symmetry locks?
    # positions = positions + rng.normal(0, 0.01, positions.shape)
    
    res = minimize(
        lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[0],
        positions.flatten(),
        jac=lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100, 'maxls': 100, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    if res.success or res.fun < np.inf:
        return res.fun, res.x.reshape(-1, 3)
    else:
        # Return the best point found
        return res.fun, res.x.reshape(-1, 3)

def _basin_hopping_chain(initial_positions, rng, time_limit):
    """
    Run a basin hopping chain.
    """
    current_energy, current_positions = _local_minimize(initial_positions, rng)
    best_energy, best_positions = current_energy, current_positions.copy()
    
    start_time = time.time()
    step_size = 0.5
    
    while time.time() - start_time < time_limit:
        # Perturb
        perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
        
        # Local minimize
        try:
            new_energy, new_positions = _local_minimize(perturbed, rng)
        except Exception:
            continue
            
        # Acceptance criterion (Metropolis-like)
        if new_energy < current_energy or (rng.random() < np.exp(-(new_energy - current_energy) / 0.5)):
            current_energy = new_energy
            current_positions = new_positions
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Occasionally increase step size
        if rng.random() < 0.05:
            step_size *= 1.2
        elif rng.random() < 0.05:
            step_size *= 0.8
            step_size = max(0.1, step_size)

    return best_energy, best_positions

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    # Time management
    start_time = time.time()
    # We have ~50s per instance if sequential, but we use threads.
    # Let's aim for ~45s per instance to be safe with 6 instances.
    time_budget = 45.0
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact
    starts.append(_random_compact(size, rng))
    
    # 3. Random Compact (different seed)
    starts.append(_random_compact(size, rng))
    
    # 4. Perturbed PT
    if starts:
        perturbed_pt = starts[0] + rng.normal(0, 0.2, starts[0].shape)
        starts.append(perturbed_pt)

    best_energy = np.inf
    best_positions = None
    lock = threading.Lock()
    
    def worker(start_pos):
        nonlocal best_energy, best_positions
        try:
            # Run basin hopping for a short time
            # Divide time budget among threads?
            # 4 threads, so ~10s each? Or just run until global time limit?
            # Let's run for 8 seconds per thread.
            local_energy, local_positions = _basin_hopping_chain(start_pos, rng, 8.0)
            
            with lock:
                if local_energy < best_energy:
                    best_energy = local_energy
                    best_positions = local_positions.copy()
        except Exception:
            pass

    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, start) for start in starts]
        
        # Wait for completion or global timeout
        # We need to be careful with timeouts in ThreadPoolExecutor
        # We will just wait for them, but they have internal time limits.
        
        for future in as_completed(futures):
            if time.time() - start_time > time_budget:
                # Cancel pending? Not easy. Just break.
                break
            try:
                future.result()
            except Exception:
                pass

    # If we have time, do more random restarts
    while time.time() - start_time < time_budget * 0.9:
        # Generate a new random start
        new_start = _random_compact(size, rng)
        
        # Quick local minimization
        try:
            energy, positions = _local_minimize(new_start, rng)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        return starts[-1]
        
    return best_positions
# EVOLVE-BLOCK-END
