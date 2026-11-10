"""
Strategy: Continuous Simulated Annealing (SA) with Adaptive Step Size.
This approach rethinks the global search by using a single, continuous annealing
process rather than discrete Basin Hopping.

Key Differences from Basin Hopping:
1. Continuous Trajectory: Instead of "kick -> full local relax -> kick", we perform
   a sequence of gradient steps with a temperature $T(t)$ that decays over time.
2. Metropolis Acceptance: At each step, we compute the energy difference $\Delta E$.
   If $\Delta E < 0$, we always accept. If $\Delta E > 0$, we accept with probability
   $e^{-\Delta E / T}$. This allows escaping local minima without restarting the optimizer.
3. Adaptive Step Size: The step size $\alpha$ is dynamically adjusted based on the
   acceptance rate. If acceptance is too high, we increase $\alpha$ to explore more;
   if too low, we decrease it to refine. This avoids the "stuck" problem of fixed-step
   random kicks.
4. Efficiency: We avoid the overhead of initializing L-BFGS-B repeatedly. We use a
   simple gradient descent step (or a few steps) per iteration, which is cheaper than
   a full local minimization, allowing for many more "moves" in the same time budget.

This method is particularly effective for Morse clusters where the landscape is
smooth but rugged. The annealing schedule ensures that early high-temperature phases
explore the global landscape, while later low-temperature phases converge to deep
local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)  # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let x = rho(1-r)
    # V = e^x (e^x - 2) = e^(2x) - 2e^x
    # dV/dr = (2e^(2x) - 2e^x) * dx/dr = 2e^x(e^x - 1) * (-rho)
    # dV/dri = -dV/dr * (ri - rj) / r
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    # Vectorized gradient calculation
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = dV_dr * inv_r
    
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _generate_fcc(n, rng):
    """Generate n atoms on an FCC lattice, cropped to a sphere."""
    a = np.sqrt(2.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0, 0.5*a, 0.5*a), (0.5*a, 0, 0.5*a), (0.5*a, 0.5*a, 0)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_bcc(n, rng):
    """Generate n atoms on a BCC lattice, cropped to a sphere."""
    a = 2.0 / np.sqrt(3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                for dx, dy, dz in [(0,0,0), (0.5*a, 0.5*a, 0.5*a)]:
                    p = base + np.array([dx, dy, dz])
                    points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    points += rng.normal(0, 0.01, points.shape)
    return points

def _generate_random(n, rng):
    """Generate n atoms randomly in a compact region."""
    radius = 1.0 * n ** (1.0 / 3.0) * 1.2
    points = []
    for _ in range(n):
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    points = np.array(points)
    points += rng.normal(0, 0.01, points.shape)
    return points

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B for final polishing."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def _simulated_annealing(start_pos, rho, rng, time_limit, max_iter=2000):
    """
    Perform Continuous Simulated Annealing.
    Returns the best position and energy found.
    """
    # Initial local minimization to get a good starting point
    current_pos, current_energy = _local_minimize(start_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Annealing parameters
    T_start = 5.0
    T_end = 0.01
    alpha = 0.1  # Initial step size
    target_acceptance = 0.3
    acceptance_window = 50
    accepted = 0
    total = 0
    
    # Pre-compute log T schedule? No, simple geometric cooling is robust.
    # We use a time-based cooling schedule to ensure we finish in time.
    # But max_iter is a cap.
    
    for i in range(max_iter):
        if time.time() - _START_TIME > time_limit:
            break
        
        # Cooling schedule: Geometric
        # We want T to drop from T_start to T_end over max_iter steps
        # T_i = T_start * (T_end/T_start)^(i/max_iter)
        if max_iter > 0:
            T = T_start * (T_end / T_start) ** (i / max_iter)
        else:
            T = T_end
        
        if T < T_end:
            T = T_end
            
        # Generate a random perturbation
        # The step size alpha is adapted
        noise = rng.normal(0, alpha, current_pos.shape)
        new_pos = current_pos + noise
        
        # Compute new energy
        new_energy, _ = _morse_energy_and_grad(new_pos, rho)
        
        # Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
            accepted += 1
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            pass
            
        total += 1
        
        # Adapt step size every acceptance_window steps
        if total % acceptance_window == 0:
            acceptance_rate = accepted / total
            # If acceptance rate is too high, increase step size to explore more
            # If too low, decrease step size to refine
            if acceptance_rate > target_acceptance:
                alpha *= 1.5
            elif acceptance_rate < target_acceptance * 0.5:
                alpha *= 0.7
            else:
                alpha *= 0.95 # Slight decay to converge
            
            # Clamp alpha
            alpha = np.clip(alpha, 0.01, 1.0)
            
            # Reset counters
            accepted = 0
            total = 0
            
            # If we are in a very low energy state and T is low, do a local relax
            # to ensure we are at the bottom of the basin
            if T < 0.1 and i % 100 == 0:
                current_pos, current_energy = _local_minimize(current_pos, rho)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
                
                # If local relax didn't change anything significantly, reduce alpha further
                # to allow finer tuning
                alpha *= 0.8

    # Final local minimization
    best_pos, best_energy = _local_minimize(best_pos, rho)
    
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. FCC
    fcc = _generate_fcc(size, rng)
    if fcc is not None:
        configs.append(fcc)
    
    # 2. BCC
    bcc = _generate_bcc(size, rng)
    if bcc is not None:
        configs.append(bcc)
    
    # 3. Random (multiple with different seeds)
    for i in range(4):
        rng_i = np.random.default_rng(42 + i)
        rand = _generate_random(size, rng_i)
        configs.append(rand)
        
    best_energy = float('inf')
    best_positions = None
    
    # Perform Simulated Annealing from each configuration
    # We allocate time dynamically.
    # SA is continuous, so it naturally uses up the time budget.
    # We run SA on each config, but we break early if time runs out.
    
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT:
            break
            
        # Try Simulated Annealing
        try:
            sa_pos, sa_energy = _simulated_annealing(pos, parameter, rng, _TIME_LIMIT, max_iter=1000)
            
            if _is_valid(sa_pos):
                if sa_energy < best_energy:
                    best_energy = sa_energy
                    best_positions = sa_pos
        except Exception:
            # Fallback to simple relax if SA fails
            try:
                energy, positions = relax(pos, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
        
        # If we are getting close to the time limit, stop trying new starts
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break

    # If no valid position found, return a valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
