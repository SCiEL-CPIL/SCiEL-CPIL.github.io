import numpy as np
import time
import scipy.optimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy_penalized(pos, rho, penalty_weight=1000.0, min_dist=0.5):
    """
    Compute Morse energy plus a penalty for violating minimum distance.
    pos: (n, 3) array
    """
    n = pos.shape[0]
    # Vectorized distance calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Morse energy: sum_{i<j} exp(rho(1-r))(exp(rho(1-r)) - 2)
    # We only need i<j
    i, j = np.triu_indices(n, k=1)
    r = dists[i, j]
    x = rho * (1.0 - r)
    exp_x = np.exp(x)
    e_morse = np.sum(exp_x * (exp_x - 2.0))
    
    # Penalty for distance violation
    violations = np.maximum(0.0, min_dist - r)
    e_penalty = penalty_weight * np.sum(violations ** 2)
    
    return e_morse + e_penalty

def _morse_gradient_penalized(pos, rho, penalty_weight=1000.0, min_dist=0.5):
    """
    Compute gradient of Morse energy plus penalty.
    """
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    # Get upper triangle indices
    i, j = np.triu_indices(n, k=1)
    r = dists[i, j]
    
    # Avoid division by zero (shouldn't happen if distinct)
    r_safe = np.where(r < 1e-10, 1e-10, r)
    
    # Morse force magnitude: dE/dr
    # E(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let u = rho(1-r). E = exp(u)(exp(u)-2). dE/dr = dE/du * du/dr
    # dE/du = exp(u)(exp(u)-2) + exp(u)*exp(u) = exp(u)(2exp(u)-2)
    # du/dr = -rho
    # dE/dr = -rho * exp(u) * (2exp(u) - 2) = -2*rho*exp(u)*(exp(u)-1)
    
    x = rho * (1.0 - r_safe)
    exp_x = np.exp(x)
    de_dr = -2.0 * rho * exp_x * (exp_x - 1.0)
    
    # Penalty force
    # P(r) = W * (max(0, min_dist - r))^2
    # dP/dr = 0 if r > min_dist
    # dP/dr = W * 2 * (min_dist - r) * (-1) = -2W(min_dist - r) if r < min_dist
    # Note: The penalty is added to energy, so we need the gradient of the penalty.
    # If r < min_dist, penalty = W*(min_dist - r)^2.
    # d(penalty)/dr = 2W(min_dist - r) * (-1) = -2W(min_dist - r).
    # Wait, if r < min_dist, min_dist - r is positive.
    # Force on atom i from atom j: -dE/dr * (r_vec / r)
    # Gradient of energy w.r.t pos[i]: sum_j -dE/dr_ij * (pos[i]-pos[j])/r_ij
    
    # Let's compute the vector forces
    # r_vec = pos[i] - pos[j]
    # grad_i = -de_dr * r_vec / r
    # grad_j = +de_dr * r_vec / r
    
    # We need to accumulate these.
    # For efficiency, we can use scatter_add or a loop for small N. N is up to 80.
    # Loop is fine for gradient if N is small, but vectorized is better.
    
    # Create a matrix of forces
    # F_ij vector = -de_dr_ij * (pos_i - pos_j) / r_ij
    # This is getting complex to vectorize fully without memory overhead.
    # Let's use the provided morse_energy_gradient for the Morse part and add penalty gradient.
    
    # Get Morse gradient
    e_morse, grad_morse = morse_energy_gradient(pos, rho)
    
    # Add penalty gradient
    # Penalty term: W * sum (max(0, min_dist - r_ij))^2
    # dP/dpos_i = sum_j 2W(max(0, min_dist - r_ij)) * (-1) * (pos_i - pos_j)/r_ij
    # Let v_ij = max(0, min_dist - r_ij)
    # grad_P_i = -2W * sum_j v_ij * (pos_i - pos_j)/r_ij
    
    grad_penalty = np.zeros_like(pos)
    for k in range(len(i)):
        idx_i = i[k]
        idx_j = j[k]
        r_k = r[k]
        if r_k < min_dist:
            v_k = min_dist - r_k
            r_vec = pos[idx_i] - pos[idx_j]
            # dP/dr = -2W v_k
            # dP/dpos_i = -dP/dr * r_vec/r = 2W v_k * r_vec/r
            # dP/dpos_j = -dP/dpos_i
            factor = 2.0 * penalty_weight * v_k / r_k
            grad_penalty[idx_i] += factor * r_vec
            grad_penalty[idx_j] -= factor * r_vec
            
    return e_morse + penalty_weight * np.sum(np.maximum(0, min_dist - r)**2), grad_morse + grad_penalty

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    rng = np.random.default_rng(42 + n + int(rho))
    
    best_energy = np.inf
    best_positions = None
    
    # 1. Generate diverse initial configurations
    # Strategy: Mix of random close pack (via rejection) and scaled lattice structures
    initial_configs = []
    
    # Generate 5 diverse starts
    num_starts = 5
    
    for s in range(num_starts):
        if time.time() > deadline - 100.0: break
        
        if s == 0:
            # Random in sphere
            radius = (n / (4/3 * np.pi * 0.6**3)) ** (1/3) # Approx packing
            radius = n ** (1/3) * 0.8
            pts = []
            while len(pts) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                        pts.append(p)
            if len(pts) == n:
                initial_configs.append(np.array(pts))
        else:
            # Perturbed FCC/BCC-like structure
            # Generate a grid of points, take n, center, add noise
            side = int(np.ceil(n ** (1.0/3.0))) + 2
            pts = []
            for i in range(-side, side+1):
                for j in range(-side, side+1):
                    for k in range(-side, side+1):
                        # Add some basis vectors for density
                        for b in [(0,0,0), (0.5,0.5,0)]:
                            p = np.array([i, j, k]) + np.array(b)
                            pts.append(p)
                        if len(pts) >= n:
                            break
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            pts = np.array(pts[:n])
            pts -= np.mean(pts, axis=0)
            # Add noise to break symmetry
            noise_scale = 0.2 * (s + 1)
            pts += rng.normal(0, noise_scale, size=pts.shape)
            
            # Quick check for min distance, if violated, try to fix or skip
            # We'll rely on the penalty in optimization to fix it, but ensure it's not catastrophic
            initial_configs.append(pts)
            
    if not initial_configs:
        # Fallback
        cfg = rng.uniform(-n**0.33, n**0.33, size=(n, 3))
        initial_configs.append(cfg)

    # 2. Local Minimization using L-BFGS-B with Penalty
    # We use a high penalty weight to enforce constraints
    penalty_weight = 1000.0
    
    for cfg in initial_configs:
        if time.time() > deadline - 50.0: break
        
        try:
            # Use L-BFGS-B
            # We define the function to return energy and gradient
            # Note: scipy minimize expects fun(x) -> (f, g) if jac is callable
            def fun_and_grad(x):
                pos = x.reshape(n, 3)
                e, g = _morse_gradient_penalized(pos, rho, penalty_weight)
                return e, g.reshape(-1)
            
            res = scipy.optimize.minimize(
                fun_and_grad, 
                cfg, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            pos_res = res.x.reshape(n, 3)
            # Verify constraint
            diff = pos_res[:, np.newaxis, :] - pos_res[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.499:
                # Constraint violation, try to repair or skip
                # For now, we still evaluate the energy but mark it as lower priority
                # Or we can run a few steps of projection
                pass
                
            # Calculate true energy (without penalty) for comparison
            # We can use the suite's relax to get a precise local min from this start
            # This ensures we are comparing apples to apples with the evaluator's expectation
            # But relax might be slow. Let's use the L-BFGS-B result if it's good.
            # Actually, the task says "Energies are recomputed exactly".
            # So we should return a configuration that is a local minimum of the *constrained* problem.
            # If L-BFGS-B converged well, pos_res is a good candidate.
            
            # Let's refine with the suite's relax to ensure high precision
            # relax is likely a robust local minimizer
            try:
                energy, positions = relax(pos_res, potential, rho)
                # Check if relax violated constraints (it shouldn't if it's good)
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=-1)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.5:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
            except Exception:
                # If relax fails, use the L-BFGS-B result
                e_raw, _ = _morse_gradient_penalized(pos_res, rho, 0.0) # Get raw energy
                if e_raw < best_energy:
                    best_energy = e_raw
                    best_positions = pos_res.copy()
                    
        except Exception:
            continue

    # 3. Basin Hopping / Simulated Annealing refinement
    if best_positions is None:
        # Fallback if all failed
        best_positions = initial_configs[0]
        best_energy = _morse_gradient_penalized(best_positions, rho, 0.0)[0]

    current_positions = best_positions
    current_energy = best_energy
    
    # Adaptive kick size
    kick_sigma = 0.2
    T = 0.1 # Initial temperature
    
    while time.time() < deadline - 10.0:
        # Perturb
        new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
        
        # Quick constraint check
        diff = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.45: # Allow some violation for optimization to fix
            # Try to repair by scaling or just reject
            # Rejection is faster
            continue
            
        # Local Minimization
        try:
            # Use L-BFGS-B again for speed
            def fun_and_grad_bh(x):
                pos = x.reshape(n, 3)
                e, g = _morse_gradient_penalized(pos, rho, penalty_weight)
                return e, g.reshape(-1)
            
            res = scipy.optimize.minimize(
                fun_and_grad_bh, 
                new_positions, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            pos_res = res.x.reshape(n, 3)
            
            # Refine with suite relax for precision
            energy, positions = relax(pos_res, potential, rho)
            
            # Check validity
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                continue
                
        except Exception:
            continue
            
        # Acceptance
        delta_e = energy - current_energy
        if delta_e < 0:
            current_energy = energy
            current_positions = positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                # Reduce temperature
                T *= 0.95
                kick_sigma *= 0.98
        else:
            # Metropolis
            if np.random.rand() < np.exp(-delta_e / T):
                current_energy = energy
                current_positions = positions
                
        # Cooldown
        T *= 0.99
        if T < 0.01:
            T = 0.01

    return best_positions
