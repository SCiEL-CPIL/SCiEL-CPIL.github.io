import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

# Constants
NUM_POINTS = 100
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
START_TIME = time.time()
UPPER_TRI_IDX = np.triu_indices(NUM_POINTS, k=1)

def get_time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def normalize_vectors(mat):
    """Normalize rows of a matrix to unit length."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-12, 1.0, norms)
    return mat / norms

def compute_energy_and_gradient(positions):
    """
    Calculate the logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    """
    # Compute pairwise dot products
    # d_ij^2 = 2 - 2 * dot(i,j)
    dots = positions @ positions.T
    dist_sq = 2.0 - 2.0 * dots
    # Ensure diagonal is 1 (distance 0), but we ignore diagonal in energy
    np.fill_diagonal(dist_sq, 1.0)
    # Clip to prevent log(0) or negative values due to floating point errors
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy: sum of -log(d_ij) for i<j
    # d_ij = sqrt(dist_sq_ij)
    # log(1/d_ij) = -log(d_ij) = -0.5 * log(dist_sq_ij)
    e_terms = -0.5 * np.log(dist_sq[UPPER_TRI_IDX])
    total_energy = np.sum(e_terms)
    
    # Gradient Calculation
    # grad_i = sum_{j!=i} (1/d_ij^2) * (x_j - x_i)  ... Wait, let's derive carefully.
    # E = - sum_{i<j} log(d_ij)
    # dE/dx_i = - sum_{j!=i} (1/d_ij) * (x_i - x_j)/d_ij  (for i<j terms, and symmetric for j<i)
    # Actually, let's use the known formula for Riesz s-energy.
    # For s=0 (log), the force is 1/r^2? No.
    # d/dr log(1/r) = -1/r.
    # Force on i from j: - dE/dr * dr/dx_i = -(-1/d) * (x_i-x_j)/d = (x_i-x_j)/d^2.
    # Wait, E = sum log(1/d). dE/dd = -1/d.
    # grad E w.r.t x_i = sum_j (-1/d_ij) * grad_xi d_ij.
    # d_ij = ||x_i - x_j||. grad_xi d_ij = (x_i - x_j)/d_ij.
    # So term is (-1/d) * (x_i-x_j)/d = -(x_i-x_j)/d^2 = (x_j-x_i)/d^2.
    
    # Vectorized Gradient
    # W_ij = 1 / d_ij^2 for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_j - x_i) = sum_j W_ij * x_j - (sum_j W_ij) * x_i
    # sum_j W_ij * x_j = W @ positions
    # sum_j W_ij = W.sum(axis=1)
    grad = W @ positions - W.sum(axis=1, keepdims=True) * positions
    
    # Project gradient onto tangent space of the sphere
    # grad_tangent = grad - (grad . x_i) * x_i
    proj = np.sum(grad * positions, axis=1, keepdims=True)
    grad = grad - proj * positions
    
    return float(total_energy), grad

def calculate_exact_energy(positions):
    """High-precision energy calculation using math.fsum."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    d_upper = dists[UPPER_TRI_IDX]
    if np.any(d_upper < 1e-9):
        return float('inf')
    terms = [math.log(1.0 / d) for d in d_upper]
    return math.fsum(terms)

def init_fibonacci(offset=0.5):
    """Generate points using Fibonacci spiral."""
    i = np.arange(NUM_POINTS) + offset
    z = 1.0 - 2.0 * i / NUM_POINTS
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def init_sobol(seed=0):
    """Generate points using Sobol sequence."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Take power of 2 samples for better distribution, then truncate
    m = 1 << (NUM_POINTS.bit_length())
    samples = sampler.random(m)
    pts = samples[:NUM_POINTS]
    return normalize_vectors(pts)

def init_random_perturbed_fibonacci(seed=0, amp=0.05):
    """Generate points by perturbing the Fibonacci spiral."""
    base = init_fibonacci(offset=0.5)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((NUM_POINTS, 3))
    perturbed = base + amp * noise
    return normalize_vectors(perturbed)

def run_gradient_flow(positions, time_budget, lr_initial=0.05, lr_final=0.001):
    """
    Custom gradient flow with momentum and adaptive learning rate.
    This is faster than L-BFGS for the initial phase.
    """
    p = positions.copy()
    v = np.zeros_like(p) # Momentum
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    best_p = p.copy()
    best_e = calculate_exact_energy(p)
    current_e = best_e
    
    start_t = time.time()
    steps = 0
    
    while time.time() - start_t < time_budget and time.time() < TIME_LIMIT - SAFETY_MARGIN:
        # Adaptive learning rate: decay over time
        progress = (time.time() - start_t) / time_budget
        if progress > 1.0: progress = 1.0
        lr = lr_initial * (1.0 - progress) + lr_final * progress
        
        e, g = compute_energy_and_gradient(p)
        
        # Update momentum
        v = beta1 * v + (1 - beta1) * g
        
        # Update position
        new_p = p - lr * v
        new_p = normalize_vectors(new_p)
        
        new_e, _ = compute_energy_and_gradient(new_p)
        
        if new_e < current_e:
            p = new_p
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_p = p.copy()
        else:
            # If energy increased, reduce learning rate slightly and retry
            lr *= 0.5
            v *= 0.5
            new_p = p - lr * v
            new_p = normalize_vectors(new_p)
            new_e, _ = compute_energy_and_gradient(new_p)
            if new_e < current_e:
                p = new_p
                current_e = new_e
                if current_e < best_e:
                    best_e = current_e
                    best_p = p.copy()
            else:
                # Stagnation, break early if no improvement for a while? 
                # Simple heuristic: just continue, but reduce lr more
                lr *= 0.5
                v *= 0.5
        
        steps += 1
    
    return best_p, best_e

def refine_with_lbfgs(positions, time_budget):
    """
    High-precision refinement using L-BFGS-B.
    """
    p = positions.copy()
    best_e = calculate_exact_energy(p)
    
    def obj_and_grad(flat):
        p_flat = flat.reshape(NUM_POINTS, 3)
        p_norm = normalize_vectors(p_flat)
        e, g = compute_energy_and_gradient(p_norm)
        return e, g.reshape(-1)
    
    start_t = time.time()
    x0 = p.reshape(-1)
    
    # Try a few iterations of L-BFGS with increasing strictness
    max_iter = 200
    ftol = 1e-12
    gtol = 1e-10
    
    try:
        res = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': 1000,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 20
            }
        )
        new_p = normalize_vectors(res.x.reshape(NUM_POINTS, 3))
        new_e = calculate_exact_energy(new_p)
        
        if new_e < best_e:
            p = new_p
            best_e = new_e
            
    except Exception:
        pass
    
    # If time remains, do a second, tighter pass
    if time.time() - start_t < time_budget - 2.0 and time.time() < TIME_LIMIT - SAFETY_MARGIN:
        try:
            x0 = p.reshape(-1)
            res2 = minimize(
                obj_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'maxfun': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 30
                }
            )
            new_p = normalize_vectors(res2.x.reshape(NUM_POINTS, 3))
            new_e = calculate_exact_energy(new_p)
            
            if new_e < best_e:
                p = new_p
                best_e = new_e
        except Exception:
            pass
            
    return p, best_e

def construct():
    """
    Main entry point.
    1. Generate diverse initializations.
    2. Run gradient flow for each to find local minima.
    3. Refine the best few candidates with L-BFGS.
    """
    total_deadline = TIME_LIMIT - SAFETY_MARGIN
    start_time = time.time()
    
    # 1. Generate Initializations
    # Mix of deterministic and random to cover basin
    inits = []
    
    # Fibonacci variants
    for off in [0.5, 0.1, 0.9]:
        inits.append(init_fibonacci(off))
        
    # Sobol variants
    for s in range(3):
        inits.append(init_sobol(seed=s))
        
    # Perturbed Fibonacci (Random Jitter)
    for s in range(5):
        inits.append(init_random_perturbed_fibonacci(seed=s, amp=0.02 + 0.01*s))
        
    # Pure Random
    rng = np.random.default_rng(123)
    for s in range(5):
        r = rng.standard_normal((NUM_POINTS, 3))
        inits.append(normalize_vectors(r))

    best_candidate = None
    best_energy = float('inf')
    local_results = []
    
    # 2. Gradient Flow Phase (Coarse Optimization)
    # Allocate time per init based on remaining time
    # We have ~135s total. Let's spend ~80s on this phase.
    phase1_budget = 80.0
    time_per_init = phase1_budget / len(inits)
    
    for idx, init_p in enumerate(inits):
        if time.time() > total_deadline - 50.0: # Reserve 50s for refinement
            break
            
        # Dynamic time allocation: give less time to later inits if we are ahead
        current_budget = min(time_per_init, (total_deadline - 50.0) - time.time())
        if current_budget < 2.0:
            break
            
        p_opt, e_opt = run_gradient_flow(init_p, time_budget=current_budget)
        local_results.append((e_opt, p_opt))
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_candidate = p_opt

    # 3. Refinement Phase (L-BFGS)
    # Sort local results by energy
    local_results.sort(key=lambda x: x[0])
    
    # Refine the top 5 candidates
    num_to_refine = min(5, len(local_results))
    phase2_budget = (TIME_LIMIT - SAFETY_MARGIN) - time.time()
    
    if num_to_refine > 0 and phase2_budget > 5.0:
        time_per_refine = phase2_budget / num_to_refine
        
        for i in range(num_to_refine):
            if time.time() > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            e_init, p_init = local_results[i]
            # If this init is already worse than our current best by a lot, skip?
            # But L-BFGS might improve it significantly. Let's try if time permits.
            
            # Don't spend more time on a bad candidate than on a good one?
            # Just use equal time.
            
            p_ref, e_ref = refine_with_lbfgs(p_init, time_budget=time_per_refine)
            
            if e_ref < best_energy:
                best_energy = e_ref
                best_candidate = p_ref

    # 4. Final Polish on the absolute best
    if best_candidate is not None and time.time() < TIME_LIMIT - SAFETY_MARGIN:
        final_budget = (TIME_LIMIT - SAFETY_MARGIN) - time.time()
        if final_budget > 2.0:
            p_final, e_final = refine_with_lbfgs(best_candidate, time_budget=final_budget)
            if e_final < best_energy:
                best_energy = e_final
                best_candidate = p_final

    # Fallback
    if best_candidate is None:
        best_candidate = init_fibonacci(0.5)
        
    return best_candidate
