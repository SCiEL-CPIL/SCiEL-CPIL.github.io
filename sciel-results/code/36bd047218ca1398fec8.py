"""
Strategy: Smooth Surrogate SA + Nelder-Mead Polish
1. Use a smooth soft-min energy function to guide Simulated Annealing.
   This avoids the discontinuities of hard-violation penalties.
2. Run multiple SA instances with different initial configurations.
3. Polish the best results using Nelder-Mead optimization on the exact radius function.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def _soft_energy(centers, beta):
    """
    Compute a smooth surrogate energy that approximates the negative of the minimum distance.
    Minimizing this energy maximizes the minimum distance (radius).
    """
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min for pairs
    # We want to maximize min(d_ij), so we minimize -softmin(d_ij)
    # softmin(d) = -1/beta * log(sum(exp(-beta * d)))
    # So we minimize 1/beta * log(sum(exp(-beta * d)))? 
    # Let's just minimize the sum of exp(-beta * d) directly? No, that's not quite right for min.
    # Standard soft-min: m(x) = -1/beta * log(sum(exp(-beta*x)))
    # We want to MAXIMIZE m(x). So we MINIMIZE -m(x) = 1/beta * log(sum(exp(-beta*x))).
    
    # To avoid overflow in exp, we can subtract max, but here we want min distance.
    # exp(-beta * d) is largest for smallest d.
    # log(sum(exp(-beta * d))) is dominated by the smallest d.
    # So minimizing log(sum(exp(-beta * d))) is equivalent to minimizing the smallest d (roughly).
    # Wait. If d is small, exp(-beta*d) is large. Sum is large. Log is large.
    # So minimizing this quantity pushes small distances up. Correct.
    
    # Wall energy
    # We want to maximize min(w_i).
    # Minimize log(sum(exp(-beta * w_i))).
    
    # Combine
    # Shift to avoid underflow/overflow issues if beta is large
    # Actually, for beta=50, d~0.1, exp(-5) is small.
    # Let's use a stable log-sum-exp.
    
    # Pairwise
    d_flat = dists[~np.eye(N, dtype=bool)]
    # We only need i<j to avoid double counting, but summing all i!=j is fine for the log-sum-exp shape.
    # Let's use i<j for efficiency and correctness of the "sum" magnitude (though log-sum-exp is shift-invariant, the scale matters for gradient magnitude if we were using it, but here we just use it as a scalar energy).
    # Actually, for SA, the absolute scale of energy matters for temperature.
    # Let's just sum over i<j.
    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    
    # Stable log-sum-exp for -beta * d
    # log(sum(exp(-beta * d)))
    # max_val = -beta * min(d_pairs)
    # log(sum(exp(-beta * d - max_val))) + max_val
    min_d = np.min(d_pairs)
    # If min_d is 0, we have a problem, but centers are distinct.
    # exp(-beta * (d - min_d))
    exp_terms = np.exp(-beta * (d_pairs - min_d))
    log_sum_exp_pairs = np.log(np.sum(exp_terms)) - beta * min_d
    
    # Wall
    min_w = np.min(wall_dists)
    exp_terms_w = np.exp(-beta * (wall_dists - min_w))
    log_sum_exp_walls = np.log(np.sum(exp_terms_w)) - beta * min_w
    
    # Energy to minimize
    # We want to maximize min(d) and min(w).
    # E = log_sum_exp_pairs + log_sum_exp_walls
    # As beta -> inf, this approximates min(d) + min(w) (with negative sign? No.)
    # Let's check:
    # If d_min is small, exp(-beta*d_min) is large. Log is large.
    # So E is large when min distance is small.
    # So minimizing E maximizes min distance. Correct.
    
    return log_sum_exp_pairs + log_sum_exp_walls

def _generate_initial_config(rng, type_id):
    """Generate different initial configurations."""
    if type_id == 0:
        # Jittered Grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        centers += rng.normal(0, 0.01, size=centers.shape)
    elif type_id == 1:
        # Random Uniform
        centers = rng.uniform(-0.45, 0.45, size=(N, 3))
    elif type_id == 2:
        # FCC-like
        # Generate FCC points and select best 50
        # FCC lattice points in [0,1]^3
        # (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        # Scale to [-0.5, 0.5]
        # Let's just use a dense grid and pick a subset that looks FCC-ish
        # Or just use a 4x4x4 grid which is BCC/FCC mixed.
        # Let's try a 5x5x5 grid and pick 50
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        # Randomly shuffle and take first 50? No, that's random.
        # Let's take a subset that is more spread out.
        # For now, just use the 4x4x4 grid (64 points) and pick 50 randomly from it?
        # Better: Use the 4x4x4 grid, but jitter more.
        centers = grid[:N].copy()
        centers += rng.normal(0, 0.05, size=centers.shape)
    else:
        # BCC-like
        # Just use another jittered grid with different seed
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        centers += rng.normal(0, 0.02, size=centers.shape)
        
    return np.clip(centers, -0.49, 0.49)

def _sa_optimize(centers, deadline, rng, beta=50.0):
    """Run Simulated Annealing to minimize soft energy."""
    current_centers = centers.copy()
    current_energy = _soft_energy(current_centers, beta)
    best_energy = current_energy
    best_centers = current_centers.copy()
    
    T = 0.05
    T_min = 1e-5
    cooling_rate = 0.9999
    move_scale = 0.05
    
    while time.time() < deadline:
        idx = rng.integers(0, N)
        delta = rng.normal(0, move_scale, size=3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        new_energy = _soft_energy(new_centers, beta)
        
        delta_E = new_energy - current_energy
        
        if delta_E < 0:
            current_centers = new_centers
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = new_centers.copy()
        else:
            if T > T_min:
                prob = np.exp(-delta_E / T)
                if rng.random() < prob:
                    current_centers = new_centers
                    current_energy = new_energy
        
        T *= cooling_rate
        move_scale *= 0.99999
        
        # Reheat if stuck?
        # Simple stagnation check
        if current_energy > best_energy + 0.1 and rng.random() < 0.01:
            T = 0.05
            move_scale = 0.05
            
    return best_centers

def _polish(centers, deadline, rng):
    """Polish the configuration using Nelder-Mead on the exact radius."""
    def objective(x):
        c = x.reshape(N, 3)
        # Ensure validity (clip)
        c = np.clip(c, -0.499, 0.499)
        r = _compute_radius(c)
        return -r
    
    # Use Nelder-Mead
    # maxiter should be reasonable
    # We have a time limit, so we can't run indefinitely.
    # Nelder-Mead is fast per iteration.
    # Let's run it until time runs out or convergence.
    
    # scipy minimize doesn't have a built-in time limit, so we wrap it.
    # We can use a callback to check time.
    
    start_time = time.time()
    
    def callback(xk):
        if time.time() > deadline:
            raise Exception("Time's up")
            
    try:
        res = minimize(objective, centers.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-5, 'fatol': 1e-7},
                       callback=callback)
        polished_centers = res.x.reshape(N, 3)
        polished_centers = np.clip(polished_centers, -0.499, 0.499)
    except Exception:
        polished_centers = centers
        
    return polished_centers

def construct():
    start_time = time.time()
    total_deadline = start_time + 115.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_radius = -1.0
    best_centers = None
    
    # Allocate time for SA and Polish
    # 70% SA, 30% Polish
    sa_deadline = start_time + (total_deadline - start_time) * 0.7
    
    # Multiple SA runs with different initializations
    num_sa_runs = 4
    sa_time_per_run = (sa_deadline - start_time) / num_sa_runs
    
    for i in range(num_sa_runs):
        if time.time() > sa_deadline:
            break
            
        run_deadline = start_time + (i + 1) * sa_time_per_run
        
        init_type = i % 4
        init_centers = _generate_initial_config(rng, init_type)
        
        # Run SA
        sa_result = _sa_optimize(init_centers, run_deadline, rng)
        
        # Evaluate
        r = _compute_radius(sa_result)
        if r > best_radius:
            best_radius = r
            best_centers = sa_result.copy()
            
        # Store top candidates for polishing
        # We'll keep a list of top 3 candidates
        
    # Polish the best candidate
    if best_centers is not None:
        polish_deadline = total_deadline
        polished = _polish(best_centers, polish_deadline, rng)
        r_polished = _compute_radius(polished)
        
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished
            
    # Final check
    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        
    return best_centers

# EVOLVE-BLOCK-END
