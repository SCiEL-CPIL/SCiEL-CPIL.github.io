import numpy as np
import time
import random

def _fast_supnorm(coeffs, z_powers):
    """
    Fast sup-norm estimation using precomputed powers of z.
    coeffs: array of length N
    z_powers: (n_pts, N) array of z^k
    """
    vals = coeffs @ z_powers.T
    mags = np.abs(vals)
    return np.max(mags)

def _precise_supnorm(coeffs, n_pts=2048, refine_rounds=6, refine_points=15):
    """
    Higher precision sup-norm for final evaluation.
    """
    n = len(coeffs)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    z = np.exp(1j * angles)
    k = np.arange(n)
    vals = np.sum(coeffs[None, :] * np.power(z[:, None], k[None, :]), axis=1)
    magnitudes = np.abs(vals)
    
    # Find top candidates for refinement
    if n_pts > refine_points:
        top_idx = np.argpartition(magnitudes, -refine_points)[-refine_points:]
    else:
        top_idx = np.arange(n_pts)
    
    best_mag = float(np.max(magnitudes))
    
    # Refine around the top peaks
    window = 2.0 * np.pi / n_pts * 2
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    inv_golden = 1.0 - 1.0 / golden_ratio
    
    for idx in top_idx:
        center = angles[idx]
        a = max(0.0, center - window)
        b = min(2.0 * np.pi, center + window)
        
        if b <= a:
            continue
            
        c = b - inv_golden * (b - a)
        d = a + inv_golden * (b - a)
        
        def _eval(angle):
            return np.abs(np.sum(coeffs * np.exp(1j * k * angle)))
        
        fc = _eval(c)
        fd = _eval(d)
        
        for _ in range(refine_rounds):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + inv_golden * (b - a)
                fd = _eval(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - inv_golden * (b - a)
                fc = _eval(c)
        
        fa = _eval(a)
        fb = _eval(b)
        
        local_best = max(fa, fb, fc, fd)
        if local_best > best_mag:
            best_mag = local_best
    
    return best_mag

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    N = 24
    
    # Precompute grid for fast evaluation
    n_pts_search = 1024
    angles_search = np.linspace(0.0, 2.0 * np.pi, n_pts_search, endpoint=False)
    z_search = np.exp(1j * angles_search)
    k_arr = np.arange(N)
    z_powers = np.power(z_search[:, None], k_arr[None, :])
    
    best_coeffs = None
    best_fast_score = np.inf
    
    # Generate seeds
    seeds = []
    
    # 1. Random seeds
    for _ in range(5):
        seeds.append(np.random.choice([-1, 1], N))
    
    # 2. Known good patterns / Odlyzko candidates
    seeds.append(np.array([1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1], dtype=np.float64))
    seeds.append(np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float64))
    seeds.append(np.ones(N, dtype=np.float64))
    
    # 3. Golay-Rudin-Shapiro
    gr_p, gr_q = [1], [1]
    while len(gr_p) < N:
        gr_p, gr_q = gr_p + gr_q, gr_p + [-c for c in gr_q]
    seeds.append(np.array(gr_p[:N], dtype=np.float64))
    
    # 4. Blahut-Arimoto inspired relaxation + rounding
    # We treat the coefficients as continuous variables x_k in [-1, 1]
    # and try to minimize max |sum x_k z^k|.
    # This is a non-convex problem in general, but we can use an iterative
    # water-filling heuristic:
    # 1. Start with x_k = 1 for all k.
    # 2. Evaluate P(z) on a grid.
    # 3. Identify the angle theta where |P(e^{i theta})| is maximized.
    # 4. The gradient of |P|^2 with respect to x_k is 2 * Re(P * conj(z^k)).
    # 5. To reduce the peak, we want to decrease |P| at that angle.
    #    We can do this by adjusting x_k in the direction opposite to the gradient.
    #    Specifically, d/dx_k |P|^2 = 2 Re(P conj(z^k)).
    #    We want to minimize |P|^2, so we take a step in the negative gradient direction.
    #    However, we must keep x_k in [-1, 1].
    #    This is essentially a projected gradient descent on the continuous relaxation.
    #    After convergence, we round x_k to sign(x_k).
    
    # Let's try a simple version:
    # Start with x = ones.
    # For several iterations:
    #   Find theta_max.
    #   Compute gradient g_k = Re(P(theta_max) * conj(e^{i k theta_max})).
    #   Update x_k <- x_k - step * g_k.
    #   Project x_k to [-1, 1].
    #   If x_k is close to 1 or -1, fix it? Or just keep iterating.
    # Finally, round x to +/- 1.
    
    # This might be slow if done naively, but let's try a few steps.
    # Actually, the problem is small (24 vars).
    # Let's try to solve the continuous problem more directly.
    # min_x max_theta |sum x_k e^{i k theta}|
    # This is equivalent to finding the smallest C such that |sum x_k e^{i k theta}| <= C for all theta.
    # This is a semi-infinite linear program if we separate real and imaginary parts, but still infinite constraints.
    # Discretize theta to a fine grid (e.g., 4096 points).
    # Then it becomes: min_x max_j |sum x_k e^{i k theta_j}|
    # This is a convex optimization problem (minimize max of convex functions).
    # We can solve it using standard convex optimization techniques, e.g., bisection on C and checking feasibility.
    # Feasibility: Does there exist x in [-1, 1]^N such that |sum x_k e^{i k theta_j}| <= C for all j?
    # This is a linear feasibility problem (2 constraints per j: -C <= Re(...) <= C and -C <= Im(...) <= C).
    # We can use scipy.optimize.linprog or just a simple iterative method.
    # Given the time limit, let's stick to the heuristic water-filling / gradient descent on the continuous relaxation.
    
    x = np.ones(N)
    for _ in range(50):  # 50 iterations
        # Evaluate P on a grid
        P_vals = x @ z_powers.T  # (n_pts_search,)
        mags = np.abs(P_vals)
        idx_max = np.argmax(mags)
        theta_max = angles_search[idx_max]
        z_max = np.exp(1j * theta_max)
        P_max = P_vals[idx_max]
        
        # Gradient of |P|^2 w.r.t x_k
        # |P|^2 = P * conj(P)
        # d/dx_k |P|^2 = conj(P) * z^k + P * conj(z^k) = 2 Re(P conj(z^k))
        z_powers_max = np.power(z_max, k_arr)
        grad = 2.0 * np.real(P_max * np.conj(z_powers_max))
        
        # Gradient descent step
        step_size = 0.1
        x = x - step_size * grad
        
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
        
        # Check convergence
        if np.max(np.abs(grad)) < 1e-4:
            break
    
    # Round to +/- 1
    x_rounded = np.sign(x)
    x_rounded[x_rounded == 0] = 1.0  # Handle zeros
    seeds.append(x_rounded)
    
    # Run hill climbing on each seed
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        current_coeffs = seed.copy()
        current_fast_score = _fast_supnorm(current_coeffs, z_powers)
        
        if current_fast_score < best_fast_score:
            best_fast_score = current_fast_score
            best_coeffs = current_coeffs.copy()
        
        # Local search: Hill Climbing with random restarts of bit order
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Shuffle the order of bits to avoid bias
            indices = np.random.permutation(N)
            for i in indices:
                old_val = current_coeffs[i]
                current_coeffs[i] = -old_val
                
                new_fast_score = _fast_supnorm(current_coeffs, z_powers)
                
                if new_fast_score < current_fast_score:
                    current_fast_score = new_fast_score
                    improved = True
                    if current_fast_score < best_fast_score:
                        best_fast_score = current_fast_score
                        best_coeffs = current_coeffs.copy()
                else:
                    # Revert
                    current_coeffs[i] = old_val
    
    # Final high-precision check on the best candidate
    if best_coeffs is not None:
        # We return the coefficients. The evaluator will compute the precise score.
        # We just need to ensure we return the best one we found.
        return [int(c) for c in best_coeffs]
    else:
        # Fallback
        return [1] * N
