<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    N = 24
    
    # Grid for fast evaluation
    n_grid = 2048
    angles = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    k_arr = np.arange(N)
    # z_powers[j, k] = exp(i * angles[j] * k)
    z_powers = np.exp(1j * np.outer(angles, k_arr))
    
    best_score = np.inf
    best_coeffs = None
    
    # Initial candidates: random, all ones, alternating, etc.
    # We will run local search from multiple starts.
    
    # Precompute for local search:
    # For a given coeffs vector c, V = P(angles) = c @ z_powers.T
    # Flip k: new V = V - 2 * c[k] * z_powers[:, k]
    # We want to find k that minimizes max |V - 2 * c[k] * z_powers[:, k]|
    
    # Let's define a function to do one step of local search
    def local_search(c, max_iters=50):
        c = c.copy()
        current_max = _fast_supnorm(c, z_powers, angles)
        
        for _ in range(max_iters):
            # Compute V
            V = c @ z_powers.T
            best_k = -1
            best_max = current_max
            
            # Vectorized evaluation of all flips
            # c[:, None] is (N, 1)
            # z_powers.T is (N, M)
            # We want to compute |V[None, :] - 2 * c[:, None] * z_powers.T|
            # This is (N, M)
            # Then take max over axis 1 -> (N,)
            
            # Optimization: V is (M,). c is (N,). z_powers.T is (N, M).
            # term_k = 2 * c[k] * z_powers.T[k, :]
            # new_vals_k = V - term_k
            # max_mag_k = max(|new_vals_k|)
            
            # Broadcasting:
            # c[:, None] * z_powers.T -> (N, M)
            # V[None, :] -> (1, M)
            # diff = V[None, :] - 2.0 * c[:, None] * z_powers.T
            # mags = np.abs(diff)
            # max_mags = np.max(mags, axis=1)
            
            diff = V[None, :] - 2.0 * c[:, None] * z_powers.T
            max_mags = np.max(np.abs(diff), axis=1)
            
            min_idx = np.argmin(max_mags)
            min_val = max_mags[min_idx]
            
            if min_val < current_max * (1.0 - 1e-12): # Strict improvement
                c[min_idx] = -c[min_idx]
                current_max = min_val
            else:
                break
        
        return c, current_max

    # Run many random restarts
    n_restarts = 0
    while time.time() < deadline:
        # Random start
        c = np.random.choice([-1.0, 1.0], size=N)
        
        # Perform local search
        c_opt, score_est = local_search(c)
        
        # Update best
        if score_est < best_score:
            best_score = score_est
            best_coeffs = c_opt.copy()
            
        n_restarts += 1
        
        # If we've found a very good score, we might want to stop early?
        # The target is 1.2583758 * sqrt(24) ~ 6.146
        # best_score is the raw supnorm.
        # target_supnorm = 1.2583758 * np.sqrt(24)
        # if best_score <= target_supnorm * 1.01:
        #     break

    # Final high-precision check on the best candidate
    if best_coeffs is not None:
        # It's possible the fast grid missed the true peak.
        # Let's do a final polish with high precision on the best candidate.
        # Also, try all single flips with high precision to see if we can improve further.
        
        best_hp_score = _supnorm_high_precision(best_coeffs)
        
        # Try single flips with high precision
        improved = True
        while improved:
            improved = False
            for k in range(N):
                c_flip = best_coeffs.copy()
                c_flip[k] = -c_flip[k]
                score_flip = _supnorm_high_precision(c_flip)
                if score_flip < best_hp_score:
                    best_hp_score = score_flip
                    best_coeffs = c_flip
                    improved = True
                    break # Restart the loop with the new best
            
            if time.time() > deadline - 5:
                break

        # Return the coefficients as a list of ints
        return [int(c) for c in best_coeffs]
    
    # Fallback if no search was performed (should not happen)
    return [1] * N
=======
def _blahut_arimoto_relax(N, n_grid=4096, max_iter=200, tol=1e-8):
    """
    Solve the continuous convex relaxation:
    min_x max_theta |sum_k x_k e^{i k theta}|
    s.t. x_k in [-1, 1]
    
    Using Blahut-Arimoto style iterative water-filling / dual ascent.
    The dual problem involves finding a measure on the unit circle.
    We use a finite grid approximation and iterate.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    k_arr = np.arange(N)
    z_powers = np.exp(1j * np.outer(angles, k_arr))  # (M, N)
    
    # Initialize x uniformly at 1 (or use a better init)
    x = np.ones(N, dtype=np.float64)
    
    for it in range(max_iter):
        if time.time() > _deadline - 15:
            break
        
        # Compute P(theta) = sum_k x_k e^{i k theta}
        P = x @ z_powers.T  # (M,)
        mags = np.abs(P)  # (M,)
        
        # The subgradient of max |P| w.r.t. x_k is related to the phase
        # We want to push x_k in the direction that reduces the max
        # Blahut-Arimoto: x_k <- x_k * (average of e^{-i arg(P) * k} weighted by |P|/max)
        # More precisely, for the epigraph formulation:
        # min_{x, t} t s.t. |sum x_k e^{ikθ}| <= t for all θ
        # Dual: max_μ sum_μ - ... 
        # Simple iteration: x_k <- clip(x_k - eta * grad_k, -1, 1)
        # where grad_k = <P, e^{ikθ}> / |P| averaged over peak points
        
        # Find the peak points
        peak_threshold = np.percentile(mags, 95)
        peak_mask = mags >= peak_threshold
        
        if not np.any(peak_mask):
            peak_mask = mags >= mags.max() * 0.99
        
        # Gradient of max|P| w.r.t. x_k at peak points:
        # d/dx_k |P| = Re(conj(P) * e^{ikθ}) / |P| at each peak
        # Average over peaks
        phase_factors = z_powers[peak_mask, :]  # (n_peaks, N)
        P_peaks = P[peak_mask]  # (n_peaks,)
        
        # grad_k = mean_j Re(conj(P_j) * e^{ikθ_j}) / |P_j|
        # = mean_j Re(conj(P_j) / |P_j| * e^{ikθ_j})
        # = mean_j Re(e^{-i arg(P_j)} * e^{ikθ_j})
        
        phase_conj = np.conj(P_peaks) / mags[peak_mask, None] if mags[peak_mask].max() > 0 else np.ones_like(P_peaks)
        # phase_conj is (n_peaks,)
        
        # grad_k = Re(mean_j phase_conj_j * e^{ikθ_j})
        # = Re(conj(mean_j phase_conj_j * conj(e^{ikθ_j})))
        # Let's compute: mean_j phase_conj_j * e^{ikθ_j} for each k
        # This is (phase_conj @ conj(z_powers[peak_mask, :]))
        # Wait, z_powers[peak_mask, k] = e^{i * angles[peak_mask] * k}
        # We want mean_j phase_conj_j * e^{i k θ_j}
        # = conj( mean_j conj(phase_conj_j) * e^{-i k θ_j} )
        # = conj( mean_j conj(phase_conj_j) * conj(e^{i k θ_j}) )
        
        # Simpler: directly compute
        # grad_complex[k] = mean_j phase_conj[j] * z_powers[peak_mask, j, k]
        # But z_powers is (M, N), so z_powers[peak_mask, :] is (n_peaks, N)
        # phase_conj is (n_peaks,)
        # grad_complex = phase_conj @ z_powers[peak_mask, :]  # (N,)
        
        grad_complex = phase_conj @ z_powers[peak_mask, :]
        grad = np.real(grad_complex)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Step size: use a simple adaptive scheme
        # We want to move x in the direction -grad
        # But x must stay in [-1, 1]
        
        # Use a projected gradient step
        eta = 0.1  # step size
        
        x_new = x - eta * grad
        x_new = np.clip(x_new, -1.0, 1.0)
        
        # Check convergence
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        
        x = x_new
    
    return x

def _round_continuous_to_binary(x, N, z_powers, angles):
    """
    Given continuous x in [-1, 1]^N, round to ±1 vector.
    Use multiple rounding strategies and pick the best.
    """
    candidates = []
    
    # Strategy 1: Sign-based rounding
    c1 = np.sign(x)
    c1[c1 == 0] = 1.0
    candidates.append(c1)
    
    # Strategy 2: Threshold at 0 with random tie-breaking
    for _ in range(10):
        c2 = np.ones(N)
        for k in range(N):
            if x[k] > 0:
                c2[k] = 1.0
            elif x[k] < 0:
                c2[k] = -1.0
            else:
                c2[k] = random.choice([-1.0, 1.0])
        candidates.append(c2)
    
    # Strategy 3: Probabilistic rounding based on |x_k|
    for _ in range(10):
        c3 = np.ones(N)
        for k in range(N):
            p = 0.5 + 0.5 * x[k]  # P(c_k = 1)
            c3[k] = 1.0 if random.random() < p else -1.0
        candidates.append(c3)
    
    # Evaluate all candidates
    best_c = None
    best_score = np.inf
    for c in candidates:
        score = _fast_supnorm(c, z_powers, angles)
        if score < best_score:
            best_score = score
            best_c = c.copy()
    
    return best_c

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 110.0  # 10s margin
    
    N = 24
    
    # Grid for fast evaluation
    n_grid = 2048
    angles = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    k_arr = np.arange(N)
    # z_powers[j, k] = exp(i * angles[j] * k)
    z_powers = np.exp(1j * np.outer(angles, k_arr))
    
    best_score = np.inf
    best_coeffs = None
    
    # Precompute for local search:
    # For a given coeffs vector c, V = P(angles) = c @ z_powers.T
    # Flip k: new V = V - 2 * c[k] * z_powers[:, k]
    # We want to find k that minimizes max |V - 2 * c[k] * z_powers[:, k]|
    
    # Let's define a function to do one step of local search
    def local_search(c, max_iters=100):
        c = c.copy()
        current_max = _fast_supnorm(c, z_powers, angles)
        
        for _ in range(max_iters):
            if time.time() > _deadline - 10:
                break
            
            # Compute V
            V = c @ z_powers.T
            best_k = -1
            best_max = current_max
            
            # Vectorized evaluation of all flips
            diff = V[None, :] - 2.0 * c[:, None] * z_powers.T
            max_mags = np.max(np.abs(diff), axis=1)
            
            min_idx = np.argmin(max_mags)
            min_val = max_mags[min_idx]
            
            if min_val < current_max * (1.0 - 1e-12): # Strict improvement
                c[min_idx] = -c[min_idx]
                current_max = min_val
            else:
                break
        
        return c, current_max
    
    def local_search_pair(c, max_iters=50):
        """Local search with pair flips to escape local minima."""
        c = c.copy()
        current_max = _fast_supnorm(c, z_powers, angles)
        
        for _ in range(max_iters):
            if time.time() > _deadline - 10:
                break
            
            V = c @ z_powers.T
            best_pair = None
            best_max = current_max
            
            # Try all pairs (i, j) with i < j
            for i in range(N):
                for j in range(i + 1, N):
                    # Flip both i and j
                    new_V = V - 2.0 * c[i] * z_powers[:, i] - 2.0 * c[j] * z_powers[:, j]
                    new_max = np.max(np.abs(new_V))
                    if new_max < best_max * (1.0 - 1e-12):
                        best_max = new_max
                        best_pair = (i, j)
            
            if best_pair is not None:
                i, j = best_pair
                c[i] = -c[i]
                c[j] = -c[j]
                current_max = best_max
            else:
                break
        
        return c, current_max
    
    # Phase 1: Blahut-Arimoto relaxation
    if time.time() < _deadline - 60:
        x_cont = _blahut_arimoto_relax(N, n_grid=4096, max_iter=100)
        c_init = _round_continuous_to_binary(x_cont, N, z_powers, angles)
        
        # Refine with local search
        c_opt, score_est = local_search(c_init)
        if score_est < best_score:
            best_score = score_est
            best_coeffs = c_opt.copy()
        
        # Try pair flips from the relaxed solution
        c_pair, score_pair = local_search_pair(c_opt)
        if score_pair < best_score:
            best_score = score_pair
            best_coeffs = c_pair.copy()
    
    # Phase 2: Random restarts with local search
    n_restarts = 0
    while time.time() < _deadline - 30:
        # Random start
        c = np.random.choice([-1.0, 1.0], size=N)
        
        # Perform local search
        c_opt, score_est = local_search(c, max_iters=80)
        
        # Update best
        if score_est < best_score:
            best_score = score_est
            best_coeffs = c_opt.copy()
        
        # Occasionally try pair flips
        if n_restarts % 5 == 0:
            c_pair, score_pair = local_search_pair(c_opt, max_iters=20)
            if score_pair < best_score:
                best_score = score_pair
                best_coeffs = c_pair.copy()
        
        n_restarts += 1
    
    # Phase 3: Final high-precision polish
    if best_coeffs is not None and time.time() < _deadline - 5:
        best_hp_score = _supnorm_high_precision(best_coeffs)
        
        # Try single flips with high precision
        improved = True
        while improved:
            improved = False
            for k in range(N):
                if time.time() > _deadline - 5:
                    break
                c_flip = best_coeffs.copy()
                c_flip[k] = -c_flip[k]
                score_flip = _supnorm_high_precision(c_flip)
                if score_flip < best_hp_score:
                    best_hp_score = score_flip
                    best_coeffs = c_flip
                    improved = True
                    break
            
            if time.time() > _deadline - 5:
                break
        
        # Try pair flips with high precision if time allows
        if time.time() < _deadline - 10:
            improved_pair = True
            while improved_pair:
                improved_pair = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if time.time() > _deadline - 5:
                            break
                        c_flip = best_coeffs.copy()
                        c_flip[i] = -c_flip[i]
                        c_flip[j] = -c_flip[j]
                        score_flip = _supnorm_high_precision(c_flip)
                        if score_flip < best_hp_score:
                            best_hp_score = score_flip
                            best_coeffs = c_flip
                            improved_pair = True
                            break
                    if improved_pair:
                        break
        
        return [int(c) for c in best_coeffs]
    
    # Fallback if no search was performed (should not happen)
    return [1] * N
>>>>>>> REPLACE
