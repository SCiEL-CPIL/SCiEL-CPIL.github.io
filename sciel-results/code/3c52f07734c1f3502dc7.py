import numpy as np
import scipy.optimize
import time

N = 100
target_sum = N / 2.0
# FFT size for zero-padded linear correlation
fft_size = 1 << (2 * N - 1).bit_length()

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        Uses FFT for efficiency.
        """
        g = 1.0 - h
        H = np.fft.rfft(h, n=fft_size)
        G = np.fft.rfft(g, n=fft_size)
        # Correlation: sum g[n] h[n+k]
        # In frequency domain: IFFT(G * conj(H))
        # Since real inputs, use rfft/irfft
        corr_full = np.fft.irfft(G * np.conj(H), n=fft_size)
        
        # Extract linear correlation part
        # np.correlate(h, g, 'full') has length 2N-1
        # Index N-1 is lag 0.
        # Lags -N+1 ... -1 are indices 0 ... N-2
        # Lags 1 ... N-1 are indices N ... 2N-2
        # In FFT result (length fft_size >= 2N-1):
        # Lag k corresponds to index k if k >= 0, or fft_size + k if k < 0.
        
        # Let's just extract from the first 2N-1 elements
        corr_linear = corr_full[:2*N-1]
        
        # Non-zero shifts:
        # Negative shifts: indices 0 to N-2 (lags -N+1 to -1)
        # Positive shifts: indices N to 2N-2 (lags 1 to N-1)
        c_nonzero = np.concatenate([corr_linear[:N-1], corr_linear[N:]])
        return c_nonzero

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def smooth_max(c, temp=0.01):
        """
        Smooth maximum using log-sum-exp.
        """
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = N/2.
        """
        h = np.array(h, dtype=float).copy()
        h = np.clip(h, 0.0, 1.0)
        
        # Iteratively adjust to satisfy sum constraint
        for _ in range(100):
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            
            if abs(diff) < 1e-12:
                break
            
            if diff > 0:
                # Need to increase sum
                mask = h < 1.0
                if not np.any(mask):
                    break
                n_incr = np.sum(mask)
                step = diff / n_incr
                h[mask] = np.minimum(h[mask] + step, 1.0)
            else:
                # Need to decrease sum
                mask = h > 0.0
                if not np.any(mask):
                    break
                n_decr = np.sum(mask)
                step = diff / n_decr
                h[mask] = np.maximum(h[mask] + step, 0.0)
        
        h = np.clip(h, 0.0, 1.0)
        return h

    def objective_lse(h, beta, lambda_sum):
        """
        Objective: LSE(correlations) + penalty(sum - N/2)^2
        """
        c_all = get_correlations(h)
        
        # LogSumExp for numerical stability
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500
        exps = np.exp(beta * (c_all - max_c))
        sum_exps = np.sum(exps)
        lse = max_c + np.log(sum_exps) / beta
        
        # Sum penalty
        sum_h = np.sum(h)
        sum_pen = lambda_sum * (sum_h - target_sum)**2
        
        val = lse + sum_pen
        return val

    def optimize_instance(h0, beta, lambda_sum, max_iter, ftol=1e-8, gtol=1e-8):
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            return objective_lse(h, beta, lambda_sum)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0.copy()
        
        # Project to feasible set
        h_opt = project_to_feasible(h_opt)
        
        return h_opt

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)

    rng = np.random.default_rng(42)
    
    # List of restart configurations
    restarts = []
    
    # 1. Uniform start
    restarts.append((np.full(N, 0.5), 5.0, 100.0, 500))
    
    # 2-5. Random perturbations
    for i in range(4):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = h_rand / np.sum(h_rand) * target_sum
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 5.0 + i * 5.0
        lambda_sum = 100.0 + i * 50.0
        
        restarts.append((h_rand, beta, lambda_sum, 500))
        
    # 6-10. Bimodal initializations
    for i in range(5):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.9, 0.1)
        h_rand = h_rand / np.sum(h_rand) * target_sum
        h_rand = np.clip(h_rand, 0, 1)
        h_rand = project_to_feasible(h_rand)
        
        beta = 15.0 + i * 5.0
        lambda_sum = 200.0 + i * 50.0
        
        restarts.append((h_rand, beta, lambda_sum, 500))

    for idx, (h0, beta, lambda_sum, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit:
            break
            
        h_opt = optimize_instance(h0, beta, lambda_sum, max_iter)
        
        current_score = score_h(h_opt)
        
        if not np.isclose(np.sum(h_opt), target_sum, atol=1e-6):
            continue
        if not (np.all(h_opt >= 0) and np.all(h_opt <= 1)):
            continue
            
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
    
    # Final polish pass with tighter tolerances
    if time.time() - start_time < time_limit - 10:
        # Use the best solution found so far as starting point
        h_polish = best_h.copy()
        
        # Try several high-beta polish iterations
        for beta_polish in [30.0, 50.0, 100.0]:
            if time.time() - start_time > time_limit:
                break
            
            h_opt = optimize_instance(h_polish, beta_polish, 500.0, 1000, ftol=1e-10, gtol=1e-10)
            
            current_score = score_h(h_opt)
            
            if np.isclose(np.sum(h_opt), target_sum, atol=1e-8) and \
               np.all(h_opt >= 0) and np.all(h_opt <= 1):
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_opt.copy()
                    h_polish = h_opt.copy()
    
    # Add a simulated annealing phase with cluster moves if time permits
    if time.time() - start_time < time_limit - 15:
        def find_clusters(h, threshold=0.1):
            """Find contiguous blocks where heights are within threshold of each other"""
            clusters = []
            i = 0
            while i < N:
                j = i
                base_val = h[i]
                while j < N and abs(h[j] - base_val) < threshold:
                    j += 1
                if j > i:
                    clusters.append((i, j))
                i = j
            return clusters
        
        def cluster_move(h, clusters, rng):
            """Perform a non-local move on a cluster: shift all values in the cluster"""
            if not clusters:
                return h.copy()
            
            # Pick a random cluster
            idx = rng.integers(0, len(clusters))
            i, j = clusters[idx]
            
            # Generate a new value for the entire cluster
            h_new = h.copy()
            
            # Strategy: shift the cluster's values by a random amount
            shift = rng.uniform(-0.2, 0.2)
            h_new[i:j] += shift
            
            # Project back to feasible set
            h_new = project_to_feasible(h_new)
            return h_new
        
        def cluster_flip(h, clusters, rng):
            """Flip a cluster: replace values with 1 - values"""
            if not clusters:
                return h.copy()
            
            idx = rng.integers(0, len(clusters))
            i, j = clusters[idx]
            
            h_new = h.copy()
            h_new[i:j] = 1.0 - h[i:j]
            h_new = project_to_feasible(h_new)
            return h_new
        
        h_curr = best_h.copy()
        best_h_sa = best_h.copy()
        best_score_sa = best_score
        
        T_init = 0.1
        T_final = 0.001
        n_iterations = 3000
        
        for it in range(n_iterations):
            if time.time() - start_time > time_limit - 3:
                break
            
            # Temperature schedule
            progress = it / n_iterations
            T = T_init * (T_final / T_init) ** progress
            
            # Find clusters
            clusters = find_clusters(h_curr, threshold=0.05)
            
            # Choose move type
            move_type = rng.random()
            if move_type < 0.4:
                h_prop = cluster_move(h_curr, clusters, rng)
            elif move_type < 0.7:
                h_prop = cluster_flip(h_curr, clusters, rng)
            else:
                # Single-point perturbation
                h_prop = h_curr.copy()
                i = rng.integers(0, N)
                h_prop[i] += rng.uniform(-0.1, 0.1)
                h_prop = project_to_feasible(h_prop)
            
            # Acceptance criterion
            score_curr = score_h(h_curr)
            score_prop = score_h(h_prop)
            
            delta = score_prop - score_curr
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h_curr = h_prop
            
            # Track best
            curr_score = score_h(h_curr)
            if curr_score < best_score_sa:
                best_score_sa = curr_score
                best_h_sa = h_curr.copy()
        
        if best_score_sa < best_score:
            best_score = best_score_sa
            best_h = best_h_sa
    
    # Ensure final solution is valid
    best_h = project_to_feasible(best_h)
    
    return best_h
