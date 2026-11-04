import numpy as np
import scipy.optimize
import time

N = 100
TARGET_SUM = N / 2.0

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses differential evolution for global search (handles non-smooth max objective),
    then polishes with L-BFGS-B using LogSumExp smoothing.
    """
    start_time = time.time()
    deadline = 135.0
    safety_margin = 5.0
    effective_deadline = deadline - safety_margin
    
    def get_correlations(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts."""
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)
    
    def project_to_feasible(h):
        """Project h onto feasible set: 0 <= h_i <= 1, sum(h) = N/2."""
        h = np.clip(h, 0.0, 1.0)
        for _ in range(100):
            current_sum = np.sum(h)
            diff = TARGET_SUM - current_sum
            if abs(diff) < 1e-14:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        residual = TARGET_SUM - np.sum(h)
        if abs(residual) > 1e-15:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h
    
    def lse_objective(h, beta):
        """LogSumExp of correlations for smooth approximation of max."""
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse
    
    def de_objective(h):
        """Objective for differential evolution: exact max overlap."""
        h = project_to_feasible(h)
        return score_h(h)
    
    def polish_with_lbfgsb(h0, beta, max_iter):
        """Polish using L-BFGS-B with LSE smoothing."""
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)
        
        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt
    
    # Phase 1: Differential Evolution for global search
    rng = np.random.default_rng(42)
    
    bounds_de = [(0, 1) for _ in range(N)]
    
    # Run DE with moderate population and iterations
    de_result = None
    if time.time() < effective_deadline - 30.0:
        try:
            de_result = scipy.optimize.differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,  # We'll do our own polish
                init='sobol',
                updating='deferred',
                workers=1
            )
        except Exception:
            de_result = None
    
    best_h = None
    best_score = None
    
    # Collect candidates
    candidates = []
    
    # Baseline
    h_uniform = np.full(N, 0.5)
    candidates.append(h_uniform)
    
    # DE result
    if de_result is not None and de_result.success:
        candidates.append(de_result.x)
    
    # Add some structured initializations
    # Alternating pattern
    h_alt = np.zeros(N)
    h_alt[::2] = 0.8
    h_alt[1::2] = 0.2
    candidates.append(project_to_feasible(h_alt))
    
    # Block pattern
    h_block = np.zeros(N)
    h_block[:N//2] = 1.0
    h_block[N//2:] = 0.0
    candidates.append(project_to_feasible(h_block))
    
    # Random candidates
    for i in range(10):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(project_to_feasible(h_rand))
    
    # Evaluate all candidates
    for h_cand in candidates:
        h_cand = project_to_feasible(h_cand)
        s = score_h(h_cand)
        if best_score is None or s < best_score:
            best_score = s
            best_h = h_cand.copy()
    
    # Phase 2: Polish the best candidate with L-BFGS-B
    if time.time() < effective_deadline - 10.0:
        for beta in [10.0, 50.0, 100.0, 200.0]:
            if time.time() > effective_deadline - 5.0:
                break
            h_polished = polish_with_lbfgsb(best_h, beta, 500)
            s = score_h(h_polished)
            if s < best_score:
                best_score = s
                best_h = h_polished.copy()
    
    # Phase 3: Local search around best solution
    if time.time() < effective_deadline - 5.0:
        # Try small perturbations and polish
        for i in range(5):
            if time.time() > effective_deadline - 3.0:
                break
            h_perturbed = best_h + rng.normal(0, 0.05, size=N)
            h_perturbed = project_to_feasible(h_perturbed)
            h_polished = polish_with_lbfgsb(h_perturbed, 50.0, 300)
            s = score_h(h_polished)
            if s < best_score:
                best_score = s
                best_h = h_polished.copy()
    
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
