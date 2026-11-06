import numpy as np
import time

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    dim = 1 << n
    psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute cost values (number of cut edges) for each basis state
    costs = np.zeros(dim, dtype=float)
    for state in range(dim):
        cut_count = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_count += 1
        costs[state] = cut_count
    
    # Precompute index pairs for mixer: for each qubit i, pairs of states (state, state|bit)
    # where bit i is 0
    mixer_pairs = []
    for i in range(n):
        bit = 1 << i
        # States where bit i is 0
        idx0 = np.array([s for s in range(dim) if (s & bit) == 0], dtype=np.int64)
        idx1 = idx0 | bit
        mixer_pairs.append((idx0, idx1))
    
    def _apply_mixer(psi, beta, n):
        """Apply exp(-i beta * sum X_i) using vectorized operations."""
        c = np.cos(beta)
        s = np.sin(beta)
        for idx0, idx1 in mixer_pairs:
            a0 = psi[idx0].copy()
            a1 = psi[idx1].copy()
            psi[idx0] = c * a0 - 1j * s * a1
            psi[idx1] = -1j * s * a0 + c * a1
        return psi
    
    def _qaoa_state(gammas, betas):
        """Compute QAOA state for given angles."""
        psi = psi0.copy()
        for k in range(p):
            psi = psi * np.exp(-1j * gammas[k] * costs)
            psi = _apply_mixer(psi, betas[k], n)
        return psi
    
    def _expected_cut(psi):
        """Compute expected cut from state vector."""
        probs = np.abs(psi)**2
        return np.dot(probs, costs)
    
    def objective(params):
        gammas = params[0:p]
        betas = params[p:2*p]
        psi = _qaoa_state(gammas, betas)
        return -_expected_cut(psi)
    
    # Simple Nelder-Mead implementation
    def nelder_mead(func, x0, max_iter=200, timeout=10.0):
        x = np.array(x0, dtype=float)
        n = len(x)
        simplex = [x.copy()]
        for i in range(n):
            new_x = x.copy()
            new_x[i] = new_x[i] + 0.1 if new_x[i] != 0 else 0.1
            simplex.append(new_x)
        
        fvals = [func(s) for s in simplex]
        best_val = min(fvals)
        best_x = simplex[fvals.index(best_val)].copy()
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        for _ in range(max_iter):
            if time.time() > start_time + timeout:
                break
            
            idx = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in idx]
            fvals = [fvals[i] for i in idx]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fvals[0]:
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + sigma * (xr - centroid)
                    fc = func(xc)
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break
                else:
                    xc = centroid - sigma * (centroid - simplex[-1])
                    fc = func(xc)
                    if fc <= fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        break
            
            if time.time() > start_time + timeout:
                break
            
            cur_min = min(fvals)
            if cur_min < best_val:
                best_val = cur_min
                best_x = simplex[fvals.index(cur_min)].copy()
            
            if abs(fvals[0] - fvals[-1]) < 1e-10:
                break
                
        return best_x, best_val

    # Physics-informed initial guesses based on known QAOA MaxCut angles
    # Asymptotic: gamma ~ pi/(4m), beta ~ pi/4 for small p
    # For p=2, typical good values: gamma1~0.5-0.8, beta1~0.3-0.5, gamma2~0.3-0.6, beta2~0.2-0.4
    initial_guesses = [
        [0.55, 0.35, 0.45, 0.25],   # physics-informed center
        [0.70, 0.40, 0.50, 0.30],   # slightly higher
        [0.40, 0.30, 0.35, 0.20],   # slightly lower
        [0.80, 0.45, 0.55, 0.35],   # higher
        [0.30, 0.25, 0.25, 0.15],   # lower
        [0.60, 0.50, 0.40, 0.30],   # high beta
        [0.45, 0.20, 0.35, 0.15],   # low beta
        [0.90, 0.35, 0.60, 0.25],   # high gamma
        [0.50, 0.40, 0.30, 0.20],   # decreasing
        [0.65, 0.30, 0.50, 0.20],   # another variant
        [0.35, 0.45, 0.25, 0.30],   # inverted pattern
        [0.75, 0.25, 0.55, 0.15],   # high gamma, low beta
    ]
    
    best_params = None
    best_val = np.inf
    
    # Phase 1: Quick scan with short timeouts to find promising regions
    time_per_restart_quick = 3.0
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time()
        current_timeout = min(time_per_restart_quick, remaining * 0.5)
        if current_timeout < 0.5:
            break
        try:
            res_x, res_val = nelder_mead(objective, x0, max_iter=60, timeout=current_timeout)
            if res_val < best_val:
                best_val = res_val
                best_params = res_x
        except Exception:
            continue
    
    # Phase 2: Focused refinement on top candidates
    if best_params is not None and time.time() < deadline - 10:
        # Use the best found as center for more thorough search
        remaining = deadline - time.time() - 5
        try:
            res_x, res_val = nelder_mead(objective, best_params, max_iter=300, timeout=remaining)
            if res_val < best_val:
                best_val = res_val
                best_params = res_x
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_params is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        try:
            res_x, res_val = nelder_mead(objective, best_params, max_iter=200, timeout=remaining)
            if res_val < best_val:
                best_val = res_val
                best_params = res_x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([[0.55, 0.35], [0.45, 0.25]])
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))
