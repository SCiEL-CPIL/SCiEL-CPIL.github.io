import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def _score_raw(h):
    """Compute the raw ratio value."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _find_sidon_set(N, K, attempts=200):
    """Find a Sidon set of size K in [0, N)."""
    for _ in range(attempts):
        candidates = list(range(N))
        random.shuffle(candidates)
        S = []
        sums = set()
        for cand in candidates:
            if len(S) == K:
                break
            conflict = False
            new_sums = set()
            if cand + cand in sums:
                conflict = True
            else:
                new_sums.add(cand + cand)
            if not conflict:
                for s in S:
                    sm = cand + s
                    if sm in sums or sm in new_sums:
                        conflict = True
                        break
                    new_sums.add(sm)
            if not conflict:
                S.append(cand)
                sums.update(new_sums)
        if len(S) == K:
            return S
    return None

def _generate_initializations(deadline):
    """Generate diverse initializations."""
    inits = []
    x_idx = np.arange(N)
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sidon-based (try various K)
    np.random.seed(42)
    for K in range(12, 4, -1):
        if time.time() > deadline - 30.0:
            break
        S = _find_sidon_set(N, K, attempts=100)
        if S is not None:
            h = np.zeros(N)
            h[S] = 1.0
            h /= np.sum(h)
            inits.append(h)
            if K >= 10:
                break
    
    # 3. Random sparse
    for trial in range(20):
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
    
    # 4. Concentrated Gaussians
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(1.0, N/3, 4):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
    
    # 5. Triangular
    for c in np.linspace(3, N-3, 4):
        for w in np.linspace(2, N, 4):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
    
    # 6. Random smooth
    for _ in range(10):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
    
    # 7. Symmetric pairs around center
    center = N // 2
    for k in range(3, 8):
        for offset in range(1, 6):
            h_pair = np.zeros(N)
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
    
    # 8. Two-spike
    for pos1 in range(0, N, 4):
        for pos2 in range(pos1+3, N, 4):
            h_two = np.zeros(N)
            h_two[pos1] = 1.0
            h_two[pos2] = 1.0
            h_two /= np.sum(h_two)
            inits.append(h_two)
    
    # 9. Single spike
    for pos in range(0, N, 5):
        h_spike = np.zeros(N)
        h_spike[pos] = 1.0
        inits.append(h_spike)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Phase 1: Generate and score initializations
    inits = _generate_initializations(deadline)
    
    # Score and sort initializations
    scored_inits = []
    for h in inits:
        s = _score_raw(h)
        scored_inits.append((s, h))
    scored_inits.sort(key=lambda x: -x[0])
    
    best_h = scored_inits[0][1].copy()
    best_score = scored_inits[0][0]
    
    # Phase 2: Epigraph optimization on top initializations
    num_conv = 2 * N - 1
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0
        
    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac
        
    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((num_conv, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
        
    constraints = [
        {'type': 'eq', 'fun': constraint_eq_fun, 'jac': constraint_eq_jac},
        {'type': 'ineq', 'fun': constraint_ineq_fun, 'jac': constraint_ineq_jac}
    ]
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    def run_epigraph(h_init, maxiter, ftol):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = np.convolve(h_init, h_init, mode='full')
        M_init = np.max(conv_init)
        x0 = np.concatenate([h_init, [M_init]])
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', jac=objective_grad,
                bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success or res.status == 0:
                h_res = res.x[:N]
                s = np.sum(h_res)
                if s > 0:
                    h_res = h_res / s
                    h_res = np.clip(h_res, 0, None)
                    h_res = h_res / np.sum(h_res)
                    return h_res
        except Exception:
            pass
        return None
    
    # Coarse optimization on top 8 initializations
    top_inits = scored_inits[:8]
    for score, h_init in top_inits:
        if time.time() > deadline - 40.0:
            break
        h_opt = run_epigraph(h_init, maxiter=300, ftol=1e-10)
        if h_opt is not None:
            s = _score_raw(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # Phase 3: High-precision polish on best solution
    if time.time() < deadline - 10.0:
        h_polished = run_epigraph(best_h, maxiter=500, ftol=1e-14)
        if h_polished is not None:
            s = _score_raw(h_polished)
            if s > best_score:
                best_score = s
                best_h = h_polished.copy()
    
    # Final cleanup
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
    
    return list(best_h)
