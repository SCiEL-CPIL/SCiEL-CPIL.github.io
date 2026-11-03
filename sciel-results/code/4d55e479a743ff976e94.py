import numpy as np
import time
import sys
from scipy.optimize import minimize
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_conv)

def _find_sidon_support(target_k):
    """Greedy construction of a Sidon-like set to minimize pairwise sum collisions."""
    if target_k <= 0:
        return []
    
    support = [0]
    used_sums = {0: 1}
    
    for _ in range(target_k - 1):
        best_candidate = None
        best_score = -1
        
        for candidate in range(N):
            if candidate in support:
                continue
            
            # Count new collisions
            new_collisions = 0
            for s in support:
                pair_sum = candidate + s
                if pair_sum in used_sums:
                    new_collisions += 1
            
            # We want to minimize collisions, so maximize (target - collisions)
            score = target_k - new_collisions
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                pair_sum = best_candidate + s
                used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
    
    return support

def _optimize_epigraph(h_init, maxiter=500, ftol=1e-14):
    """
    Optimize h using SLSQP with epigraph formulation.
    Variables: x = [h_0, ..., h_{N-1}, M]
    Minimize M subject to sum(h)=1, M >= conv_k for all k, h >= 0
    """
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
        # d(M - conv_k)/dh_j = -2 * h_{k-j} where valid
        for j in range(N):
            k_start = j
            k_end = min(j + N, num_conv)
            if k_start < num_conv and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] = -2.0 * h[indices_h]
        return jac
        
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    if np.sum(h_init) < 1e-15:
        return None
    h_init = h_init / np.sum(h_init)
    conv_init = np.convolve(h_init, h_init, mode='full')
    M_init = np.max(conv_init) + 1e-8
    x0 = np.concatenate((h_init, [M_init]))
    
    try:
        res = opt.minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            jac=objective_grad,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        
        if res.success:
            h_candidate = res.x[:N]
            h_candidate = np.clip(h_candidate, 0, None)
            if np.sum(h_candidate) > 1e-15:
                h_candidate /= np.sum(h_candidate)
                return h_candidate
    except Exception:
        pass
    return None

def _block_coordinate_refine(h_start, max_iter=80, deadline=None):
    """Block-coordinate descent refinement."""
    h = h_start.copy()
    h = np.clip(h, 0, None)
    if np.sum(h) < 1e-15:
        return h
    h /= np.sum(h)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iter_count += 1
        
        conv = np.convolve(h, h, mode='full')
        k_star = np.argmax(conv)
        M_val = conv[k_star]
        
        # For each position i, try to optimize h[i] and h[k_star - i] (if valid)
        for i in range(N):
            j = k_star - i
            if j < 0 or j >= N or j == i:
                continue
                
            hi = h[i]
            hj = h[j]
            
            t_min = -hi
            t_max = hj
            
            if t_max - t_min < 1e-15:
                continue
                
            best_t = 0.0
            best_local_score = _score_raw(h)
            
            # Finer search with more points
            for t in np.linspace(t_min * 0.9, t_max * 0.9, 20):
                h_test = h.copy()
                h_test[i] += t
                h_test[j] -= t
                if np.any(h_test < -1e-15):
                    continue
                h_test = np.clip(h_test, 0, None)
                if np.sum(h_test) < 1e-15:
                    continue
                h_test /= np.sum(h_test)
                s_test = _score_raw(h_test)
                if s_test > best_local_score:
                    best_local_score = s_test
                    best_t = t
            
            if abs(best_t) > 1e-10:
                h[i] += best_t
                h[j] -= best_t
                h = np.clip(h, 0, None)
                if np.sum(h) < 1e-15:
                    break
                h /= np.sum(h)
                improved = True
        
        # Also try single-variable moves: increase h[i], decrease others proportionally
        for i in range(N):
            if h[i] < 1e-15:
                continue
            # Try reducing h[i] slightly and distributing to other positions
            for delta_frac in [0.05, 0.1, 0.2]:
                delta = h[i] * delta_frac
                if delta < 1e-15:
                    continue
                h_test = h.copy()
                h_test[i] -= delta
                # Distribute to other positions proportional to their current values
                others = np.ones(N)
                others[i] = 0
                others *= h
                total_others = np.sum(others)
                if total_others < 1e-15:
                    continue
                others /= total_others
                h_test += delta * others
                h_test = np.clip(h_test, 0, None)
                if np.sum(h_test) < 1e-15:
                    continue
                h_test /= np.sum(h_test)
                s_test = _score_raw(h_test)
                if s_test > _score_raw(h):
                    h = h_test.copy()
                    improved = True
                    break
    
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sidon-based initializations (key structural idea from inspiration)
    for target_k in range(3, 13):
        support = _find_sidon_support(target_k)
        if len(support) >= 3:
            h_sidon = np.zeros(N)
            h_sidon[support] = 1.0
            h_sidon /= np.sum(h_sidon)
            inits.append(h_sidon)
    
    # 3. Sparse structured random
    np.random.seed(42)
    for trial in range(20):
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 4. Concentrated Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 5. Triangular
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 6. Random smooth
    for _ in range(15):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 7. Spike-like (single peak)
    for pos in range(N):
        h_spike = np.zeros(N)
        h_spike[pos] = 1.0
        inits.append(h_spike)
        
    # 8. Two-spike
    for pos1 in range(0, N, 3):
        for pos2 in range(pos1+2, N, 3):
            h_two = np.zeros(N)
            h_two[pos1] = 1.0
            h_two[pos2] = 1.0
            h_two /= np.sum(h_two)
            inits.append(h_two)
    
    # 9. Geometric progressions
    for base in [2, 3]:
        support = []
        val = 0
        while val < N:
            support.append(val)
            val = val * base + 1
        if len(support) > 2:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0
            h_geom /= np.sum(h_geom)
            inits.append(h_geom)
    
    # 10. Random Sidon-like: pick positions with large minimum pairwise difference
    np.random.seed(123)
    for trial in range(15):
        k = np.random.randint(5, 12)
        # Greedy selection with minimum spacing
        positions = []
        for _ in range(k):
            candidates = np.arange(N)
            # Penalize positions close to already selected
            scores = np.ones(N)
            for p in positions:
                dist = np.abs(candidates - p)
                scores *= (1 + dist / N)
            # Also penalize based on current convolution contribution
            if positions:
                h_temp = np.zeros(N)
                for p in positions:
                    h_temp[p] = 1.0
                conv_temp = np.convolve(h_temp, h_temp, mode='full')
                scores *= (1 + conv_temp[:N] / (np.max(conv_temp) + 1e-10))
            # Pick position with highest score
            best_pos = np.argmax(scores)
            positions.append(best_pos)
        h_sidon2 = np.zeros(N)
        for pos in positions:
            h_sidon2[pos] = 1.0
        if np.sum(h_sidon2) > 1e-10:
            h_sidon2 /= np.sum(h_sidon2)
            inits.append(h_sidon2)

    # --- Phase 2: Epigraph Formulation Optimization ---
    # Sort inits by initial score (best first) to prioritize good starting points
    inits_with_scores = [(h, _score_raw(h)) for h in inits]
    inits_with_scores.sort(key=lambda x: -x[1])
    
    # Run epigraph from multiple inits, prioritizing better initializations
    for h_init, init_score in inits_with_scores:
        if time.time() > deadline - 40:
            break
        # Skip if initial score is much worse than current best
        if init_score < best_score * 0.5:
            continue
        h_candidate = _optimize_epigraph(h_init, maxiter=500, ftol=1e-14)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 3: Block-Coordinate Descent Refinement ---
    # Refine best solution with block coordinate descent
    if time.time() < deadline - 20:
        h_refined = _block_coordinate_refine(best_h, max_iter=80, deadline=deadline - 15)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined.copy()
        
        # Second round of refinement
        if time.time() < deadline - 10:
            h_refined2 = _block_coordinate_refine(best_h, max_iter=50, deadline=deadline - 10)
            s_refined2 = _score_raw(h_refined2)
            if s_refined2 > best_score:
                best_score = s_refined2
                best_h = h_refined2.copy()
    
    # --- Phase 4: Local search with random perturbations ---
    if time.time() < deadline - 8:
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            h_pert = best_h.copy()
            # Random perturbation with varying strength
            strength = 0.1 + 0.2 * np.random.rand()
            mask = np.random.rand(N) < 0.3
            h_pert[mask] *= (1 + strength * np.random.randn(np.sum(mask)))
            h_pert = np.clip(h_pert, 0, None)
            if np.sum(h_pert) < 1e-15:
                continue
            h_pert /= np.sum(h_pert)
            
            s_pert = _score_raw(h_pert)
            if s_pert > best_score:
                best_score = s_pert
                best_h = h_pert.copy()
                
                # Refine the new best
                if time.time() < deadline - 5:
                    h_refined = _block_coordinate_refine(best_h, max_iter=30, deadline=deadline - 5)
                    s_refined = _score_raw(h_refined)
                    if s_refined > best_score:
                        best_score = s_refined
                        best_h = h_refined.copy()
    
    # Final polish: run epigraph one more time from the best solution
    if time.time() < deadline - 5:
        h_candidate = _optimize_epigraph(best_h, maxiter=300, ftol=1e-14)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
