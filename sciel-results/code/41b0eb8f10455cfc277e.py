# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _build_idx():
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_a = X.astype(np.int64).ravel()                 # x
    idx_b = (X + Y).astype(np.int64).ravel()           # x+y
    idx_c = (X - Y + 9).astype(np.int64).ravel()       # x-y  (shifted >=0)
    idx_d = (X + 2 * Y).astype(np.int64).ravel()       # x+2y
    return idx_a, idx_b, idx_c, idx_d

def _entropy(q):
    m = q > 1e-300
    if not np.any(m):
        return 0.0
    return -np.sum(q[m] * np.log2(q[m]))

def _value_from_P(P, idx_a, idx_b, idx_c, idx_d):
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qc = np.bincount(idx_c, weights=pf, minlength=lc)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return -1e9
    return D / M

def _finalize(P, idx_a, idx_b, idx_c, idx_d):
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9 or not np.isfinite(s):
        return np.full((10, 10), 1.0 / 100.0)
    P = P / s
    # recompute M to ensure validity
    pf = P.ravel()
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    ld = idx_d.max() + 1
    qa = np.bincount(idx_a, weights=pf, minlength=la)
    qb = np.bincount(idx_b, weights=pf, minlength=lb)
    qd = np.bincount(idx_d, weights=pf, minlength=ld)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    if M <= 1e-9:
        return np.full((10, 10), 1.0 / 100.0)
    return P.astype(np.float64)

def _smooth_max_entropy(qs, alpha=100.0):
    """
    Approximate max(entropy(q)) using log-sum-exp.
    qs: list of entropy values.
    """
    if not qs:
        return 0.0
    qs = np.array(qs)
    # If all are close to 0, return 0 to avoid issues
    if np.max(qs) < 1e-9:
        return 0.0
    
    # Scale to avoid overflow in exp
    max_q = np.max(qs)
    scaled = (qs - max_q) * alpha
    # log-sum-exp trick: max + log(sum(exp(x - max)))
    # We want max(qs) approx.
    # Actually, log-sum-exp of scaled values gives:
    # log(sum(exp(alpha*(q_i - max_q)))) = log(exp(0) + sum(exp(alpha*(q_i-max_q)) for i!=max))
    # This is roughly 1 + sum(small terms). log(1+eps) ~ eps.
    # So result is max_q + (1/alpha) * log(sum(exp(alpha*(q_i - max_q))))
    
    # To be safe and stable:
    # LSE = max_q + (1/alpha) * np.log(np.sum(np.exp(alpha * (qs - max_q))))
    # But if alpha is large, exp(alpha*(q_i - max_q)) for i!=max is ~0.
    # So LSE ~ max_q + (1/alpha) * log(1) = max_q.
    # This is a good approximation.
    
    # However, we need to minimize -D/M.
    # Let's just use the actual max for the polishing phase, 
    # and a soft version for DE if needed? 
    # Actually, DE handles non-smoothness okay-ish, but gradients are better for L-BFGS.
    # For DE, we can just use the true max. It's robust.
    return np.max(qs)

def _objective_de(z):
    """
    Objective for Differential Evolution.
    z: log-weights (unconstrained).
    Returns: -D/M (to minimize).
    """
    # Softmax to get weights
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    w = exp_z / exp_z.sum()
    
    # Construct P
    P = w.reshape(10, 10)
    
    # Calculate entropies
    idx_a, idx_b, idx_c, idx_d = _OBJ_CACHE['idxs']
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    qa = np.bincount(idx_a, weights=w, minlength=la)
    qb = np.bincount(idx_b, weights=w, minlength=lb)
    qc = np.bincount(idx_c, weights=w, minlength=lc)
    qd = np.bincount(idx_d, weights=w, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return 1e9
    return -D / M

def _objective_lbfgs(z):
    """
    Objective for L-BFGS-B.
    z: log-weights.
    Returns: -D/M
    """
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    w = exp_z / exp_z.sum()
    
    idx_a, idx_b, idx_c, idx_d = _OBJ_CACHE['idxs']
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    qa = np.bincount(idx_a, weights=w, minlength=la)
    qb = np.bincount(idx_b, weights=w, minlength=lb)
    qc = np.bincount(idx_c, weights=w, minlength=lc)
    qd = np.bincount(idx_d, weights=w, minlength=ld)
    
    D = _entropy(qc)
    M = max(_entropy(qa), _entropy(qb), _entropy(qd))
    
    if M < 1e-12:
        return 1e9
    return -D / M

def _grad_lbfgs(z):
    """
    Gradient of -D/M with respect to z.
    This is complex due to the max. 
    We can use a smooth approximation for the gradient or just rely on L-BFGS finite diff?
    L-BFGS-B supports analytic gradients. If we don't provide one, it uses finite differences.
    Given 100 variables, finite differences are 100+ evaluations per step.
    Analytic gradient is better.
    
    Let f = -D/M.
    D = H(qc), M = max(H(qa), H(qb), H(qd)).
    
    Let's use a log-sum-exp approximation for M in the gradient calculation to make it smooth.
    M_smooth = (1/alpha) * log(sum(exp(alpha * H_i)))
    
    Then f = -D / M_smooth.
    df/dz = -(dD/dz * M_smooth - D * dM_smooth/dz) / M_smooth^2
    
    dH(q)/dz:
    H(q) = -sum p_i log2 p_i.
    p_i = sum_{k in bin i} w_k.
    w_k = exp(z_k) / S.
    dp_i/dz_j = (delta_{ij} w_j - p_j w_j) ... wait.
    
    Let's stick to finite differences for L-BFGS to avoid implementation errors in the gradient, 
    but limit the number of iterations or use a very fast local search?
    Actually, 100 variables is small. Finite difference L-BFGS might be slow but acceptable if we limit iterations.
    Or, we can just use COBYLA which doesn't need gradients? No, L-BFGS-B is better for smooth-ish functions.
    
    Let's try providing the gradient using the smooth max approximation.
    """
    alpha = 50.0 # Smoothing parameter
    
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    w = exp_z / exp_z.sum()
    
    idx_a, idx_b, idx_c, idx_d = _OBJ_CACHE['idxs']
    la = idx_a.max() + 1
    lb = idx_b.max() + 1
    lc = idx_c.max() + 1
    ld = idx_d.max() + 1
    
    qa = np.bincount(idx_a, weights=w, minlength=la)
    qb = np.bincount(idx_b, weights=w, minlength=lb)
    qc = np.bincount(idx_c, weights=w, minlength=lc)
    qd = np.bincount(idx_d, weights=w, minlength=ld)
    
    H_a = _entropy(qa)
    H_b = _entropy(qb)
    H_c = _entropy(qc)
    H_d = _entropy(qd)
    
    D = H_c
    
    # Smooth max of H_a, H_b, H_d
    Hs = np.array([H_a, H_b, H_d])
    max_H = np.max(Hs)
    if max_H < 1e-9:
        return np.zeros(100)
        
    # M_smooth = max_H + (1/alpha) * log(sum(exp(alpha*(H_i - max_H))))
    # Let u_i = alpha * (H_i - max_H)
    u = alpha * (Hs - max_H)
    sum_exp_u = np.sum(np.exp(u))
    M_smooth = max_H + (1.0 / alpha) * np.log(sum_exp_u)
    
    if M_smooth < 1e-9:
        return np.zeros(100)
        
    # Gradient of M_smooth w.r.t H_i:
    # dM/dH_i = (1/alpha) * (1/sum_exp_u) * sum_exp_u * alpha * exp(u_i) * (1/alpha) ?? No.
    # M = max_H + (1/alpha) log(sum exp(alpha(H_i - max_H)))
    # dM/dH_i = (1/alpha) * (1/sum) * sum * exp(alpha(H_i - max_H)) * alpha = exp(u_i) / sum_exp_u
    # Note: max_H depends on H_i, but in the log-sum-exp formulation, the max is just a shift.
    # The derivative of the LSE part is the softmax of the scaled values.
    # So dM/dH_i = softmax(u)_i
    
    p_m = np.exp(u) / sum_exp_u
    
    # Gradient of D (H_c) w.r.t z
    # H_c = -sum_c p_c log2 p_c
    # dH_c/dz_j = sum_c (dp_c/dz_j) * (-log2 p_c - 1/ln(2))
    # dp_c/dz_j = w_j * (1 if j in bin c else 0) - w_j * p_c
    # Let B_c be the set of indices in bin c.
    # dH_c/dz_j = w_j * ( -log2 p_{c(j)} - 1/ln(2) - p_{c(j)} )
    # where c(j) is the bin index for cell j.
    
    # We need p_{c(j)} for each cell j.
    # idx_c maps cell j to bin index.
    p_c_for_cell = qc[idx_c]
    
    # term = -log2(p) - 1/ln(2) - p
    # Handle p=0
    mask_p = p_c_for_cell > 1e-300
    term_c = np.zeros(100)
    term_c[mask_p] = -np.log2(p_c_for_cell[mask_p]) - 1.0/np.log(2) - p_c_for_cell[mask_p]
    
    grad_D = w * term_c
    
    # Gradient of f = -D / M_smooth
    # f = -D * M_smooth^{-1}
    # df/dz = - (grad_D * M_smooth + D * grad_M_smooth * (-1) * M_smooth^{-2} * M_smooth) ??
    # df/dz = - (grad_D * M_smooth - D * grad_M_smooth) / M_smooth^2
    
    # We need grad_M_smooth w.r.t z.
    # M_smooth = sum_i H_i * p_m_i (effectively, in the differential sense)
    # dM_smooth/dz = sum_i (dH_i/dz) * p_m_i
    
    # We need grad_H_a, grad_H_b, grad_H_d.
    # Similar to grad_D but for respective bins.
    
    p_a_for_cell = qa[idx_a]
    p_b_for_cell = qb[idx_b]
    p_d_for_cell = qd[idx_d]
    
    mask_a = p_a_for_cell > 1e-300
    term_a = np.zeros(100)
    term_a[mask_a] = -np.log2(p_a_for_cell[mask_a]) - 1.0/np.log(2) - p_a_for_cell[mask_a]
    grad_H_a = w * term_a
    
    mask_b = p_b_for_cell > 1e-300
    term_b = np.zeros(100)
    term_b[mask_b] = -np.log2(p_b_for_cell[mask_b]) - 1.0/np.log(2) - p_b_for_cell[mask_b]
    grad_H_b = w * term_b
    
    mask_d = p_d_for_cell > 1e-300
    term_d = np.zeros(100)
    term_d[mask_d] = -np.log2(p_d_for_cell[mask_d]) - 1.0/np.log(2) - p_d_for_cell[mask_d]
    grad_H_d = w * term_d
    
    # grad_M_smooth = p_m[0]*grad_H_a + p_m[1]*grad_H_b + p_m[2]*grad_H_d
    grad_M = p_m[0] * grad_H_a + p_m[1] * grad_H_b + p_m[2] * grad_H_d
    
    grad_f = - (grad_D * M_smooth - D * grad_M) / (M_smooth * M_smooth)
    
    return grad_f

_OBJ_CACHE = {}

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    idx_a, idx_b, idx_c, idx_d = _build_idx()
    _OBJ_CACHE['idxs'] = (idx_a, idx_b, idx_c, idx_d)
    
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin
    
    best_val = -1e9
    best_P = np.full((10, 10), 1.0 / 100.0)
    
    # Initial best: Uniform
    val_u = _value_from_P(best_P, idx_a, idx_b, idx_c, idx_d)
    if val_u > best_val:
        best_val = val_u
        best_P = best_P.copy()
        
    # 1. Differential Evolution (Global Search)
    # Bounds: log-weights can be anything, but let's constrain to [-20, 20] to keep probabilities reasonable
    # and avoid under/overflow issues in softmax.
    bounds = [(-20.0, 20.0) for _ in range(100)]
    
    # DE requires a callable objective.
    # We will run DE for a limited number of generations or until time limit.
    # maxiter is generations. popsize is multiplier.
    
    # To monitor time, we can't easily interrupt DE. 
    # We can run it in a thread? No, keep it simple.
    # We estimate time per generation.
    # Let's run DE with a conservative maxiter.
    # 100 vars, popsize 15 -> 1500 evals/gen.
    # If 1 eval is ~1ms, 1 gen is 1.5s. 20 gens = 30s.
    # Let's try 30 gens.
    
    try:
        # Use a seed for reproducibility
        res_de = differential_evolution(
            _objective_de, 
            bounds, 
            maxiter=30, 
            popsize=15, 
            tol=1e-5, 
            seed=42,
            init='sobol'
        )
        
        if res_de.success:
            z_de = res_de.x
            # Convert to P
            z_shifted = z_de - np.max(z_de)
            exp_z = np.exp(z_shifted)
            w = exp_z / exp_z.sum()
            P_de = w.reshape(10, 10)
            val_de = _value_from_P(P_de, idx_a, idx_b, idx_c, idx_d)
            
            if val_de > best_val:
                best_val = val_de
                best_P = P_de.copy()
                
        # Check time
        if time.time() > deadline - 100.0:
             P_final = _finalize(best_P, idx_a, idx_b, idx_c, idx_d)
             return P_final

    except Exception:
        pass

    # 2. Polishing with L-BFGS-B (Local Refinement)
    # Start from the best z found by DE, and also from uniform z=0.
    
    candidates_z = []
    
    # Uniform
    candidates_z.append(np.zeros(100))
    
    # From DE result
    # We need to recover z from P_de.
    # Since we stored P_de, we can just use the z_de from res_de if available.
    # But res_de is out of scope if exception. Let's re-derive or store.
    # Let's store z_de in a variable outside the try block?
    # Better: just re-run a short DE or use the best_P found.
    
    # Let's take the current best_P and convert to z.
    # z_j = log(P_j / sum(P)) ... no, softmax is not invertible uniquely.
    # But if we assume the support is the same, we can set z_j = log(P_j) for P_j > 0, and very small for others.
    # This is an approximation.
    
    # Alternatively, just run L-BFGS from z=0 (uniform) and from a few random starts.
    # And from the DE result if we kept it.
    
    # Let's keep the DE result z.
    z_de_stored = None
    try:
        # Re-run DE quickly? No, expensive.
        # We'll just rely on the fact that DE found a good P.
        # We can convert best_P to a z vector for L-BFGS start.
        # z_j = log(best_P_j + 1e-10)
        # This puts weight on the support.
        z_from_P = np.log(best_P.ravel() + 1e-10)
        candidates_z.append(z_from_P)
    except Exception:
        pass
    
    # Add a few random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        z_rand = rng.normal(0, 2, 100)
        candidates_z.append(z_rand)
        
    for z_start in candidates_z:
        if time.time() > deadline - 20.0:
            break
            
        try:
            res_lbfgs = minimize(
                _objective_lbfgs, 
                z_start, 
                method='L-BFGS-B', 
                jac=_grad_lbfgs,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            z_opt = res_lbfgs.x
            z_shifted = z_opt - np.max(z_opt)
            exp_z = np.exp(z_shifted)
            w = exp_z / exp_z.sum()
            P_opt = w.reshape(10, 10)
            val_opt = _value_from_P(P_opt, idx_a, idx_b, idx_c, idx_d)
            
            if val_opt > best_val:
                best_val = val_opt
                best_P = P_opt.copy()
                
        except Exception:
            continue

    P_final = _finalize(best_P, idx_a, idx_b, idx_c, idx_d)
    return P_final

# EVOLVE-BLOCK-END
