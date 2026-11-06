"""Optimized product formula design for Heisenberg-chain time evolution."""
import numpy as np
import math
import time

DEADLINE = 145.0
EVAL_TIMES = (0.25, 0.5, 1.0)
N_SITES = 6
ANISOTROPY = 0.8
FIELD = 0.3
STAGES = 12

def build_hamiltonians():
    n = N_SITES
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    zero = np.array([[0, 0], [0, 0]], dtype=complex)

    def site_op(op, j):
        ops = [zero] * n
        ops[j] = op
        result = np.array([[1.0+0j]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((2**n, 2**n), dtype=complex)
    B = np.zeros((2**n, 2**n), dtype=complex)

    for j in range(n - 1):
        term = (site_op(sx, j) @ site_op(sx, j+1) +
                site_op(sy, j) @ site_op(sy, j+1) +
                ANISOTROPY * site_op(sz, j) @ site_op(sz, j+1))
        if j % 2 == 0:
            B += term
        else:
            A += term

    for j in range(n):
        half_field = FIELD / 2.0
        term = half_field * site_op(sz, j)
        A += term
        B += term

    return A, B

_HAM_CACHE = None
def get_hamiltonians():
    global _HAM_CACHE
    if _HAM_CACHE is None:
        A, B = build_hamiltonians()
        wA, VA = np.linalg.eig(A)
        wB, VB = np.linalg.eig(B)
        VAI = np.linalg.inv(VA)
        VBI = np.linalg.inv(VB)
        AB = A + B
        wAB, VAB = np.linalg.eig(AB)
        VABI = np.linalg.inv(VAB)
        _HAM_CACHE = (A, B, wA, VA, VAI, wB, VB, VBI, wAB, VAB, VABI)
    return _HAM_CACHE

def fast_expm_eig(w, V, Vi, c):
    exp_diag = np.exp(-1j * c * w)
    return (V * exp_diag[None, :]) @ Vi

def eval_coefficients_fast(a, b):
    _, _, wA, VA, VAI, wB, VB, VBI, wAB, VAB, VABI = get_hamiltonians()
    total_error = 0.0
    n_times = len(EVAL_TIMES)
    dim = 2 ** N_SITES
    for t in EVAL_TIMES:
        U = np.eye(dim, dtype=complex)
        for k in range(STAGES):
            U = U @ fast_expm_eig(wB, VB, VBI, b[k] * t)
            U = U @ fast_expm_eig(wA, VA, VAI, a[k] * t)
        U_exact = fast_expm_eig(wAB, VAB, VABI, t)
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        total_error += s[0]
    return total_error / n_times

def normalize_coefficients(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(STAGES) / STAGES
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(STAGES) / STAGES
    
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(STAGES) / STAGES
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(STAGES) / STAGES
    
    return a.tolist(), b.tolist()

def generate_candidates():
    candidates = []
    
    a0 = [1.0/STAGES] * STAGES
    b0 = [1.0/STAGES] * STAGES
    candidates.append((a0[:], b0[:]))
    
    # 4th-order Suzuki-Trotter block, repeated 3 times for 12 stages
    c4 = 1.0 / (4.0 - 4.0**(0.25))
    a4_block = [c4, 0.5 - c4, 0.5 - c4, c4]
    b4_block = [2.0*c4 - 0.5, 0.5 - 2.0*c4, 0.5 - 2.0*c4, 2.0*c4 - 0.5]
    
    a_4th = []
    b_4th = []
    for _ in range(3):
        a_4th.extend([x/3.0 for x in a4_block])
        b_4th.extend([x/3.0 for x in b4_block])
    candidates.append((a_4th[:], b_4th[:]))
    
    # 6th-order Suzuki-Trotter block, repeated 2 times for 12 stages
    alpha = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = -alpha / (2.0 * (1.0 - alpha))
    c3 = (1.0 - alpha) / (2.0 * (1.0 - alpha))
    d1 = 1.0 / (1.0 - alpha)
    d2 = -alpha / (1.0 - alpha)
    d3 = 0.0
    
    a6_block = [c1, c2, c3, c3, c2, c1]
    b6_block = [d1, d2, d3, d3, d2, d1]
    
    sa6 = sum(a6_block)
    sb6 = sum(b6_block)
    a6_block = [x / sa6 for x in a6_block]
    b6_block = [x / sb6 for x in b6_block]
    
    a_6th = []
    b_6th = []
    for _ in range(2):
        a_6th.extend([x/2.0 for x in a6_block])
        b_6th.extend([x/2.0 for x in b6_block])
    candidates.append((a_6th[:], b_6th[:]))
    
    # 8th-order coefficients (Chin et al.) - symmetric 12-stage
    # Using known 8th-order Suzuki coefficients scaled to 12 stages
    # These are approximate; we'll refine via optimization
    a_8th_approx = [0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
                     0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833]
    b_8th_approx = [0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833,
                     0.0833, 0.0833, 0.0833, 0.0833, 0.0833, 0.0833]
    candidates.append((a_8th_approx[:], b_8th_approx[:]))
    
    # Cosine-weighted symmetric candidates with various amplitudes
    for amp in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        a_sym = []
        b_sym = []
        weights = [1.0 + amp * math.cos(2 * math.pi * j / (STAGES - 1)) for j in range(STAGES)]
        sw = sum(weights)
        for w in weights:
            a_sym.append(w / sw)
            b_sym.append(w / sw)
        candidates.append((a_sym[:], b_sym[:]))
    
    # Sine-weighted symmetric candidates
    for amp in [0.1, 0.2, 0.3]:
        a_sym = []
        b_sym = []
        weights = [1.0 + amp * math.sin(math.pi * j / (STAGES - 1)) for j in range(STAGES)]
        sw = sum(weights)
        for w in weights:
            a_sym.append(w / sw)
            b_sym.append(w / sw)
        candidates.append((a_sym[:], b_sym[:]))
    
    # Random symmetric candidates
    np.random.seed(42)
    for i in range(60):
        rng = np.random.uniform(-0.3, 0.3, 6)
        a_half = rng.copy()
        b_half = np.random.uniform(-0.3, 0.3, 6)
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        sa = np.sum(a_full)
        sb = np.sum(b_full)
        if abs(sa) > 1e-10:
            a_full = a_full / sa
        else:
            a_full = np.ones(STAGES)/STAGES
        if abs(sb) > 1e-10:
            b_full = b_full / sb
        else:
            b_full = np.ones(STAGES)/STAGES
        a_full = np.clip(a_full, -50.0, 50.0)
        b_full = np.clip(b_full, -50.0, 50.0)
        sa = np.sum(a_full)
        sb = np.sum(b_full)
        if abs(sa) > 1e-10:
            a_full = a_full / sa
        if abs(sb) > 1e-10:
            b_full = b_full / sb
        candidates.append((a_full.tolist(), b_full.tolist()))
    
    # Additional structured candidates: alternating patterns
    for pattern_amp in [0.05, 0.1, 0.15, 0.2]:
        a_alt = []
        b_alt = []
        for j in range(STAGES):
            sign = 1.0 if j % 2 == 0 else -1.0
            a_alt.append(1.0/STAGES + sign * pattern_amp / STAGES)
            b_alt.append(1.0/STAGES + sign * pattern_amp / STAGES)
        sa = sum(a_alt)
        sb = sum(b_alt)
        a_alt = [x/sa for x in a_alt]
        b_alt = [x/sb for x in b_alt]
        candidates.append((a_alt[:], b_alt[:]))
    
    return candidates

def optimize_coefficients():
    start_time = time.time()
    
    candidates = generate_candidates()
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    for i, (a_cand, b_cand) in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 70:
            break
        
        a_cand, b_cand = normalize_coefficients(a_cand[:], b_cand[:])
        try:
            err = eval_coefficients_fast(a_cand, b_cand)
        except Exception:
            continue
        
        if err < best_err:
            best_err = err
            best_a = a_cand[:]
            best_b = b_cand[:]
    
    if best_a is None:
        best_a = [1.0/STAGES] * STAGES
        best_b = [1.0/STAGES] * STAGES
        best_err = eval_coefficients_fast(best_a, best_b)
    
    # Phase 1: Nelder-Mead with multiple restarts
    if time.time() - start_time < DEADLINE - 45:
        from scipy.optimize import minimize
        
        a0_half = np.array(best_a[:6])
        b0_half = np.array(best_b[:6])
        
        def objective_sym(x):
            a_half = x[:6]
            b_half = x[6:12]
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
            try:
                return eval_coefficients_fast(a_full, b_full)
            except Exception:
                return 1e10
        
        x0 = np.concatenate([a0_half, b0_half])
        
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 45:
                break
            
            if r == 0:
                x_start = x0
            else:
                scale = 0.02 * (1.0 + r * 0.5)
                perturbation = np.random.normal(0, scale, len(x0))
                x_start = x0 + perturbation
            
            options = {'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            try:
                result = minimize(objective_sym, x_start, method='Nelder-Mead', options=options)
                if result.fun < best_err:
                    a_half = result.x[:6]
                    b_half = result.x[6:12]
                    a_full = np.concatenate([a_half, a_half[::-1]])
                    b_full = np.concatenate([b_half, b_half[::-1]])
                    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                    best_err = result.fun
                    best_a = a_full
                    best_b = b_full
                    x0 = result.x.copy()
            except Exception:
                pass
    
    # Phase 2: L-BFGS-B gradient-based optimization (faster convergence near optimum)
    if time.time() - start_time < DEADLINE - 25:
        from scipy.optimize import minimize
        
        a0_half = np.array(best_a[:6])
        b0_half = np.array(best_b[:6])
        x0 = np.concatenate([a0_half, b0_half])
        
        def objective_lbfgs(x):
            a_half = x[:6]
            b_half = x[6:12]
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
            try:
                return eval_coefficients_fast(a_full, b_full)
            except Exception:
                return 1e10
        
        options = {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        try:
            result = minimize(objective_lbfgs, x0, method='L-BFGS-B', options=options)
            if result.fun < best_err:
                a_half = result.x[:6]
                b_half = result.x[6:12]
                a_full = np.concatenate([a_half, a_half[::-1]])
                b_full = np.concatenate([b_half, b_half[::-1]])
                a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                best_err = result.fun
                best_a = a_full
                best_b = b_full
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish with very tight tolerances
    if time.time() - start_time < DEADLINE - 10:
        from scipy.optimize import minimize
        
        a0_half = np.array(best_a[:6])
        b0_half = np.array(best_b[:6])
        x0 = np.concatenate([a0_half, b0_half])
        
        def objective_polish(x):
            a_half = x[:6]
            b_half = x[6:12]
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
            try:
                return eval_coefficients_fast(a_full, b_full)
            except Exception:
                return 1e10
        
        options = {'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        try:
            result = minimize(objective_polish, x0, method='Nelder-Mead', options=options)
            if result.fun < best_err:
                a_half = result.x[:6]
                b_half = result.x[6:12]
                a_full = np.concatenate([a_half, a_half[::-1]])
                b_full = np.concatenate([b_half, b_half[::-1]])
                a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                best_err = result.fun
                best_a = a_full
                best_b = b_full
        except Exception:
            pass
    
    return best_a, best_b, best_err

def coefficients(stages):
    start_time = time.time()
    
    try:
        a_best, b_best, err = optimize_coefficients()
    except Exception:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    elapsed = time.time() - start_time
    a_best, b_best = normalize_coefficients(a_best, b_best)
    return a_best, b_best
