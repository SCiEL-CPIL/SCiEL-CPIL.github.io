"""Optimized product formula design for Heisenberg-chain time evolution."""
import numpy as np
import math
import time
import sys

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
    
    # Uniform
    a0 = [1.0/STAGES] * STAGES
    b0 = [1.0/STAGES] * STAGES
    candidates.append((a0[:], b0[:]))
    
    # 4th-order Suzuki-Trotter based patterns
    c1 = 1.0 / (4.0 - 4.0**0.25)
    c2 = 0.5 - c1
    c3 = 2.0*c1 - 0.5
    
    # Symmetric 4th-order patterns repeated
    # Pattern 1: Standard 4th order 6-stage, repeated 2x
    a_4th = [c1, c2, 0.0, c2, c1, 0.0, c1, c2, 0.0, c2, c1, 0.0]
    b_4th = [0.0, c3, 1.0-c3, c3, 0.0, 0.0, 0.0, c3, 1.0-c3, c3, 0.0, 0.0]
    # Normalize
    sa = sum(a_4th)
    sb = sum(b_4th)
    if abs(sa) > 1e-10: a_4th = [x/sa for x in a_4th]
    if abs(sb) > 1e-10: b_4th = [x/sb for x in b_4th]
    candidates.append((a_4th, b_4th))
    
    # Symmetric patterns with various weight distributions
    for name, a_w, b_w in [
        ("endpoint_heavy", [0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.08, 0.12],
                         [0.12, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.08, 0.12]),
        ("center_heavy", [0.04, 0.05, 0.06, 0.08, 0.12, 0.14, 0.14, 0.12, 0.08, 0.06, 0.05, 0.04],
                         [0.04, 0.05, 0.06, 0.08, 0.12, 0.14, 0.14, 0.12, 0.08, 0.06, 0.05, 0.04]),
        ("alternating", [0.08, 0.04, 0.08, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04, 0.08],
                         [0.08, 0.04, 0.08, 0.04, 0.08, 0.04, 0.04, 0.04, 0.04, 0.08, 0.04, 0.08]),
    ]:
        sa = sum(a_w)
        sb = sum(b_w)
        a_norm = [x/sa for x in a_w]
        b_norm = [x/sb for x in b_w]
        candidates.append((a_norm[:], b_norm[:]))
    
    # Cosine-weighted symmetric candidates
    for amp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        weights = [1.0 + amp * math.cos(2 * math.pi * j / (STAGES - 1)) for j in range(STAGES)]
        sw = sum(weights)
        a_sym = [w / sw for w in weights]
        b_sym = [w / sw for w in weights]
        candidates.append((a_sym[:], b_sym[:]))
    
    # Sine-weighted symmetric candidates
    for amp in [0.1, 0.2, 0.3, 0.4]:
        weights = [1.0 + amp * math.sin(math.pi * j / (STAGES - 1)) for j in range(STAGES)]
        sw = sum(weights)
        a_sym = [w / sw for w in weights]
        b_sym = [w / sw for w in weights]
        candidates.append((a_sym[:], b_sym[:]))
    
    # Random symmetric candidates
    np.random.seed(42)
    for i in range(150):
        a_half = np.random.uniform(-0.5, 0.5, 6)
        b_half = np.random.uniform(-0.5, 0.5, 6)
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
    
    # Random asymmetric candidates
    for i in range(50):
        a_rand = np.random.uniform(-0.3, 0.3, 12)
        b_rand = np.random.uniform(-0.3, 0.3, 12)
        sa = np.sum(a_rand)
        sb = np.sum(b_rand)
        if abs(sa) > 1e-10:
            a_rand = a_rand / sa
        if abs(sb) > 1e-10:
            b_rand = b_rand / sb
        a_rand = np.clip(a_rand, -50.0, 50.0)
        b_rand = np.clip(b_rand, -50.0, 50.0)
        sa = np.sum(a_rand)
        sb = np.sum(b_rand)
        if abs(sa) > 1e-10:
            a_rand = a_rand / sa
        if abs(sb) > 1e-10:
            b_rand = b_rand / sb
        candidates.append((a_rand.tolist(), b_rand.tolist()))
    
    return candidates

def _make_objective_symmetric():
    """Create a symmetric objective function for 6D parameter space."""
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
    return objective_sym

def _make_objective_asymmetric():
    """Create an asymmetric objective function for 12D parameter space."""
    def objective_asym(x):
        a_full = x[:12]
        b_full = x[12:24]
        a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
        try:
            return eval_coefficients_fast(a_full, b_full)
        except Exception:
            return 1e10
    return objective_asym

def optimize_coefficients():
    start_time = time.time()
    
    candidates = generate_candidates()
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    # Phase 0: Evaluate candidates
    for i, (a_cand, b_cand) in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 50:
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
    
    # Phase 1: L-BFGS-B on symmetric 6D parameter space with multiple restarts
    if time.time() - start_time < DEADLINE - 40:
        from scipy.optimize import minimize
        
        obj_sym = _make_objective_symmetric()
        
        a0_half = np.array(best_a[:6])
        b0_half = np.array(best_b[:6])
        x0_sym = np.concatenate([a0_half, b0_half])
        
        # Try multiple random restarts with L-BFGS-B
        np.random.seed(123)
        n_restarts = 20
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 40:
                break
            
            if r == 0:
                x_start = x0_sym
            else:
                # Random perturbation
                scale = 0.05 + 0.02 * r
                perturbation = np.random.normal(0, scale, len(x0_sym))
                x_start = x0_sym + perturbation
            
            options = {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxcor': 10}
            try:
                result = minimize(obj_sym, x_start, method='L-BFGS-B', options=options)
                if result.fun < best_err:
                    a_half = result.x[:6]
                    b_half = result.x[6:12]
                    a_full = np.concatenate([a_half, a_half[::-1]])
                    b_full = np.concatenate([b_half, b_half[::-1]])
                    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                    best_err = result.fun
                    best_a = a_full
                    best_b = b_full
                    x0_sym = result.x.copy()
            except Exception:
                pass
    
    # Phase 2: Asymmetric refinement using L-BFGS-B on full 12D space
    if time.time() - start_time < DEADLINE - 20:
        from scipy.optimize import minimize
        
        obj_asym = _make_objective_asymmetric()
        
        a0_full = np.array(best_a)
        b0_full = np.array(best_b)
        x0_asym = np.concatenate([a0_full, b0_full])
        
        options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxcor': 15}
        try:
            result = minimize(obj_asym, x0_asym, method='L-BFGS-B', options=options)
            if result.fun < best_err:
                a_full = result.x[:12]
                b_full = result.x[12:24]
                a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                best_err = result.fun
                best_a = a_full
                best_b = b_full
        except Exception:
            pass
    
    # Phase 3: Final polish with very tight tolerances
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        obj_asym = _make_objective_asymmetric()
        
        a0_full = np.array(best_a)
        b0_full = np.array(best_b)
        x0_asym = np.concatenate([a0_full, b0_full])
        
        options = {'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxcor': 20}
        try:
            result = minimize(obj_asym, x0_asym, method='L-BFGS-B', options=options)
            if result.fun < best_err:
                a_full = result.x[:12]
                b_full = result.x[12:24]
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
