"""Bootstrap portfolio: generate varied high-order splitting candidates and keep the best."""
import numpy as np
from numpy.linalg import norm as npnorm
import math
import time

DEADLINE = 145.0  # safety margin before 150s wall-clock kill
EVAL_TIMES = (0.25, 0.5, 1.0)
N_SITES = 6
ANISOTROPY = 0.8
FIELD = 0.3
STAGES = 12

# Precompute A and B (even/odd bond halves with half field each)
def build_hamiltonians():
    n = N_SITES
    # XXZ Heisenberg chain: sum over bonds j=0..n-2 of (1/2)(Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + anis* Sz_j Sz_{j+1}) + field * sum_j Sz_j
    # Actually standard Heisenberg/XXZ: H = sum_{j=0}^{n-2} [Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + gamma Sz_j Sz_{j+1}] + h sum_j Sz_j
    # With anisotropy gamma=0.8 and field h=0.3
    # Pauli matrices
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    zero = np.array([[0, 0], [0, 0]], dtype=complex)

    # Helper to build operator on site j
    def site_op(op, j):
        ops = [zero] * n
        ops[j] = op
        result = np.array([[1.0+0j]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((2**n, 2**n), dtype=complex)
    B = np.zeros((2**n, 2**n), dtype=complex)

    # Bonds: j = 0..n-2
    for j in range(n - 1):
        term = (site_op(sx, j) @ site_op(sx, j+1) +
                site_op(sy, j) @ site_op(sy, j+1) +
                ANISOTROPY * site_op(sz, j) @ site_op(sz, j+1))
        if j % 2 == 0:
            B += term  # even bonds (0, 2, ...) go to B
        else:
            A += term  # odd bonds (1, 3, ...) go to A

    # Uniform field split: half in A, half in B
    for j in range(n):
        half_field = FIELD / 2.0
        term = half_field * site_op(sz, j)
        A += term
        B += term

    return A, B

# Cache hamiltonians and their eigen-decompositions
_HAM_CACHE = None
_EIG_CACHE = None

def get_hamiltonians():
    global _HAM_CACHE
    if _HAM_CACHE is None:
        _HAM_CACHE = build_hamiltonians()
    return _HAM_CACHE

def get_eigdecomps():
    """Cache eigen-decompositions of A, B, and A+B for fast expm."""
    global _EIG_CACHE
    if _EIG_CACHE is None:
        A, B = get_hamiltonians()
        # A and B are Hermitian, so use eigh
        evals_A, evecs_A = np.linalg.eigh(A)
        evals_B, evecs_B = np.linalg.eigh(B)
        AB = A + B
        evals_AB, evecs_AB = np.linalg.eigh(AB)
        _EIG_CACHE = (evals_A, evecs_A, evals_B, evecs_B, evals_AB, evecs_AB)
    return _EIG_CACHE

def fast_expm_hermitian(evals, evecs, x):
    """Compute exp(x * H) where H is Hermitian with known eigen-decomp.
    exp(x*H) = V @ diag(exp(x * evals)) @ V^H
    """
    phase = np.exp(x * evals)
    return (evecs * phase) @ evecs.conj().T

def eval_coefficients_fast(a, b):
    """Compute mean spectral-norm error over EVAL_TIMES using cached eigen-decomps."""
    evals_A, evecs_A, evals_B, evecs_B, evals_AB, evecs_AB = get_eigdecomps()
    total_dim = 2 ** N_SITES
    total_error = 0.0
    n_times = len(EVAL_TIMES)
    for t in EVAL_TIMES:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(total_dim, dtype=complex)
        for k in range(STAGES):
            U = U @ fast_expm_hermitian(evals_B, evecs_B, -1j * b[k] * t)
            U = U @ fast_expm_hermitian(evals_A, evecs_A, -1j * a[k] * t)
        # Exact propagator
        U_exact = fast_expm_hermitian(evals_AB, evecs_AB, -1j * t)
        # Spectral norm of difference
        diff = U - U_exact
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        total_error += s[0]
    return total_error / n_times

def eval_coefficients(a, b):
    """Compute mean spectral-norm error over EVAL_TIMES."""
    return eval_coefficients_fast(a, b)

def normalize_coefficients(a, b):
    """Ensure sums are exactly 1 and values are in [-50, 50]."""
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    else:
        a = [1.0/STAGES] * STAGES
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    else:
        b = [1.0/STAGES] * STAGES
    # Clip to [-50, 50] and re-normalize
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    else:
        a = [1.0/STAGES] * STAGES
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    else:
        b = [1.0/STAGES] * STAGES
    return a, b

def generate_symmetric_candidates():
    """Generate a portfolio of candidate (a, b) lists from known splitting schemes."""
    candidates = []
    
    # Baseline: uniform
    a0 = [1.0/STAGES] * STAGES
    b0 = [1.0/STAGES] * STAGES
    candidates.append((a0[:], b0[:]))
    
    # Symmetric patterns: for symmetric splitting, a and b sequences are palindromes
    # and the middle stage has a middle coefficient.
    # With 12 stages, symmetric means a[0]=a[11], a[1]=a[10], ..., a[5]=a[6]
    # and same for b.
    
    # Fourth-order Suzuki-Trotter-like patterns
    # Known 4th-order coefficients for 2-subproblem splitting:
    # Standard 4th order: a1=a4=1/(2(4-4^(1/4))), a2=a3=1/(2(1-4^(1/4)))
    # b1=b4=1/(4-4^(1/4)), b2=b3=1/(1-4^(1/4)) ... 
    # Actually let's use well-known 4th-order coefficients.
    
    # Dubrovin 4th order for 2 parts:
    c4 = 1.0 / (4.0 - 4.0**(0.25))
    a4_vals = [c4, 0.5 - c4, 0.5 - c4, c4]
    b4_vals = [2.0*c4 - 0.5, 0.5 - 2.0*c4, 0.5 - 2.0*c4, 2.0*c4 - 0.5]
    
    # Replicate to 12 stages by repeating the 4-stage pattern 3 times
    for rep in range(1, 4):
        a_cand = []
        b_cand = []
        for _ in range(rep):
            a_cand.extend([x/rep for x in a4_vals])
            b_cand.extend([x/rep for x in b4_vals])
        if len(a_cand) <= STAGES:
            # Pad with zeros if needed
            while len(a_cand) < STAGES:
                a_cand.append(0.0)
                b_cand.append(0.0)
            candidates.append((a_cand[:], b_cand[:]))
    
    # 6th-order symmetric coefficients (Sugano, or known 6th-order splitting)
    # One set of 6th-order coefficients for 2-part splitting:
    # From literature: 
    # Let's use a known 6th-order scheme with 6 stages per "block"
    # Symmetric 6-stage 6th-order coefficients (one known set):
    a6 = [1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0]
    b6 = [1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0, 1/12.0]
    # Actually uniform is not 6th order. Let me use known coefficients.
    # 6th order Dubrovin-like:
    # For now, include a symmetric random-ish portfolio.
    
    # Generate symmetric random candidates
    np.random.seed(42)
    n_random = 100
    for i in range(n_random):
        # Symmetric: 6 free a values, 6 free b values
        a_half = np.random.uniform(-0.3, 0.3, 6)
        b_half = np.random.uniform(-0.3, 0.3, 6)
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        # Normalize
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
        # Clip
        a_full = np.clip(a_full, -50.0, 50.0)
        b_full = np.clip(b_full, -50.0, 50.0)
        sa = np.sum(a_full)
        sb = np.sum(b_full)
        if abs(sa) > 1e-10:
            a_full = a_full / sa
        if abs(sb) > 1e-10:
            b_full = b_full / sb
        candidates.append((a_full.tolist(), b_full.tolist()))
    
    # Nelder-Mead local refinement starting from best symmetric candidates
    from scipy.optimize import minimize
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    start_time = time.time()
    
    # First pass: evaluate all candidates quickly
    for i, (a_cand, b_cand) in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 30:
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
        # Fallback to seed-like
        best_a = [1.0/STAGES] * STAGES
        best_b = [1.0/STAGES] * STAGES
    
    # Try local optimization with Nelder-Mead on best candidate
    if time.time() - start_time < DEADLINE - 15:
        a0 = np.array(best_a)
        b0 = np.array(best_b)
        
        def objective(x):
            a_opt = x[:STAGES].tolist()
            b_opt = x[STAGES:].tolist()
            a_opt, b_opt = normalize_coefficients(a_opt, b_opt)
            try:
                return eval_coefficients_fast(a_opt, b_opt)
            except Exception:
                return 1e10
        
        x0 = np.concatenate([a0, b0])
        
        options = {'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
        try:
            result = minimize(objective, x0, method='Nelder-Mead', options=options)
            if result.fun < best_err:
                a_opt, b_opt = normalize_coefficients(result.x[:STAGES].tolist(), result.x[STAGES:].tolist())
                best_err = result.fun
                best_a = a_opt
                best_b = b_opt
        except Exception:
            pass
    
    # Second round: try optimization from multiple starting points if time permits
    remaining_time = DEADLINE - 15 - (time.time() - start_time)
    if remaining_time > 20:
        # Try from top few candidates
        scored_candidates = []
        for a_cand, b_cand in candidates:
            a_cand, b_cand = normalize_coefficients(a_cand[:], b_cand[:])
            try:
                err = eval_coefficients_fast(a_cand, b_cand)
                scored_candidates.append((err, a_cand, b_cand))
            except Exception:
                continue
        
        scored_candidates.sort(key=lambda x: x[0])
        
        for err_c, a_c, b_c in scored_candidates[:5]:
            if time.time() - start_time > DEADLINE - 10:
                break
            if err_c >= best_err:
                break
            
            a0 = np.array(a_c)
            b0 = np.array(b_c)
            
            def objective2(x):
                a_opt = x[:STAGES].tolist()
                b_opt = x[STAGES:].tolist()
                a_opt, b_opt = normalize_coefficients(a_opt, b_opt)
                try:
                    return eval_coefficients_fast(a_opt, b_opt)
                except Exception:
                    return 1e10
            
            x0 = np.concatenate([a0, b0])
            
            options = {'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            try:
                result = minimize(objective2, x0, method='Nelder-Mead', options=options)
                if result.fun < best_err:
                    a_opt, b_opt = normalize_coefficients(result.x[:STAGES].tolist(), result.x[STAGES:].tolist())
                    best_err = result.fun
                    best_a = a_opt
                    best_b = b_opt
            except Exception:
                pass
    
    return best_a, best_b, best_err

def coefficients(stages):
    """Return optimized coefficients for the 12-stage product formula."""
    start_time = time.time()
    
    try:
        a_best, b_best, err = generate_symmetric_candidates()
    except Exception:
        # Fallback to seed
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Just return what we have, normalized
        a_best, b_best = normalize_coefficients(a_best, b_best)
        return a_best, b_best
    
    return a_best, b_best
