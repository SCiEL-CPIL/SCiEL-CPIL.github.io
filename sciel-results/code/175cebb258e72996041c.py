import numpy as np
import time
import math

# Constants
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill
EVAL_TIMES = (0.25, 0.5, 1.0)
N_SITES = 6
ANISOTROPY = 0.8
FIELD = 0.3
STAGES = 12

# --- Hamiltonian Construction ---

def build_hamiltonians():
    n = N_SITES
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
    """Compute exp(x * H) where H is Hermitian with known eigen-decomp."""
    phase = np.exp(x * evals)
    return (evecs * phase) @ evecs.conj().T

def eval_coefficients_fast(a, b):
    """Compute mean spectral-norm error over EVAL_TIMES using cached eigen-decomps."""
    evals_A, evecs_A, evals_B, evecs_B, evals_AB, evecs_AB = get_eigdecomps()
    total_dim = 2 ** N_SITES
    total_error = 0.0
    n_times = len(EVAL_TIMES)
    
    # Pre-compute the exponentials for each stage and time to save time?
    # No, coefficients change, so we must compute exp(-i * coeff * t * H) each time.
    # However, we can pre-compute the eigen-decomposition parts.
    
    for t in EVAL_TIMES:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(total_dim, dtype=complex)
        for k in range(STAGES):
            # Multiply by exp(-i b_k t B)
            U = U @ fast_expm_hermitian(evals_B, evecs_B, -1j * b[k] * t)
            # Multiply by exp(-i a_k t A)
            U = U @ fast_expm_hermitian(evals_A, evecs_A, -1j * a[k] * t)
        
        # Exact propagator
        U_exact = fast_expm_hermitian(evals_AB, evecs_AB, -1j * t)
        
        # Spectral norm of difference
        diff = U - U_exact
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        total_error += s[0]
    return total_error / n_times

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

def generate_high_order_candidates():
    """Generate a portfolio of candidate (a, b) lists from known high-order splitting schemes."""
    candidates = []
    
    # 1. Uniform (1st order)
    a0 = [1.0/STAGES] * STAGES
    b0 = [1.0/STAGES] * STAGES
    candidates.append((a0[:], b0[:]))
    
    # 2. Symmetric 4th-order Suzuki-Trotter
    # Standard 4th order coefficients for 2 parts (Suzuki 1976):
    # Let alpha = 1/(4 - 4^(1/4))
    # a1 = a4 = alpha
    # a2 = a3 = 0.5 - alpha
    # b1 = b4 = 2*alpha - 0.5
    # b2 = b3 = 0.5 - 2*alpha
    # We have 12 stages. We can replicate this 4-stage pattern 3 times.
    alpha = 1.0 / (4.0 - 4.0**(0.25))
    a4 = [alpha, 0.5 - alpha, 0.5 - alpha, alpha]
    b4 = [2*alpha - 0.5, 0.5 - 2*alpha, 0.5 - 2*alpha, 2*alpha - 0.5]
    
    # Scale to sum to 1 over 12 stages (each block sums to 1, so 3 blocks sum to 3, divide by 3)
    a4_scaled = [x/3.0 for x in a4]
    b4_scaled = [x/3.0 for x in b4]
    
    a_cand = a4_scaled * 3
    b_cand = b4_scaled * 3
    candidates.append((a_cand[:], b_cand[:]))
    
    # 3. Symmetric 6th-order (Dubrovin 1988 or similar)
    # 6th order symmetric scheme with 6 stages?
    # Actually, a common 6th order scheme has 6 a-coeffs and 5 b-coeffs (or vice versa).
    # For 12 stages, we can use a 6-stage 6th order block repeated twice.
    # Coefficients for 6th order (from e.g. Sugano 1977 or Dubrovin):
    # Let's use a known set. 
    # One set of 6th order coefficients (symmetric, 6 a's, 5 b's):
    # This is complex to derive manually. Let's use a "quasi-high-order" random symmetric search
    # centered around the 4th order solution.
    
    # 4. Random symmetric candidates centered around 4th order
    np.random.seed(42)
    for _ in range(50):
        # Perturb the 4th order solution
        a_cand = np.array(a4_scaled * 3)
        b_cand = np.array(b4_scaled * 3)
        
        # Add symmetric noise
        noise_a = np.random.normal(0, 0.05, 6)
        noise_b = np.random.normal(0, 0.05, 6)
        
        a_full = np.concatenate([a_cand[:6] + noise_a, a_cand[6:] + noise_a[::-1]])
        b_full = np.concatenate([b_cand[:6] + noise_b, b_cand[6:] + noise_b[::-1]])
        
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
            
        a_cand = a_full.tolist()
        b_cand = b_full.tolist()
        candidates.append((a_cand, b_cand))
        
    # 5. Pure random symmetric candidates
    for _ in range(50):
        a_half = np.random.uniform(-0.2, 0.2, 6)
        b_half = np.random.uniform(-0.2, 0.2, 6)
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
            
        a_cand = a_full.tolist()
        b_cand = b_full.tolist()
        candidates.append((a_cand, b_cand))
        
    return candidates

def optimize_from_start(start_a, start_b, time_limit):
    """
    Perform local optimization starting from (start_a, start_b).
    Uses a parameterization to handle the sum=1 constraint.
    """
    from scipy.optimize import minimize
    
    # Parameterization:
    # a[0..10] are free, a[11] = 1 - sum(a[0..10])
    # b[0..10] are free, b[11] = 1 - sum(b[0..10])
    
    a0 = np.array(start_a)
    b0 = np.array(start_b)
    
    # Initial point for optimizer
    x0 = np.concatenate([a0[:11], b0[:11]])
    
    def objective(x):
        a_opt = x[:11].tolist()
        b_opt = x[11:].tolist()
        
        # Calculate last coefficients
        a_opt.append(1.0 - sum(a_opt))
        b_opt.append(1.0 - sum(b_opt))
        
        # Check bounds
        if any(abs(v) > 50.0 for v in a_opt) or any(abs(v) > 50.0 for v in b_opt):
            return 1e10
            
        try:
            return eval_coefficients_fast(a_opt, b_opt)
        except Exception:
            return 1e10
            
    options = {'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=options)
        if result.fun < 1e9:
            a_opt = result.x[:11].tolist()
            b_opt = result.x[11:].tolist()
            a_opt.append(1.0 - sum(a_opt))
            b_opt.append(1.0 - sum(b_opt))
            # Final normalization and clipping
            a_opt, b_opt = normalize_coefficients(a_opt, b_opt)
            return a_opt, b_opt, result.fun
    except Exception:
        pass
        
    return start_a, start_b, float('inf')

def coefficients(stages):
    """Return optimized coefficients for the 12-stage product formula."""
    start_time = time.time()
    
    # 1. Generate candidates
    candidates = generate_high_order_candidates()
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    # 2. Evaluate candidates to find the best starting point for optimization
    # We don't need to evaluate all if time is tight, but let's evaluate the top few diverse ones
    # Sort by a simple heuristic or just evaluate the first 10
    
    # Let's evaluate the first 10 candidates to pick the best seed for optimization
    eval_candidates = []
    for i, (a_c, b_c) in enumerate(candidates[:10]):
        if time.time() - start_time > DEADLINE - 20:
            break
        a_c, b_c = normalize_coefficients(a_c, b_c)
        try:
            err = eval_coefficients_fast(a_c, b_c)
            eval_candidates.append((err, a_c, b_c))
        except Exception:
            continue
            
    if eval_candidates:
        eval_candidates.sort(key=lambda x: x[0])
        # Use the best 3 as starting points for optimization
        start_points = [ (a_c, b_c) for err, a_c, b_c in eval_candidates[:3] ]
    else:
        # Fallback to uniform
        start_points = [([1.0/STAGES]*STAGES, [1.0/STAGES]*STAGES)]
        
    # 3. Optimize from each starting point
    for a_start, b_start in start_points:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        a_opt, b_opt, err = optimize_from_start(a_start, b_start, DEADLINE - 10 - (time.time() - start_time))
        
        if err < best_err:
            best_err = err
            best_a = a_opt
            best_b = b_opt
            
    # 4. Final fallback
    if best_a is None:
        best_a = [1.0/STAGES] * STAGES
        best_b = [1.0/STAGES] * STAGES
        
    # 5. Deadline guard
    if time.time() - start_time > DEADLINE - 5:
        a_best, b_best = normalize_coefficients(best_a, best_b)
        return a_best, b_best
        
    return best_a, best_b
