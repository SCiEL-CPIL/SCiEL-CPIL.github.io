import numpy as np
from scipy.optimize import minimize
import time

# Constants
N_SITES = 6
DIM = 2 ** N_SITES
ANISOTROPY = 0.8
FIELD = 0.3
STAGES = 12
EVAL_TIMES = (0.25, 0.5, 1.0)
DEADLINE = 140.0  # Safety margin before 150s kill

def build_hamiltonians():
    """Build A and B for the 6-site XXZ chain."""
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

    A = np.zeros((DIM, DIM), dtype=complex)
    B = np.zeros((DIM, DIM), dtype=complex)

    # Bonds j=0 to n-2
    for j in range(n - 1):
        term = (site_op(sx, j) @ site_op(sx, j+1) +
                site_op(sy, j) @ site_op(sy, j+1) +
                ANISOTROPY * site_op(sz, j) @ site_op(sz, j+1))
        if j % 2 == 0:
            B += term
        else:
            A += term

    # Field split
    for j in range(n):
        half_field = FIELD / 2.0
        term = half_field * site_op(sz, j)
        A += term
        B += term

    return A, B

# Precompute Hamiltonians
A, B = build_hamiltonians()
AB = A + B

def normalize_coefficients(a, b):
    """Ensure sums are 1 and values in [-50, 50]."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    # Normalize sums to 1
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
        
    # Clip to bounds
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Re-normalize after clipping
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
        
    return a, b

def eval_coefficients(a, b):
    """Compute mean spectral-norm error."""
    a = np.array(a)
    b = np.array(b)
    total_error = 0.0
    for t in EVAL_TIMES:
        U = np.eye(DIM, dtype=complex)
        for k in range(STAGES):
            U = U @ np.linalg.expm(-1j * b[k] * t * B)
            U = U @ np.linalg.expm(-1j * a[k] * t * A)
        U_exact = np.linalg.expm(-1j * t * AB)
        diff = U - U_exact
        # Spectral norm
        s = np.linalg.svd(diff, compute_uv=False)
        total_error += s[0]
    return total_error / len(EVAL_TIMES)

def generate_candidates():
    """Generate a portfolio of candidate (a, b) lists."""
    candidates = []
    
    # 1. Uniform
    a0 = np.ones(STAGES) / STAGES
    b0 = np.ones(STAGES) / STAGES
    candidates.append((a0, b0))
    
    # 2. Strang Splitting (2nd order) repeated
    # Pattern: 0.5, 0.5 for a; 1, 0 for b per substep
    # With 12 stages, we can do 6 substeps of Strang
    a_strang = []
    b_strang = []
    for _ in range(6):
        a_strang.extend([1.0/12.0, 1.0/12.0])
        b_strang.extend([1.0/6.0, 0.0])
    candidates.append((np.array(a_strang), np.array(b_strang)))
    
    # 3. 4th Order Suzuki-Trotter Coefficients
    # Standard 4th order coefficients for 4 stages:
    # a1=a4=1/(2(4-4^0.25)), a2=a3=1/(2(1-4^0.25))
    # b1=b4=1/(4-4^0.25), b2=b3=1/(1-4^0.25)
    # We have 12 stages. We can replicate the 4-stage pattern 3 times.
    c4 = 1.0 / (4.0 - 4.0**0.25)
    a4_base = [c4, 0.5 - c4, 0.5 - c4, c4]
    b4_base = [2.0*c4 - 0.5, 0.5 - 2.0*c4, 0.5 - 2.0*c4, 2.0*c4 - 0.5]
    
    a4 = []
    b4 = []
    for _ in range(3):
        a4.extend([x/3.0 for x in a4_base])
        b4.extend([x/3.0 for x in b4_base])
    candidates.append((np.array(a4), np.array(b4)))
    
    # 4. Random Symmetric Candidates
    np.random.seed(42)
    for _ in range(20):
        a_half = np.random.uniform(-0.2, 0.2, 6)
        b_half = np.random.uniform(-0.2, 0.2, 6)
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        a_full, b_full = normalize_coefficients(a_full, b_full)
        candidates.append((a_full, b_full))
        
    return candidates

def objective(x):
    """Objective function for Nelder-Mead."""
    a_raw = x[:STAGES]
    b_raw = x[STAGES:]
    a, b = normalize_coefficients(a_raw, b_raw)
    try:
        return eval_coefficients(a, b)
    except Exception:
        return 1e10

def coefficients(stages):
    start_time = time.time()
    
    # Generate candidates
    candidates = generate_candidates()
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    # Evaluate candidates
    for a, b in candidates:
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            err = eval_coefficients(a, b)
            if err < best_err:
                best_err = err
                best_a = a
                best_b = b
        except Exception:
            continue
            
    if best_a is None:
        best_a = np.ones(STAGES) / STAGES
        best_b = np.ones(STAGES) / STAGES
        best_err = eval_coefficients(best_a, best_b)
        
    # Local Refinement with Nelder-Mead
    if time.time() - start_time < DEADLINE - 10:
        x0 = np.concatenate([best_a, best_b])
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_err:
                a_opt = res.x[:STAGES]
                b_opt = res.x[STAGES:]
                a_opt, b_opt = normalize_coefficients(a_opt, b_opt)
                best_a = a_opt
                best_b = b_opt
                best_err = res.fun
        except Exception:
            pass
            
    # Final check and return
    best_a, best_b = normalize_coefficients(best_a, best_b)
    
    # Ensure types are lists of floats
    return list(best_a), list(best_b)
