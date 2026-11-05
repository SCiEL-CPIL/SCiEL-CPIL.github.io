import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

# Constants
DELTA = 0.8
FIELD = 0.3
N_SITES = 6
DIM = 2 ** N_SITES
TIMES = [0.25, 0.5, 1.0]

# Pauli matrices
I = np.eye(2)
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

def kron_all(*mats):
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result

def build_operators():
    """
    Constructs A (odd bonds + half field) and B (even bonds + half field).
    Sites are indexed 0 to 5.
    Odd bonds: (0,1), (2,3), (4,5)
    Even bonds: (1,2), (3,4)
    Field: h/2 * sum Sz_j
    """
    A = np.zeros((DIM, DIM), dtype=complex)
    B = np.zeros((DIM, DIM), dtype=complex)
    
    # Helper to add interaction term
    def add_interaction(op, site_i, site_j):
        # Term: Sx_i Sx_j + Sy_i Sy_j + Delta Sz_i Sz_j
        # We build the full operator by kron
        terms = []
        for k in range(N_SITES):
            if k == site_i or k == site_j:
                # Placeholder, we will sum the 3 terms separately
                pass
        # It's easier to construct the 3 terms separately
        # Term 1: Sx_i Sx_j
        mats = [I for _ in range(N_SITES)]
        mats[site_i] = Sx
        mats[site_j] = Sx
        op += kron_all(*mats)
        
        # Term 2: Sy_i Sy_j
        mats = [I for _ in range(N_SITES)]
        mats[site_i] = Sy
        mats[site_j] = Sy
        op += kron_all(*mats)
        
        # Term 3: Delta Sz_i Sz_j
        mats = [I for _ in range(N_SITES)]
        mats[site_i] = Sz
        mats[site_j] = Sz
        op += DELTA * kron_all(*mats)

    # Odd bonds for A
    add_interaction(A, 0, 1)
    add_interaction(A, 2, 3)
    add_interaction(A, 4, 5)
    
    # Even bonds for B
    add_interaction(B, 1, 2)
    add_interaction(B, 3, 4)
    
    # Field term: h/2 * sum Sz_j
    # Split equally between A and B
    field_op = np.zeros((DIM, DIM), dtype=complex)
    for j in range(N_SITES):
        mats = [I for _ in range(N_SITES)]
        mats[j] = Sz
        field_op += kron_all(*mats)
        
    half_field = (FIELD / 2.0) * field_op
    A += half_field
    B += half_field
    
    return A, B

# Pre-compute operators and exact propagators to save time during optimization
A, B = build_operators()
AB = A + B
EXACT_PROPAGATORS = {}
for t in TIMES:
    EXACT_PROPAGATORS[t] = expm(-1j * t * AB)

def compute_error(a, b):
    """
    Computes the mean spectral norm error.
    a, b are lists of 12 floats.
    """
    total_err = 0.0
    for t in TIMES:
        U = np.eye(DIM, dtype=complex)
        # Product formula: U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        # Order k=1..12. Standard matrix multiplication U = U @ next_factor
        # Since exp(-i b t B) and exp(-i a t A) are applied in sequence.
        # U_total = E_b1 E_a1 E_b2 E_a2 ... E_b12 E_a12
        # We can compute this iteratively.
        
        # Optimization: Pre-compute exp for all k? No, t is fixed, but a,b change.
        # We must compute exp for each step.
        
        for k in range(12):
            # exp(-i b_k t B)
            # We can cache expm for specific values? No, a,b are continuous.
            U = U @ expm(-1j * b[k] * t * B)
            U = U @ expm(-1j * a[k] * t * A)
        
        diff = U - EXACT_PROPAGATORS[t]
        # Spectral norm is max singular value
        # np.linalg.svd(diff, compute_uv=False)[-1] is the smallest.
        # We want the largest.
        # np.linalg.norm(diff, 2) computes spectral norm.
        err = np.linalg.norm(diff, 2)
        total_err += err
        
    return total_err / len(TIMES)

def objective(params):
    """
    params: 10 floats representing symmetric coefficients.
    a_1..a_5 are params[0:5], a_6 is params[5].
    Symmetry: a_7=a_6, ..., a_12=a_1.
    Constraint: 2*sum(a_1..a_5) + a_6 = 1.
    Similarly for b.
    """
    a_sym = params[0:6]
    b_sym = params[6:12]
    
    # Enforce sum constraint by scaling? Or just let the optimizer handle it?
    # Better to enforce it exactly to avoid constraint violations.
    # sum_a = 2 * sum(a_sym[0:5]) + a_sym[5]
    # We want sum_a = 1.
    # If sum_a != 1, we can scale, but that changes the shape.
    # It is better to parameterize such that the constraint is satisfied.
    # Let x_1..x_5 be free, x_6 = 1 - 2*sum(x_1..x_5).
    # This ensures sum = 1.
    
    a_1_5 = a_sym[0:5]
    a_6 = 1.0 - 2.0 * np.sum(a_1_5)
    a_full = np.array([a_1_5[0], a_1_5[1], a_1_5[2], a_1_5[3], a_1_5[4], a_6,
                       a_6, a_1_5[4], a_1_5[3], a_1_5[2], a_1_5[1], a_1_5[0]])
    
    b_1_5 = b_sym[0:5]
    b_6 = 1.0 - 2.0 * np.sum(b_1_5)
    b_full = np.array([b_1_5[0], b_1_5[1], b_1_5[2], b_1_5[3], b_1_5[4], b_6,
                       b_6, b_1_5[4], b_1_5[3], b_1_5[2], b_1_5[1], b_1_5[0]])
    
    # Check bounds [-50, 50]
    if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
        return 1e10 # Penalty
    
    return compute_error(a_full, b_full)

def coefficients(stages):
    start_time = time.time()
    
    # Initial guess: Uniform
    # a_k = 1/12. Symmetric: a_1..a_5 = 1/12, a_6 = 1 - 2*5/12 = 1 - 10/12 = 2/12 = 1/6.
    # Let's check: 2*(5/12) + 1/6 = 10/12 + 2/12 = 12/12 = 1. Correct.
    init_a = [1.0/12.0] * 5 + [1.0/6.0]
    init_b = [1.0/12.0] * 5 + [1.0/6.0]
    x0 = np.array(init_a + init_b)
    
    best_params = x0
    best_score = objective(x0)
    
    # Optimization
    try:
        # Use L-BFGS-B
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 200, 'maxfev': 200})
        
        # Check time
        if time.time() - start_time < 45.0:
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        
        # If we have time, try a second run with a different start or method
        if time.time() - start_time < 30.0:
            # Try Nelder-Mead
            res2 = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'maxfev': 100})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
                
    except Exception as e:
        pass # Ignore errors, use best so far

    # Extract final a, b
    a_1_5 = best_params[0:5]
    a_6 = 1.0 - 2.0 * np.sum(a_1_5)
    a_full = [a_1_5[0], a_1_5[1], a_1_5[2], a_1_5[3], a_1_5[4], a_6,
              a_6, a_1_5[4], a_1_5[3], a_1_5[2], a_1_5[1], a_1_5[0]]
    
    b_1_5 = best_params[6:11]
    b_6 = 1.0 - 2.0 * np.sum(b_1_5)
    b_full = [b_1_5[0], b_1_5[1], b_1_5[2], b_1_5[3], b_1_5[4], b_6,
              b_6, b_1_5[4], b_1_5[3], b_1_5[2], b_1_5[1], b_1_5[0]]
    
    # Ensure finite
    for i in range(12):
        if not np.isfinite(a_full[i]): a_full[i] = 1.0/12.0
        if not np.isfinite(b_full[i]): b_full[i] = 1.0/12.0
        
    return a_full, b_full
