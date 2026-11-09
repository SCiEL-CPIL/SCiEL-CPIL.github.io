"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _build_hamiltonians():
    """
    Build the A and B Hamiltonians for a 6-site open XXZ Heisenberg chain.
    A: odd bonds + half field
    B: even bonds + half field
    Anisotropy lambda = 0.8, field h = 0.3
    """
    N = 6
    lambda_aniso = 0.8
    h_field = 0.3
    
    # Spin operators
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Build operators for N sites
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    # Build A (odd bonds: 1-2, 3-4, 5-6) and B (even bonds: 2-3, 4-5)
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Field term: h * sum Sz_i / 2
    # The problem says "each including half of the uniform field".
    # So A gets 0.5 * h * sum Sz, B gets 0.5 * h * sum Sz.
    field_term = 0.5 * h_field
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
    
    # Bond terms
    # Helper to add bond interaction
    def add_bond(H, i, j):
        # H += Sx_i Sx_j + Sy_i Sy_j + lambda Sz_i Sz_j
        sx_i = op_on_site(sx, i, N)
        sx_j = op_on_site(sx, j, N)
        sy_i = op_on_site(sy, i, N)
        sy_j = op_on_site(sy, j, N)
        sz_i = op_on_site(sz, i, N)
        sz_j = op_on_site(sz, j, N)
        
        H += sx_i @ sx_j
        H += sy_i @ sy_j
        H += lambda_aniso * (sz_i @ sz_j)
        
    # Odd bonds for A: (0,1), (2,3), (4,5)
    add_bond(A, 0, 1)
    add_bond(A, 2, 3)
    add_bond(A, 4, 5)
    
    # Even bonds for B: (1,2), (3,4)
    add_bond(B, 1, 2)
    add_bond(B, 3, 4)
    
    return A, B

def coefficients(stages):
    """
    Optimize 12-stage symmetric product formula coefficients.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for 60s limit
    
    # Pre-compute Hamiltonians
    A, B = _build_hamiltonians()
    H_total = A + B
    
    # Pre-compute exact propagators for the 3 time points
    t_values = [0.25, 0.5, 1.0]
    U_exacts = []
    for t in t_values:
        # exp(-i t H_total)
        # Use scipy.linalg.expm
        try:
            from scipy.linalg import expm
            U_exact = expm(-1j * t * H_total)
        except ImportError:
            # Fallback to numpy if scipy is not available (though task says it is)
            # This is a crude fallback, might be slow or inaccurate for large matrices
            # But 64x64 is small.
            # We can use the eigen-decomposition method for expm
            eigvals, eigvecs = np.linalg.eigh(H_total)
            U_exact = (eigvecs * np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        U_exacts.append(U_exact)
    
    # Diagonalize A and B for faster exponential calculation
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Helper to apply the product formula
    def apply_product_formula(a, b, t, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B):
        # U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        # We iterate k=1 to 12
        # U_curr = I
        U_curr = np.eye(64, dtype=complex)
        
        for k in range(12):
            # Step 1: exp(-i b_k t B)
            # B = V_B D_B V_B^dag
            # exp(-i b_k t B) = V_B exp(-i b_k t D_B) V_B^dag
            exp_B = np.exp(-1j * b[k] * t * eigvals_B)
            U_B = (eigvecs_B * exp_B) @ eigvecs_B.conj().T
            U_curr = U_B @ U_curr
            
            # Step 2: exp(-i a_k t A)
            exp_A = np.exp(-1j * a[k] * t * eigvals_A)
            U_A = (eigvecs_A * exp_A) @ eigvecs_A.conj().T
            U_curr = U_A @ U_curr
            
        return U_curr

    # Objective function for Nelder-Mead
    def objective(x):
        if time.time() > deadline:
            return float('inf')
            
        # x is array of 6 floats representing the first half of the symmetric sequence
        # a = [x0, x1, x2, x3, x4, x5, x5, x4, x3, x2, x1, x0]
        # b = [x0, x1, x2, x3, x4, x5, x5, x4, x3, x2, x1, x0]
        # Sum constraint: 2 * sum(x) = 1 => sum(x) = 0.5
        
        s = sum(x)
        penalty = (s - 0.5)**2 * 1000
        
        seq = [x[0], x[1], x[2], x[3], x[4], x[5], x[5], x[4], x[3], x[2], x[1], x[0]]
        a = seq
        b = seq
        
        error = 0.0
        for i, t in enumerate(t_values):
            U_approx = apply_product_formula(a, b, t, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B)
            diff = U_approx - U_exacts[i]
            err = np.linalg.norm(diff, 2)
            error += err
            
        return error / len(t_values) + penalty

    # Initial guess: 4th order coefficients distributed?
    # 4th order: c1, c2, c3.
    # c1 = 1/(2-2^(1/3)) ~ 1.351
    # c2 = -2^(1/3)/(2-2^(1/3)) ~ -0.701
    # c3 = 1 - c1 - c2 ~ 0.349
    # Sum = 1.
    # In our symmetric 6-param scheme, sum(x) = 0.5.
    # Let's just start with uniform 1/12 scaled to sum 0.5?
    # Uniform: 1/12 * 12 = 1. So x_i = 1/24? No, sum(x)=0.5.
    # If we use uniform a_k = 1/12, then sum(a) = 1.
    # In symmetric form: a = [x, x, x, x, x, x, x, x, x, x, x, x]
    # 12x = 1 => x = 1/12.
    # Sum(x) = 6 * (1/12) = 0.5. Correct.
    x0 = np.array([1.0/12.0] * 6)
    
    # Try Nelder-Mead
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if time.time() < deadline:
            best_x = res.x
        else:
            best_x = x0
    except Exception:
        best_x = x0

    # Convert best_x to final a, b
    # Ensure sum is exactly 1 by scaling if necessary?
    # The penalty should have kept it close.
    # Let's normalize to be safe.
    x = best_x
    s = sum(x)
    if s != 0:
        x = x / (2 * s) # So that 2*sum(x) = 1
    
    seq = [x[0], x[1], x[2], x[3], x[4], x[5], x[5], x[4], x[3], x[2], x[1], x[0]]
    a = seq
    b = seq
    
    # Pad or truncate to match stages if necessary (though stages is likely 12)
    # The task says "twelve-stage product formula", so stages should be 12.
    # But the function signature takes 'stages'.
    if stages != 12:
        # Fallback to simple Strang if stages is not 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        # Truncate if too long
        a = a[:stages]
        b = b[:stages]
    
    # Final check for finite numbers in [-50, 50]
    a = [max(-50.0, min(50.0, float(ai))) for ai in a]
    b = [max(-50.0, min(50.0, float(bi))) for bi in b]
    
    return a, b
# EVOLVE-BLOCK-END
