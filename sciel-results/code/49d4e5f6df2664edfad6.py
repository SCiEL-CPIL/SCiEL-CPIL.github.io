import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def coefficients(stages):
    if stages != 12:
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b

    # Start time to enforce deadline
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit

    # --- Build Hamiltonians ---
    # Spin-1/2 matrices
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    idn = np.eye(2, dtype=complex)

    # 6-site chain
    # A: Odd bonds (1-2, 3-4, 5-6) + Field on odd sites?
    # The problem states: "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    # This usually means:
    # A = H_odd_bonds + 0.5 * H_field
    # B = H_even_bonds + 0.5 * H_field
    # Where H_field = h * sum(S_i^z)
    
    anisotropy = 0.8
    field = 0.3

    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)

    # Helper to add interaction between site i and i+1 (0-indexed)
    def add_interaction(mat, i, j):
        # Tensor product for 6 sites
        # Sites are 0..5
        ops = [idn]*6
        ops[i] = sx
        ops[j] = sx
        term_xx = ops[0]
        for k in range(1, 6):
            term_xx = np.kron(term_xx, ops[k])
        
        ops[i] = sy
        ops[j] = sy
        term_yy = ops[0]
        for k in range(1, 6):
            term_yy = np.kron(term_yy, ops[k])
            
        ops[i] = sz
        ops[j] = sz
        term_zz = ops[0]
        for k in range(1, 6):
            term_zz = np.kron(term_zz, ops[k])
            
        return term_xx + term_yy + anisotropy * term_zz

    # Odd bonds: (0,1), (2,3), (4,5)
    A += add_interaction(A, 0, 1)
    A += add_interaction(A, 2, 3)
    A += add_interaction(A, 4, 5)
    
    # Even bonds: (1,2), (3,4)
    B += add_interaction(B, 1, 2)
    B += add_interaction(B, 3, 4)

    # Field: h * sum(S_i^z)
    # Split half to A, half to B
    field_term = np.zeros((64, 64), dtype=complex)
    for i in range(6):
        ops = [idn]*6
        ops[i] = sz
        term = ops[0]
        for k in range(1, 6):
            term = np.kron(term, ops[k])
        field_term += term
        
    A += 0.5 * field * field_term
    B += 0.5 * field * field_term

    # Exact propagator generator
    H = A + B

    # --- Objective Function ---
    def objective(params):
        if time.time() > deadline:
            return 1e10
        a = np.array(params[:11])
        b = np.array(params[11:22])
        
        # Enforce sum constraints by fixing last element
        a_12 = 1.0 - np.sum(a)
        b_12 = 1.0 - np.sum(b)
        
        a_full = np.append(a, a_12)
        b_full = np.append(b, b_12)
        
        # Check bounds [-50, 50]
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        error_sum = 0.0
        for t in [0.25, 0.5, 1.0]:
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                U_approx = expm(-1j * b_full[k] * t * B) @ U_approx
                U_approx = expm(-1j * a_full[k] * t * A) @ U_approx
            
            U_exact = expm(-1j * t * H)
            err = np.linalg.norm(U_approx - U_exact, 2)
            error_sum += err
            
        return error_sum / 3.0

    # --- Initial Guess ---
    # Use 4th-order Suzuki-Trotter as a good starting point
    c = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - c
    c1 = 1.0 / denom
    c2 = -c / denom
    c3 = c1
    
    # Map 4th order to 12 stages (as in current program)
    a_init = np.zeros(12)
    b_init = np.zeros(12)
    a_init[0] = c1 / 2.0
    b_init[1] = c1
    a_init[2] = (c1 + c2) / 2.0
    b_init[3] = c2
    a_init[4] = (c2 + c3) / 2.0
    b_init[5] = c3
    a_init[6] = c3 / 2.0
    
    # Flatten for optimizer (first 11 of each)
    x0 = np.concatenate([a_init[:11], b_init[:11]])

    # --- Optimization ---
    # Try Nelder-Mead for robustness with spectral norm
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if time.time() > deadline:
            # Fallback to initial guess if optimization took too long
            a_res = a_init
            b_res = b_init
        else:
            if result.success:
                a_opt = np.append(result.x[:11], 1.0 - np.sum(result.x[:11]))
                b_opt = np.append(result.x[11:22], 1.0 - np.sum(result.x[11:22]))
                
                # Verify bounds
                if np.all((a_opt >= -50) & (a_opt <= 50)) and np.all((b_opt >= -50) & (b_opt <= 50)):
                    a_res = a_opt
                    b_res = b_opt
                else:
                    a_res = a_init
                    b_res = b_init
            else:
                a_res = a_init
                b_res = b_init
                
    except Exception:
        a_res = a_init
        b_res = b_init

    return list(a_res), list(b_res)
