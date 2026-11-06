"""Product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds) and B (even bonds) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    XXZ Hamiltonian: sum_j [gamma/2 (S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) + S_j^z S_{j+1}^z] - h sum_j S_j^z
    Split: A = odd bonds (1-2, 3-4, 5-6) + half field, B = even bonds (2-3, 4-5) + half field
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3
    
    # Build spin operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sp = sx + 1j * sy  # sigma_+ = |0><1|
    sm = sx - 1j * sy  # sigma_- = |1><0|
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply single-site operator at site (0-indexed)."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    def bond_hamiltonian(j, gamma, h_half):
        """Bond Hamiltonian between sites j and j+1 (0-indexed)."""
        # gamma/2 (S_j^+ S_{j+1}^- + S_j^- S_{j+1}^+) + S_j^z S_{j+1}^z
        term1 = gamma / 2 * (np.kron(op_on_site(sp, j), op_on_site(sm, j + 1)) +
                              np.kron(op_on_site(sm, j), op_on_site(sp, j + 1)))
        term2 = np.kron(op_on_site(sz, j), op_on_site(sz, j + 1))
        return term1 + term2
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    for j in range(0, N - 1, 2):
        A += bond_hamiltonian(j, gamma, h / 2)
    
    # Even bonds: (1,2), (3,4)
    for j in range(1, N - 1, 2):
        B += bond_hamiltonian(j, gamma, h / 2)
    
    # Add half field to each
    for site in range(N):
        A -= (h / 2) * op_on_site(sz, site)
        B -= (h / 2) * op_on_site(sz, site)
    
    return A, B


def coefficients(stages):
    """Find optimal 12-stage product formula coefficients."""
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s safety margin
    
    A, B = _build_hamiltonians()
    AB = A + B
    
    # Precompute exact propagators at evaluation times
    eval_times = [0.25, 0.5, 1.0]
    exact_propagators = [expm(-1j * t * AB) for t in eval_times]
    
    def compute_error(a_list, b_list):
        """Compute mean spectral norm error at the three evaluation times."""
        total_error = 0.0
        for k, t in enumerate(eval_times):
            U = np.eye(2 ** 6, dtype=complex)
            for i in range(len(a_list)):
                if b_list[i] != 0:
                    U = U @ expm(-1j * b_list[i] * t * B)
                if a_list[i] != 0:
                    U = U @ expm(-1j * a_list[i] * t * A)
            err = np.linalg.norm(U - exact_propagators[k], 2)
            total_error += err
        return total_error / len(eval_times)
    
    # 4th-order Suzuki coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c2 = 1.0 - 2.0 * c1
    
    # Base 5-stage 4th order: A(c1/2), B(c1), A(c2/2), B(c2), A(c1/2)
    # Actually S_2(t) = A(t/2) B(t) A(t/2), so S_4 = S_2(c1*t) S_2(c2*t) S_2(c1*t)
    # A coefficients: c1/2, c1/2+c2/2, c2/2+c1/2, c2/2, c1/2 -- wait, let me be careful
    # S_2(c1*t): A(c1*t/2), B(c1*t), A(c1*t/2)
    # S_2(c2*t): A(c2*t/2), B(c2*t), A(c2*t/2)
    # S_2(c1*t): A(c1*t/2), B(c1*t), A(c1*t/2)
    # Combined (merging adjacent A's):
    # A: c1/2, c1/2+c2/2, c2/2+c1/2, c2/2, c1/2
    # B: c1, c2, c1
    # That's 5 A stages and 3 B stages
    
    a_base = [c1 / 2, c1 / 2 + c2 / 2, c2 / 2 + c1 / 2, c2 / 2, c1 / 2]
    b_base = [c1, c2, c1]
    
    # Pad to 12 stages
    a_init = list(a_base) + [0.0] * (12 - len(a_base))
    b_init = list(b_base) + [0.0] * (12 - len(b_base))
    
    # Verify sums
    a_sum = sum(a_init)
    b_sum = sum(b_init)
    # Normalize to ensure sum=1
    if abs(a_sum) > 1e-15:
        a_init = [x / a_sum for x in a_init]
    if abs(b_sum) > 1e-15:
        b_init = [x / b_sum for x in b_init]
    
    # Define objective with constraints handled via parameterization
    # We'll use 22 free parameters (drop last a and last b, compute from sum constraint)
    def objective(params):
        a = list(params[:11]) + [1.0 - sum(params[:11])]
        b = list(params[11:22]) + [1.0 - sum(params[11:22])]
        return compute_error(a, b)
    
    def constraint_check(a, b):
        """Check if coefficients are in valid range."""
        for x in a + b:
            if x < -50 or x > 50:
                return False
        return True
    
    x0 = a_init[:11] + b_init[:11]
    
    # Run Nelder-Mead
    best_result = None
    best_x = x0
    best_obj = objective(x0)
    
    def _callback(xk):
        nonlocal best_x, best_obj
        obj_k = objective(xk)
        a_k = list(xk[:11]) + [1.0 - sum(xk[:11])]
        b_k = list(xk[11:22]) + [1.0 - sum(xk[11:22])]
        if constraint_check(a_k, b_k):
            if obj_k < best_obj:
                best_obj = obj_k
                best_x = xk.copy()
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
                'callback': _callback,
            }
        )
        
        # Check if result is valid
        x_res = result.x
        a_res = list(x_res[:11]) + [1.0 - sum(x_res[:11])]
        b_res = list(x_res[11:22]) + [1.0 - sum(x_res[11:22])]
        
        if constraint_check(a_res, b_res):
            obj_res = objective(x_res)
            if obj_res < best_obj:
                best_obj = obj_res
                best_x = x_res.copy()
        
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < DEADLINE:
        # Do a second round with tighter tolerance if we have time
        try:
            x0_2 = best_x
            def _callback2(xk):
                nonlocal best_x, best_obj
                obj_k = objective(xk)
                a_k = list(xk[:11]) + [1.0 - sum(xk[:11])]
                b_k = list(xk[11:22]) + [1.0 - sum(xk[11:22])]
                if constraint_check(a_k, b_k):
                    if obj_k < best_obj:
                        best_obj = obj_k
                        best_x = xk.copy()
            
            result2 = minimize(
                objective,
                x0_2,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                    'callback': _callback2,
                }
            )
            
            x_res2 = result2.x
            a_res2 = list(x_res2[:11]) + [1.0 - sum(x_res2[:11])]
            b_res2 = list(x_res2[11:22]) + [1.0 - sum(x_res2[11:22])]
            
            if constraint_check(a_res2, b_res2):
                obj_res2 = objective(x_res2)
                if obj_res2 < best_obj:
                    best_obj = obj_res2
                    best_x = x_res2.copy()
                    
        except Exception:
            pass
    
    # Construct final answer
    a_final = list(best_x[:11]) + [1.0 - sum(best_x[:11])]
    b_final = list(best_x[11:22]) + [1.0 - sum(best_x[11:22])]
    
    # Clamp to valid range
    a_final = [max(-50, min(50, x)) for x in a_final]
    b_final = [max(-50, min(50, x)) for x in b_final]
    
    # Renormalize after clamping
    a_sum = sum(a_final)
    b_sum = sum(b_final)
    if abs(a_sum) > 1e-15:
        a_final = [x / a_sum for x in a_final]
    if abs(b_sum) > 1e-15:
        b_final = [x / b_sum for x in b_final]
    
    # Final clamp
    a_final = [max(-50, min(50, x)) for x in a_final]
    b_final = [max(-50, min(50, x)) for x in b_final]
    
    return a_final, b_final
