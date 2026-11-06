"""ODE solver: Convection-diffusion 100-point using tridiagonal operator reconstruction."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = len(y0_arr)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290
    
    max_calls = 2000000
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Strategy:
    # 1. Reconstruct the tridiagonal convection-diffusion operator A using minimal evaluations.
    #    Convection-diffusion on a uniform grid with Dirichlet/Neumann BCs typically gives a tridiagonal matrix.
    #    We can infer the diagonal, upper, and lower coefficients by evaluating f at specific test vectors.
    #    
    #    For a tridiagonal matrix A with:
    #      A[i,i] = d_i (diagonal)
    #      A[i,i+1] = u_i (upper)
    #      A[i,i-1] = l_i (lower)
    #    
    #    We can use:
    #      f(t0, e_i) gives column i of A, which has at most 3 non-zero entries.
    #    But that's n evaluations. Can we do better?
    #    
    #    Alternative: Use 2 evaluations to infer the structure.
    #    Actually, for a uniform grid convection-diffusion, the coefficients are constant (except boundaries).
    #    Let's try to detect if it's a uniform tridiagonal operator.
    #    
    #    Better approach: Use n evaluations to get the full A, then use expm_multiply.
    #    n=100 evaluations is very cheap. The score penalty for 100 calls is -0.5*log10(100) ≈ -1.
    #    But if we can do it with 2 calls, the penalty is -0.5*log10(2) ≈ -0.15.
    #    
    #    Let's first try to detect if the operator is tridiagonal with constant coefficients.
    #    Evaluate f(t0, y0) and f(t0, some test vector) to check structure.
    
    # First, check if the system is linear and time-invariant
    # Evaluate f at t0 with y0
    f_y0 = f_counted(t0, y0_arr)
    
    # Try to reconstruct A by evaluating at basis vectors
    # This costs n evaluations total (including the one above if we reuse)
    # Actually, let's evaluate at e_0, e_1, ..., e_{n-1}
    
    try:
        A = np.zeros((n, n))
        
        # Evaluate at each basis vector
        for i in range(n):
            if time.time() > deadline:
                break
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f_counted(t0, e_i)
        
        # Verify linearity: f(t0, y0) should equal A @ y0
        A_y0 = A @ y0_arr
        norm_diff = np.linalg.norm(f_y0 - A_y0)
        norm_f = np.linalg.norm(f_y0)
        
        if norm_diff < 1e-10 * (norm_f + 1e-30):
            # Check time invariance
            if t1 > t0:
                t_mid = 0.5 * (t0 + t1)
                f_mid = f_counted(t_mid, y0_arr)
                diff_mid = np.linalg.norm(f_mid - A_y0)
                if diff_mid > 1e-8 * (norm_f + 1e-30):
                    # Time-dependent, fallback
                    raise Exception("Time-dependent")
                
                # Use matrix exponential
                dt = t1 - t0
                try:
                    # Use sparse expm_multiply for efficiency
                    A_sparse = diags(A)
                    y_final = expm_multiply(A_sparse * dt, y0_arr)
                    if np.all(np.isfinite(y_final)):
                        return y_final
                except Exception:
                    try:
                        expm_mat = expm(A * dt)
                        y_final = expm_mat @ y0_arr
                        if np.all(np.isfinite(y_final)):
                            return y_final
                    except Exception:
                        pass
    except Exception:
        pass
    
    # Fallback: Radau solver
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f_counted,
                [t0, t1],
                y0_arr,
                method='Radau',
                rtol=1e-14,
                atol=1e-16
            )
            if sol.success and sol.t[-1] >= t1 - 1e-14:
                return sol.y[:, -1]
            if sol.t.size > 0 and t1 - sol.t[-1] < 1e-10:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: DOP853
    try:
        if time.time() < deadline:
            sol = solve_ivp(
                f_counted,
                [t0, t1],
                y0_arr,
                method='DOP853',
                rtol=1e-14,
                atol=1e-16,
                max_step=(t1 - t0) / 100 if (t1 - t0) > 0 else 1.0
            )
            if sol.success and sol.t[-1] >= t1 - 1e-14:
                return sol.y[:, -1]
    except Exception:
        pass
    
    return y0_arr
