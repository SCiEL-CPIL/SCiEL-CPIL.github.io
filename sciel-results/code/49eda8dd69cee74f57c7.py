import numpy as np
import time
import sys

def _build_hamiltonians():
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    # Spin operators are 0.5 * Pauli
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(I2)
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res
            
    def two_site_op(op1, site1, op2, site2):
        ops = []
        for i in range(N):
            if i == site1:
                ops.append(op1)
            elif i == site2:
                ops.append(op2)
            else:
                ops.append(I2)
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res

    def bond_term(site1, site2):
        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds
    # 0-1 (Odd) -> A
    # 1-2 (Even) -> B
    # 2-3 (Odd) -> A
    # 3-4 (Even) -> B
    # 4-5 (Odd) -> A
    
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    # Field
    # H_field = h * sum Sz_j
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def _nelder_mead(f, x0, max_iter=500, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5, tol=1e-10, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        # Use a larger initial step for better exploration
        step = 0.1
        x_new[i] += step
        simplex.append(x_new)
        
    values = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Sort simplex by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
            
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        
        # Centroid of all but worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    values[i] = f(simplex[i])
                    
    # Return best point
    order = np.argsort(values)
    return simplex[order[0]], values[order[0]]

def _chebyshev_nodes(n):
    """Generate Chebyshev nodes in [-1, 1]"""
    k = np.arange(n)
    return np.cos(np.pi * (2*k + 1) / (2*n))

def _chebyshev_to_coefficients(cheb_coeffs_a, cheb_coeffs_b):
    """
    Convert Chebyshev coefficients to actual stage coefficients.
    cheb_coeffs_a: array of length 12, represents a_k = sum_j c_j * T_j(x_k)
    where x_k are Chebyshev nodes mapped to [0, 1]
    """
    n = 12
    # Map Chebyshev nodes from [-1,1] to [0,1]
    t_nodes = 0.5 * (_chebyshev_nodes(n) + 1)
    
    # Build Vandermonde-like matrix for Chebyshev polynomials
    def cheb_eval(T, x):
        """Evaluate Chebyshev polynomial T at x"""
        if T == 0:
            return np.ones_like(x)
        elif T == 1:
            return x.copy()
        else:
            T0 = np.ones_like(x)
            T1 = x.copy()
            for k in range(2, T+1):
                T2 = 2*x*T1 - T0
                T0 = T1
                T1 = T2
            return T1
    
    # Build design matrix
    M = np.zeros((n, n))
    for j in range(n):
        M[:, j] = cheb_eval(j, t_nodes)
    
    # Solve for coefficients
    # We want a_k = sum_j c_j * T_j(t_k)
    # So a = M @ c
    a = M @ cheb_coeffs_a
    b = M @ cheb_coeffs_b
    
    # Enforce sum constraint by adjusting
    a = a - np.sum(a) / n + 1.0/n
    b = b - np.sum(b) / n + 1.0/n
    
    return a, b

def _eval_error_chebyshev(cheb_coeffs_a, cheb_coeffs_b, A, B, H, U_exact_cache, eigvals_A, eigvecs_A, eigvals_B, eigvecs_B, time_points):
    """
    Evaluate the maximum error over a set of time points using Chebyshev parametrization.
    """
    a, b = _chebyshev_to_coefficients(cheb_coeffs_a, cheb_coeffs_b)
    
    if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
        return 1e10
    
    max_err = 0.0
    for t in time_points:
        U_approx = np.eye(64, dtype=complex)
        for k in range(12):
            if a[k] != 0:
                phase = np.exp(-1j * a[k] * t * eigvals_A)
                E_A_k = eigvecs_A @ np.diag(phase) @ eigvecs_A.conj().T
            else:
                E_A_k = np.eye(64, dtype=complex)
            
            if b[k] != 0:
                phase = np.exp(-1j * b[k] * t * eigvals_B)
                E_B_k = eigvecs_B @ np.diag(phase) @ eigvecs_B.conj().T
            else:
                E_B_k = np.eye(64, dtype=complex)
            
            U_approx = E_B_k @ E_A_k @ U_approx
        
        # Get exact propagator
        if t in U_exact_cache:
            U_target = U_exact_cache[t]
        else:
            phase = np.exp(-1j * t * np.linalg.eigvalsh(H))
            U_target = np.linalg.eigh(H)[1] @ np.diag(phase) @ np.linalg.eigh(H)[1].conj().T
            U_exact_cache[t] = U_target
        
        diff = U_approx - U_target
        err = np.linalg.norm(diff, 2)
        max_err = max(max_err, err)
    
    return max_err

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute exact propagators at Chebyshev nodes and evaluation points
    eigvals, eigvecs = np.linalg.eigh(H)
    
    # Use Chebyshev nodes in [0, 1] for robust error evaluation
    n_cheb = 20
    cheb_times = 0.5 * (_chebyshev_nodes(n_cheb) + 1)
    # Add the actual evaluation points
    eval_times = [0.25, 0.5, 1.0]
    all_times = sorted(set(np.round(cheb_times, 10).tolist() + eval_times))
    
    U_exact_cache = {}
    for t in all_times:
        phase = np.exp(-1j * t * eigvals)
        U_t = eigvecs @ np.diag(phase) @ eigvecs.conj().T
        U_exact_cache[t] = U_t
    
    # Precompute eigendecomps of A and B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    # Objective function: minimize max error over Chebyshev nodes
    def objective(cheb_params):
        if time.time() > deadline:
            return 1e10
        
        cheb_a = cheb_params[:12]
        cheb_b = cheb_params[12:24]
        
        return _eval_error_chebyshev(
            cheb_a, cheb_b, A, B, H, U_exact_cache,
            eigvals_A, eigvecs_A, eigvals_B, eigvecs_B,
            all_times
        )
    
    # Initial guess: zero Chebyshev coefficients (uniform splitting)
    x0 = np.zeros(24)
    
    # Run Nelder-Mead on Chebyshev coefficients
    x_best, val_best = _nelder_mead(
        objective, x0, 
        max_iter=2000, 
        time_limit=deadline - time.time(),
        start_time=start_time
    )
    
    # Convert Chebyshev coefficients to actual stage coefficients
    a, b = _chebyshev_to_coefficients(x_best[:12], x_best[12:24])
    
    # Ensure float types and final constraint satisfaction
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    # Final adjustment to ensure exact sum = 1
    a_sum = sum(a)
    b_sum = sum(b)
    for i in range(12):
        a[i] += (1.0 - a_sum) / 12
        b[i] += (1.0 - b_sum) / 12
    
    return a, b

if __name__ == "__main__":
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")
