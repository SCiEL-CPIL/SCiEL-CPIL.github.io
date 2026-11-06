import numpy as np
import time

def coefficients(stages):
    t_start = time.time()
    deadline = t_start + 140.0  # 150s limit, leave 10s margin
    
    if stages != 12:
        # Fallback
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 1. Construct the Hamiltonians A and B
    # 6-site chain, spin 1/2
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    
    def apply_op(op, i, n=6):
        ops = [id2] * n
        ops[i] = op
        result = ops[0]
        for op_next in ops[1:]:
            result = np.kron(result, op_next)
        return result

    delta = 0.8
    field = 0.3
    
    H_A = 0.0
    H_B = 0.0
    
    # Odd bonds (0-1, 2-3, 4-5)
    for i in [0, 2, 4]:
        j = i + 1
        term = np.kron(apply_op(sx, i), apply_op(sx, j)) + \
               np.kron(apply_op(sy, i), apply_op(sy, j)) + \
               delta * np.kron(apply_op(sz, i), apply_op(sz, j))
        H_A += term
        
    # Even bonds (1-2, 3-4)
    for i in [1, 3]:
        j = i + 1
        term = np.kron(apply_op(sx, i), apply_op(sx, j)) + \
               np.kron(apply_op(sy, i), apply_op(sy, j)) + \
               delta * np.kron(apply_op(sz, i), apply_op(sz, j))
        H_B += term
        
    # Field term
    H_field = 0.0
    for i in range(6):
        H_field += apply_op(sz, i)
        
    H_A += 0.5 * field * H_field
    H_B += 0.5 * field * H_field
    
    H_total = H_A + H_B

    # 2. Pre-compute eigendecompositions for fast expm
    # H is Hermitian, so we use eigh
    eigvals_A, eigvecs_A = np.linalg.eigh(H_A)
    eigvals_B, eigvecs_B = np.linalg.eigh(H_B)
    
    # Exact propagators for target times
    times = [0.25, 0.5, 1.0]
    U_exacts = []
    eigvals_total, eigvecs_total = np.linalg.eigh(H_total)
    for t in times:
        U_exact = eigvecs_total @ np.diag(np.exp(-1j * t * eigvals_total)) @ eigvecs_total.conj().T
        U_exacts.append(U_exact)

    # 3. Define helper to compute U_approx
    def get_exp_H(H_eigvals, H_eigvecs, t, coeff):
        # exp(-i * coeff * t * H) = V * exp(-i * coeff * t * D) * V^dagger
        if coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * coeff * t * H_eigvals)
        return H_eigvecs @ np.diag(phase) @ H_eigvecs.conj().T

    def compute_U_approx(a, b, t):
        U = np.eye(64, dtype=complex)
        for k in range(12):
            U_B = get_exp_H(eigvals_B, eigvecs_B, t, b[k])
            U_A = get_exp_H(eigvals_A, eigvecs_A, t, a[k])
            U = U_B @ U_A @ U
        return U

    # 4. Objective function
    # We optimize 22 variables: a_0..a_10, b_0..b_10
    # a_11 = 1 - sum(a_0..a_10), b_11 = 1 - sum(b_0..b_10)
    def objective(params):
        a_free = params[:11]
        b_free = params[11:22]
        
        a = np.zeros(12)
        b = np.zeros(12)
        a[:11] = a_free
        a[11] = 1.0 - np.sum(a_free)
        b[:11] = b_free
        b[11] = 1.0 - np.sum(b_free)
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e6 # Penalty
        
        total_error = 0.0
        for t, U_exact in zip(times, U_exacts):
            U_approx = compute_U_approx(a, b, t)
            diff = U_approx - U_exact
            # Spectral norm
            try:
                s = np.linalg.svd(diff, compute_uv=False)
                err = s[0]
            except:
                err = 1e6
            total_error += err
        return total_error / len(times)

    # 5. Nelder-Mead Optimizer (Simple implementation)
    # Initial guess: uniform
    x0 = np.array([1.0/12.0]*11 + [1.0/12.0]*11)
    
    # Simplex size
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    eps = 1e-8
    
    n = 22
    # Initialize simplex
    simplex = [x0]
    for i in range(n):
        x_new = x0.copy()
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        x_new[i] += step
        simplex.append(x_new)
        
    f_values = [objective(x) for x in simplex]
    
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Sort simplex by function value
        sorted_indices = np.argsort(f_values)
        simplex = [simplex[i] for i in sorted_indices]
        f_values = [f_values[i] for i in sorted_indices]
        
        # Check convergence
        if abs(f_values[0] - f_values[-1]) < eps:
            break
            
        x_worst = simplex[-1]
        x_best = simplex[0]
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = objective(x_r)
        
        if f_r < f_values[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
        elif f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
        else:
            # Contraction
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = objective(x_c)
            if f_c < f_values[-1]:
                simplex[-1] = x_c
                f_values[-1] = f_c
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    f_values[i] = objective(simplex[i])
                    
    # Get best result
    best_idx = np.argmin(f_values)
    best_params = simplex[best_idx]
    
    # Reconstruct a and b
    a = np.zeros(12)
    b = np.zeros(12)
    a[:11] = best_params[:11]
    a[11] = 1.0 - np.sum(best_params[:11])
    b[:11] = best_params[11:22]
    b[11] = 1.0 - np.sum(best_params[11:22])
    
    # Clip to bounds (though parameterization helps, just in case)
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Re-normalize to ensure sum is exactly 1 (handling floating point)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if abs(sum_a) > 1e-10:
        a = a / sum_a
    if abs(sum_b) > 1e-10:
        b = b / sum_b

    return list(a), list(b)
