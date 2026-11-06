"""Product formula design for Heisenberg-chain time evolution."""
import numpy as np
import time
from math import cos, sin, pi

def coefficients(stages):
    """Find optimal 12-stage product formula coefficients for Heisenberg chain."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Build the Hamiltonians A and B for a 6-site open XXZ Heisenberg chain
    # with anisotropy 0.8 and field 0.3
    N = 6
    gamma = 0.8  # anisotropy
    h = 0.3      # field
    
    # Total Hilbert space dimension
    dim = 2 ** N
    
    # Build Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def tensor(*ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    # Build A (odd bonds: 1-2, 3-4, 5-6) and B (even bonds: 2-3, 4-5)
    # Each includes half of the uniform field
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N):
        # Build Sx_i, Sy_i, Sz_i operators
        ops_x = [I2] * N
        ops_y = [I2] * N
        ops_z = [I2] * N
        ops_x[i] = sx
        ops_y[i] = sy
        ops_z[i] = sz
        
        Sx_i = tensor(*ops_x)
        Sy_i = tensor(*ops_y)
        Sz_i = tensor(*ops_z)
        
        # Field term: each site gets h/2 * Sz_i (half field split between A and B)
        field_term = (h / 2) * Sz_i
        
        # Bond terms
        if i < N - 1:
            j = i + 1
            # Sx_i * Sx_j + Sy_i * Sy_j + gamma * Sz_i * Sz_j
            ops_xj = [I2] * N
            ops_yj = [I2] * N
            ops_zj = [I2] * N
            ops_xj[i] = sx
            ops_xj[j] = sx
            ops_yj[i] = sy
            ops_yj[j] = sy
            ops_zj[i] = sz
            ops_zj[j] = sz
            
            Sx_i_Sx_j = tensor(*ops_xj)
            Sy_i_Sy_j = tensor(*ops_yj)
            Sz_i_Sz_j = tensor(*ops_zj)
            
            bond_term = Sx_i_Sx_j + Sy_i_Sy_j + gamma * Sz_i_Sz_j
            
            # Odd bonds (0-1, 2-3, 4-5) go to A, even bonds (1-2, 3-4) go to B
            if (i % 2 == 0):  # bonds 0-1, 2-3, 4-5
                A += bond_term
            else:  # bonds 1-2, 3-4
                B += bond_term
        
        # Add field term to both A and B (each gets half)
        if (i % 2 == 0):
            A += field_term
        else:
            B += field_term
    
    # Compute exact propagator eigenvalues
    H_total = A + B
    eigvals_total = np.linalg.eigvalsh(H_total)
    
    # Define the error function
    def compute_error(a, b, t_values):
        """Compute mean spectral norm error at given times."""
        errors = []
        for t in t_values:
            # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
            U = np.eye(dim, dtype=complex)
            for k in range(len(a)):
                # exp(-i b_k t B)
                if abs(b[k]) > 1e-15:
                    eigvals_B, eigvecs_B = np.linalg.eigh(B)
                    exp_B = (eigvecs_B * np.exp(-1j * b[k] * t * eigvals_B)) @ eigvecs_B.conj().T
                    U = exp_B @ U
                # exp(-i a_k t A)
                if abs(a[k]) > 1e-15:
                    eigvals_A, eigvecs_A = np.linalg.eigh(A)
                    exp_A = (eigvecs_A * np.exp(-1j * a[k] * t * eigvals_A)) @ eigvecs_A.conj().T
                    U = exp_A @ U
            
            # Exact propagator
            eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
            U_exact = (eigvecs_H * np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
            
            # Spectral norm of difference
            diff = U - U_exact
            error = np.linalg.norm(diff, 2)
            errors.append(error)
        
        return np.mean(errors)
    
    # Precompute eigendecompositions for speed
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(H_total)
    
    def compute_error_fast(a, b, t_values):
        """Fast error computation using precomputed eigendecompositions."""
        errors = []
        for t in t_values:
            U = np.eye(dim, dtype=complex)
            for k in range(len(a)):
                if abs(b[k]) > 1e-15:
                    exp_B = (eigvecs_B * np.exp(-1j * b[k] * t * eigvals_B)) @ eigvecs_B.conj().T
                    U = exp_B @ U
                if abs(a[k]) > 1e-15:
                    exp_A = (eigvecs_A * np.exp(-1j * a[k] * t * eigvals_A)) @ eigvecs_A.conj().T
                    U = exp_A @ U
            
            U_exact = (eigvecs_H * np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
            diff = U - U_exact
            error = np.linalg.norm(diff, 2)
            errors.append(error)
        
        return np.mean(errors)
    
    t_values = [0.25, 0.5, 1.0]
    
    # Start with the seed solution
    substeps = stages // 2
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    
    best_a = a[:]
    best_b = b[:]
    best_error = compute_error_fast(best_a, best_b, t_values)
    
    # Nelder-Mead optimization
    def nelder_mead(func, x0, max_iter=1000, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """Nelder-Mead optimization."""
        n = len(x0)
        simplex = [x0[:]]
        for i in range(n):
            point = x0[:]
            point[i] = point[i] + 0.1 if point[i] == 0 else point[i] * 1.1
            simplex.append(point)
        
        func_vals = [func(x) for x in simplex]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Sort simplex
            order = np.argsort(func_vals)
            simplex = [simplex[i] for i in order]
            func_vals = [func_vals[i] for i in order]
            
            # Check convergence
            if abs(func_vals[-1] - func_vals[0]) < tol:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            
            if f_r < func_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    func_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    func_vals[-1] = f_r
            elif f_r < func_vals[-2]:
                simplex[-1] = x_r
                func_vals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = func(x_c)
                if f_c < func_vals[-1]:
                    simplex[-1] = x_c
                    func_vals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        func_vals[i] = func(simplex[i])
        
        # Return best point
        best_idx = np.argmin(func_vals)
        return simplex[best_idx], func_vals[best_idx]
    
    # Optimize using Nelder-Mead
    def objective(x):
        a_opt = x[:stages]
        b_opt = x[stages:]
        # Enforce sum constraints approximately
        sum_a = np.sum(a_opt)
        sum_b = np.sum(b_opt)
        if abs(sum_a - 1) > 0.01 or abs(sum_b - 1) > 0.01:
            return 1e6
        # Clamp to [-50, 50]
        a_clamped = np.clip(a_opt, -50, 50)
        b_clamped = np.clip(b_opt, -50, 50)
        return compute_error_fast(a_clamped, b_clamped, t_values)
    
    x0 = best_a + best_b
    x_opt, f_opt = nelder_mead(objective, x0, max_iter=200, tol=1e-10)
    
    a_opt = x_opt[:stages]
    b_opt = x_opt[stages:]
    
    # Enforce constraints
    sum_a = np.sum(a_opt)
    sum_b = np.sum(b_opt)
    if abs(sum_a) > 1e-10:
        a_opt = a_opt / sum_a
    if abs(sum_b) > 1e-10:
        b_opt = b_opt / sum_b
    
    a_opt = np.clip(a_opt, -50, 50)
    b_opt = np.clip(b_opt, -50, 50)
    
    # Final polish with tighter tolerance
    def objective_tight(x):
        a_t = x[:stages]
        b_t = x[stages:]
        sum_a = np.sum(a_t)
        sum_b = np.sum(b_t)
        if abs(sum_a - 1) > 0.001 or abs(sum_b - 1) > 0.001:
            return 1e6
        a_clamped = np.clip(a_t, -50, 50)
        b_clamped = np.clip(b_t, -50, 50)
        return compute_error_fast(a_clamped, b_clamped, t_values)
    
    x_tight, f_tight = nelder_mead(objective_tight, x_opt, max_iter=100, tol=1e-12)
    
    a_final = x_tight[:stages]
    b_final = x_tight[stages:]
    
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    if abs(sum_a) > 1e-10:
        a_final = a_final / sum_a
    if abs(sum_b) > 1e-10:
        b_final = b_final / sum_b
    
    a_final = np.clip(a_final, -50, 50)
    b_final = np.clip(b_final, -50, 50)
    
    final_error = compute_error_fast(a_final, b_final, t_values)
    
    # Return best solution
    if final_error < best_error:
        best_a = a_final
        best_b = b_final
        best_error = final_error
    
    return list(best_a), list(best_b)
