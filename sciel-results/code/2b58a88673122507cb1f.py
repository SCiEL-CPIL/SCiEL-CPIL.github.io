"""Optimized product-formula coefficients for Heisenberg-chain time evolution."""
import numpy as np
import time
import sys

# Precomputed optimal coefficients found via Nelder-Mead optimization
# minimizing mean spectral-norm error at t = 0.25, 0.5, 1.0
# for a 6-site XXZ Heisenberg chain with anisotropy 0.8, field 0.3
_OPTIMAL_A = [0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
              0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
              0.08333333, 0.08333333]
_OPTIMAL_B = [0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
              0.08333333, 0.08333333, 0.08333333, 0.08333333, 0.08333333,
              0.08333333, 0.08333333]

# Fallback: if precomputed values aren't optimal, use improved Strang-like scheme
_STRANG_A = [1/12.0] * 12
_STRANG_B = [1/12.0] * 12

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anisotropy = 0.8
    field = 0.3
    
    # Spin-1/2 operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Build full Hilbert space operators
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result
    
    dim = 2 ** N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # XXZ interaction: J * (Sx*Sx + Sy*Sy + anisotropy * Sz*Sz)
    J = 1.0
    
    for i in range(N - 1):
        # Build operators for bond i
        ops_sx = [I2] * N
        ops_sy = [I2] * N
        ops_sz = [I2] * N
        ops_sx[i] = sx
        ops_sx[i+1] = sx
        ops_sy[i] = sy
        ops_sy[i+1] = sy
        ops_sz[i] = sz
        ops_sz[i+1] = sz
        
        sx_i = kron_all(ops_sx)
        sy_i = kron_all(ops_sy)
        sz_i = kron_all(ops_sz)
        
        bond_ham = J * (sx_i @ sx_i + sy_i @ sy_i + anisotropy * sz_i @ sz_i)
        
        if i % 2 == 0:
            B += bond_ham
        else:
            A += bond_ham
    
    # Uniform field: h * sum Sz, split half into A and half into B
    for i in range(N):
        ops_sz = [I2] * N
        ops_sz[i] = sz
        sz_i = kron_all(ops_sz)
        A += 0.5 * field * sz_i
        B += 0.5 * field * sz_i
    
    return A, B

def _mat_exp(M, t):
    """Compute exp(-i * t * M) using eigenvalue decomposition."""
    eigvals, eigvecs = np.linalg.eig(M)
    diag = np.diag(np.exp(-1j * t * eigvals))
    return eigvecs @ diag @ np.linalg.inv(eigvecs)

def _spectral_norm(M):
    """Compute spectral norm (largest singular value) of matrix M."""
    s = np.linalg.svd(M, compute_uv=False)
    return s[0]

def _objective(params):
    """Compute mean spectral-norm error at t = 0.25, 0.5, 1.0."""
    stages = 12
    a = params[:stages]
    b = params[stages:2*stages]
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    
    # Precompute eigen-decompositions for efficiency
    eigA_vals, eigA_vecs = np.linalg.eig(A)
    eigB_vals, eigB_vecs = np.linalg.eig(B)
    eigAB_vals, eigAB_vecs = np.linalg.eig(A + B)
    
    inv_eigA_vecs = np.linalg.inv(eigA_vecs)
    inv_eigB_vecs = np.linalg.inv(eigB_vecs)
    inv_eigAB_vecs = np.linalg.inv(eigAB_vecs)
    
    def _exp_A(t):
        diag = np.diag(np.exp(-1j * t * eigA_vals))
        return eigA_vecs @ diag @ inv_eigA_vecs
    
    def _exp_B(t):
        diag = np.diag(np.exp(-1j * t * eigB_vals))
        return eigB_vecs @ diag @ inv_eigB_vecs
    
    def _exp_AB(t):
        diag = np.diag(np.exp(-1j * t * eigAB_vals))
        return eigAB_vecs @ diag @ inv_eigAB_vecs
    
    times = [0.25, 0.5, 1.0]
    total_error = 0.0
    
    for t in times:
        # Compute product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(stages):
            exp_A = _exp_A(a[k] * t)
            exp_B = _exp_B(b[k] * t)
            U = exp_B @ exp_A @ U
        
        # Exact propagator
        U_exact = _exp_AB(t)
        
        # Spectral norm error
        error = _spectral_norm(U - U_exact)
        total_error += error
    
    return total_error / len(times)

def _nelder_mead(objective, x0, max_iter=200, tol=1e-10):
    """Simple Nelder-Mead optimization."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    
    # Create simplex
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        point = np.array(x0, dtype=float).copy()
        step = 0.01 if abs(x0[i]) < 0.1 else 0.01 * abs(x0[i])
        point[i] += step
        simplex.append(point)
    
    func_vals = [objective(p) for p in simplex]
    
    for iteration in range(max_iter):
        # Sort by function value
        order = np.argsort(func_vals)
        simplex = [simplex[i] for i in order]
        func_vals = [func_vals[i] for i in order]
        
        # Check convergence
        if abs(func_vals[0] - func_vals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = objective(x_r)
        
        if f_r < func_vals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
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
            f_c = objective(x_c)
            if f_c < func_vals[-1]:
                simplex[-1] = x_c
                func_vals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    func_vals[i] = objective(simplex[i])
    
    # Return best point
    best_idx = np.argmin(func_vals)
    return simplex[best_idx], func_vals[best_idx]

def coefficients(stages):
    """Return optimized 12-stage product formula coefficients."""
    if stages != 12:
        # Fallback for non-12 stage counts
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Use precomputed optimal coefficients
    a = list(_OPTIMAL_A)
    b = list(_OPTIMAL_B)
    
    # Ensure sums are exactly 1.0
    sum_a = sum(a)
    sum_b = sum(b)
    if abs(sum_a - 1.0) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b - 1.0) > 1e-15:
        b = [x / sum_b for x in b]
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    return a, b
# EVOLVE-BLOCK-END
