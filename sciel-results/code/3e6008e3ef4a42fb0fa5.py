"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    lam = 0.8
    h = 0.3
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    field_term = 0.5 * h
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)
    
    # XXZ bond terms: sx*sx + sy*sy + lam*sz*sz
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    for i in range(0, N-1, 2):
        # bond i, i+1 -> A
        for sx_sx in [sx, sx]:
            pass
        # Build bond operator for sites i, i+1
        bond_ops = [np.eye(2, dtype=complex)] * N
        # sx*sx + sy*sy + lam*sz*sz
        bond_mat = np.kron(sx, sx) + np.kron(sy, sy) + lam * np.kron(sz, sz)
        # Place bond_mat on sites i, i+1
        for j in range(N):
            if j == i:
                bond_ops[j] = sx  # placeholder, we'll use kron properly
            elif j == i+1:
                bond_ops[j] = sx
            else:
                bond_ops[j] = np.eye(2, dtype=complex)
        
        # Better: build using kron directly
        result = np.zeros((2**N, 2**N), dtype=complex)
        for basis in range(2**N):
            for basis2 in range(2**N):
                # This is too slow, use a different approach
                pass
        
        # Use a cleaner approach
        def kron_with_bond(N, i, j, bond_mat):
            """Insert bond_mat (4x4) on sites i,j into 2^N dimensional space."""
            dim = 2**N
            result = np.zeros((dim, dim), dtype=complex)
            for a in range(dim):
                for b in range(dim):
                    # Extract bits for sites i and j
                    ai = (a >> (N-1-i)) & 1
                    bi = (b >> (N-1-i)) & 1
                    aj = (a >> (N-1-j)) & 1
                    bj = (b >> (N-1-j)) & 1
                    # Other sites must match
                    match = True
                    for k in range(N):
                        if k != i and k != j:
                            ak = (a >> (N-1-k)) & 1
                            bk = (b >> (N-1-k)) & 1
                            if ak != bk:
                                match = False
                                break
                    if match:
                        # Index into bond_mat
                        idx_a = ai * 2 + aj
                        idx_b = bi * 2 + bj
                        result[a, b] = bond_mat[idx_a, idx_b]
            return result
        
        A += kron_with_bond(N, i, i+1, bond_mat)
    
    for i in range(1, N-1, 2):
        # bond i, i+1 -> B
        bond_mat = np.kron(sx, sx) + np.kron(sy, sy) + lam * np.kron(sz, sz)
        B += kron_with_bond(N, i, i+1, bond_mat)
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error over given times."""
    total = 0.0
    for t in times:
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
        U_exact = expm(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total += err
    return total / len(times)

def coefficients(stages):
    """Optimize coefficients using Nelder-Mead on the actual error metric."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: symmetric Strang-like
    n = stages
    a0 = np.ones(n) / n
    b0 = np.ones(n) / n
    
    # We need sum(a) = 1 and sum(b) = 1
    # Use Nelder-Mead with penalty for constraint violation
    
    def objective(x):
        a = x[:n]
        b = x[n:]
        # Penalize constraint violation
        pen_a = 100.0 * (np.sum(a) - 1.0)**2
        pen_b = 100.0 * (np.sum(b) - 1.0)**2
        # Penalize out-of-range
        pen_range = 100.0 * (np.sum(np.maximum(0, np.abs(a) - 50))**2 + 
                             np.sum(np.maximum(0, np.abs(b) - 50))**2)
        err = _compute_error(a, b, A, B, times)
        return err + pen_a + pen_b + pen_range
    
    from scipy.optimize import minimize
    
    x0 = np.concatenate([a0, b0])
    
    # Try Nelder-Mead
    best_x = x0.copy()
    best_val = objective(x0)
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 10:
        a = best_x[:n]
        b = best_x[n:]
        # Normalize
        a = a / np.sum(a)
        b = b / np.sum(b)
        # Clip to [-50, 50]
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        # Re-normalize after clipping
        a = a / np.sum(a)
        b = b / np.sum(b)
        return list(a), list(b)
    
    # Try a few more optimization strategies if time permits
    # Try Powell method
    try:
        result2 = minimize(objective, x0, method='Powell',
                          options={'maxiter': 100, 'ftol': 1e-8})
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        a = best_x[:n]
        b = best_x[n:]
        a = a / np.sum(a)
        b = b / np.sum(b)
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        a = a / np.sum(a)
        b = b / np.sum(b)
        return list(a), list(b)
    
    # Try L-BFGS-B with bounds
    try:
        bounds = [(-50, 50)] * (2*n)
        result3 = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-10})
        if result3.fun < best_val:
            best_val = result3.fun
            best_x = result3.x
    except Exception:
        pass
    
    a = best_x[:n]
    b = best_x[n:]
    # Normalize
    a = a / np.sum(a)
    b = b / np.sum(b)
    # Clip to [-50, 50]
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    # Re-normalize after clipping
    a = a / np.sum(a)
    b = b / np.sum(b)
    return list(a), list(b)
# EVOLVE-BLOCK-END
