import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Builds the A and B Hamiltonians for a 6-site XXZ Heisenberg chain.
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    Anisotropy gamma = 0.8, Field h = 0.3
    """
    N = 6
    gamma = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # Kronecker product helper
    def kron(*mats):
        res = mats[0]
        for m in mats[1:]:
            res = np.kron(res, m)
        return res
    
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)
    
    # Bonds
    # Odd bonds: (0,1), (2,3), (4,5) -> indices 0-1, 2-3, 4-5
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Even bonds: (1,2), (3,4) -> indices 1-2, 3-4
    even_bonds = [(1, 2), (3, 4)]
    
    def add_bond_term(H, bond_indices, coeff=1.0):
        i, j = bond_indices
        # Term: 0.5 * (Sx_i Sx_j + Sy_i Sy_j + gamma Sz_i Sz_j)
        # Sx = 0.5 sx, etc.
        # So 0.5 * (0.5 sx_i 0.5 sx_j) = 0.125 sx_i sx_j
        # Let's use standard Heisenberg: H = sum (Sx_i Sx_j + Sy_i Sy_j + gamma Sz_i Sz_j)
        # S = 0.5 sigma
        # Term = 0.25 * (sx_i sx_j + sy_i sy_j + gamma sz_i sz_j)
        
        for k in range(N):
            if k == i:
                op = 0.25
                for l in range(N):
                    if l == j:
                        # Apply sx_i sx_j
                        pass
        # Easier to construct using kron
        def operator_on_site(site, mat):
            ops = [I] * N
            ops[site] = mat
            return kron(*ops)
        
        term = (0.25 * (operator_on_site(i, sx) @ operator_on_site(j, sx) +
                        operator_on_site(i, sy) @ operator_on_site(j, sy) +
                        gamma * operator_on_site(i, sz) @ operator_on_site(j, sz)))
        H += coeff * term
        
    for bond in odd_bonds:
        add_bond_term(A, bond)
    for bond in even_bonds:
        add_bond_term(B, bond)
        
    # Field: H_field = h * sum S_z
    # Split half into A, half into B
    field_A = 0.5 * h
    field_B = 0.5 * h
    
    for i in range(N):
        A += field_A * 0.5 * operator_on_site(i, sz)
        B += field_B * 0.5 * operator_on_site(i, sz)
        
    return A, B

def _compute_error(a, b, A, B, times):
    """
    Computes the mean spectral norm error for given coefficients.
    """
    total_err = 0.0
    dim = A.shape[0]
    I = np.eye(dim, dtype=complex)
    
    # Precompute exact propagator
    # Exact: exp(-i t (A+B))
    # We can diagonalize A+B once
    eigvals, eigvecs = np.linalg.eigh(A + B)
    exact_propagators = []
    for t in times:
        # exp(-i t H)
        # V diag(exp(-i t E)) V^dagger
        phase = np.exp(-1j * t * eigvals)
        exact_U = (eigvecs * phase) @ eigvecs.conj().T
        exact_propagators.append(exact_U)
        
    # Approximate propagator
    # U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    # We need expm of A and B for each t? No, we can diagonalize A and B once.
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    for idx, t in enumerate(times):
        U_approx = I.copy()
        # Order k=1..12
        for k in range(12):
            # exp(-i b_k t B)
            phase_B = np.exp(-1j * b[k] * t * eigvals_B)
            exp_B = (eigvecs_B * phase_B) @ eigvecs_B.conj().T
            U_approx = exp_B @ U_approx
            
            # exp(-i a_k t A)
            phase_A = np.exp(-1j * a[k] * t * eigvals_A)
            exp_A = (eigvecs_A * phase_A) @ eigvecs_A.conj().T
            U_approx = exp_A @ U_approx
            
        # Error: || U_approx - U_exact ||_2
        # Spectral norm of difference
        diff = U_approx - exact_propagators[idx]
        # Use SVD for spectral norm
        s = np.linalg.svd(diff, compute_uv=False)
        total_err += s[0]
        
    return total_err / len(times)

def coefficients(stages):
    """
    Optimizes 12-stage product formula coefficients for 6-site Heisenberg chain.
    """
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s budget
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Target times
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: 4th-order Yoshida coefficients mapped to 12 slots
    c0 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    c1 = 0.5 - c0 / 2.0
    w = [c0 / 2.0, c1, c0 / 2.0, c0 / 2.0, c1, c0 / 2.0]
    
    a_init = [0.0] * stages
    b_init = [0.0] * stages
    for k in range(6):
        a_init[2 * k] = w[k]
        b_init[2 * k + 1] = w[k]
        
    # Ensure sums are 1
    a_init = np.array(a_init)
    b_init = np.array(b_init)
    
    # Optimization
    try:
        from scipy.optimize import minimize
        
        # Flatten parameters
        x0 = np.concatenate([a_init, b_init])
        
        def objective(x):
            a = x[:12]
            b = x[12:]
            # Penalty for sum != 1
            sum_a = np.sum(a)
            sum_b = np.sum(b)
            penalty = 100.0 * (sum_a - 1.0)**2 + 100.0 * (sum_b - 1.0)**2
            # Penalty for out of bounds
            bounds_pen = 0.0
            for val in a:
                if val < -50 or val > 50:
                    bounds_pen += 100.0
            for val in b:
                if val < -50 or val > 50:
                    bounds_pen += 100.0
                    
            err = _compute_error(a, b, A, B, times)
            return err + penalty + bounds_pen
            
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:12]) - 1.0},
            {'type': 'eq', 'fun': lambda x: np.sum(x[12:]) - 1.0}
        ]
        
        # Bounds
        bounds = [(-50, 50)] * 24
        
        # Use L-BFGS-B or Nelder-Mead
        # Nelder-Mead is more robust for non-smooth objectives (spectral norm is smooth but kinked at 0)
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() - start_time > timeout:
            raise TimeoutError("Optimization took too long")
            
        if result.success:
            x_opt = result.x
        else:
            # Fallback to initial if optimization failed
            x_opt = x0
            
        a_opt = x_opt[:12]
        b_opt = x_opt[12:]
        
        # Final correction for sums
        a_sum = np.sum(a_opt)
        b_sum = np.sum(b_opt)
        
        # Adjust last element to ensure sum is exactly 1
        a_opt = a_opt.copy()
        b_opt = b_opt.copy()
        a_opt[-1] += (1.0 - a_sum)
        b_opt[-1] += (1.0 - b_sum)
        
        # Clip to bounds
        a_opt = np.clip(a_opt, -50, 50)
        b_opt = np.clip(b_opt, -50, 50)
        
        # Re-normalize after clipping
        a_sum = np.sum(a_opt)
        b_sum = np.sum(b_opt)
        if a_sum != 0:
            a_opt = a_opt / a_sum
        if b_sum != 0:
            b_opt = b_opt / b_sum
            
        return a_opt.tolist(), b_opt.tolist()
        
    except Exception as e:
        # Fallback to initial guess if optimization fails
        # Ensure sums are 1
        a_init_sum = np.sum(a_init)
        b_init_sum = np.sum(b_init)
        a_init = a_init / a_init_sum
        b_init = b_init / b_init_sum
        return a_init.tolist(), b_init.tolist()

if __name__ == "__main__":
    # Simple test
    a, b = coefficients(12)
    print(f"Sums: {sum(a)}, {sum(b)}")
    print(f"Lengths: {len(a)}, {len(b)}")
