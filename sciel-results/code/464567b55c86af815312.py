"""
12-stage product formula for 6-site XXZ Heisenberg chain.
Uses direct numerical optimization (Nelder-Mead) to minimize the mean spectral-norm
error at t = 0.25, 0.5, 1.0, subject to sum(a) = sum(b) = 1 and bounds [-50, 50].
"""
import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Build A (odd bonds) and B (even bonds) for a 6-site open XXZ Heisenberg chain
    with anisotropy 0.8 and field 0.3.
    
    The Hamiltonian is:
      H = sum_{i=0}^{4} [Jx (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y) + Jz S_i^z S_{i+1}^z]
          + h * sum_{i=0}^{5} S_i^z
    
    with Jx = Jy = 1, Jz = 0.8, h = 0.3.
    
    Split into odd bonds (0-1, 2-3, 4-5) and even bonds (1-2, 3-4),
    each including half of the uniform field.
    
    A = odd bonds + half field
    B = even bonds + half field
    """
    # 6 sites, each spin-1/2
    # Basis: |000000> to |111111>, 2^6 = 64 states
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    n = 6
    dim = 2 ** n
    
    def op_on_site(op, site):
        """Apply operator `op` on `site` (0-indexed from left) to the full Hilbert space."""
        # Build as tensor product: sites 0,1,...,5
        # site 0 is the leftmost
        ops = [I2] * n
        ops[site] = op
        result = ops[0]
        for i in range(1, n):
            result = np.kron(result, ops[i])
        return result
    
    Jx = 1.0
    Jy = 1.0
    Jz = 0.8
    h = 0.3
    
    # Odd bonds: (0,1), (2,3), (4,5)
    A = np.zeros((dim, dim), dtype=complex)
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        # Jx * (S_i^x S_j^x + S_i^y S_j^y) + Jz * S_i^z S_j^z
        # S = sigma/2
        sx_i = op_on_site(sx / 2, i)
        sx_j = op_on_site(sx / 2, j)
        sy_i = op_on_site(sy / 2, i)
        sy_j = op_on_site(sy / 2, j)
        sz_i = op_on_site(sz / 2, i)
        sz_j = op_on_site(sz / 2, j)
        A += Jx * (sx_i @ sx_j + sy_i @ sy_j) + Jz * (sz_i @ sz_j)
    
    # Half field on A
    for i in range(n):
        sz_i = op_on_site(sz / 2, i)
        A += (h / 2) * sz_i
    
    # Even bonds: (1,2), (3,4)
    B = np.zeros((dim, dim), dtype=complex)
    for (i, j) in [(1, 2), (3, 4)]:
        sx_i = op_on_site(sx / 2, i)
        sx_j = op_on_site(sx / 2, j)
        sy_i = op_on_site(sy / 2, i)
        sy_j = op_on_site(sy / 2, j)
        sz_i = op_on_site(sz / 2, i)
        sz_j = op_on_site(sz / 2, j)
        B += Jx * (sx_i @ sx_j + sy_i @ sy_j) + Jz * (sz_i @ sz_j)
    
    # Half field on B
    for i in range(n):
        sz_i = op_on_site(sz / 2, i)
        B += (h / 2) * sz_i
    
    return A, B


def _compute_error(a, b, A, B, times):
    """
    Compute the mean spectral-norm error for the given coefficients.
    U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    Error = mean over times of ||U(t) - exp(-i t (A+B))||_2
    """
    dim = A.shape[0]
    total_error = 0.0
    n_times = len(times)
    
    for t in times:
        # Compute exact propagator
        H = A + B
        # exp(-i t H)
        # Use eigendecomposition for accuracy
        eigs, evecs = np.linalg.eigh(H)
        U_exact = evecs @ np.diag(np.exp(-1j * t * eigs)) @ evecs.conj().T
        
        # Compute product formula
        U_prod = np.eye(dim, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            eigs_b, evecs_b = np.linalg.eigh(B)
            exp_b = evecs_b @ np.diag(np.exp(-1j * b[k] * t * eigs_b)) @ evecs_b.conj().T
            # exp(-i a_k t A)
            eigs_a, evecs_a = np.linalg.eigh(A)
            exp_a = evecs_a @ np.diag(np.exp(-1j * a[k] * t * eigs_a)) @ evecs_a.conj().T
            U_prod = exp_b @ exp_a @ U_prod
        
        # Spectral norm of difference
        diff = U_prod - U_exact
        # Spectral norm = largest singular value
        try:
            s = np.linalg.svd(diff, compute_uv=False)
            total_error += s[0]
        except:
            total_error += np.inf
    
    return total_error / n_times


def coefficients(stages):
    """
    Return (a, b) lists of length `stages` (expected 12) with sum(a)=sum(b)=1.
    Uses Nelder-Mead optimization to minimize the mean spectral-norm error.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    times = [0.25, 0.5, 1.0]
    
    # Precompute eigendecompositions of A and B for speed
    eigs_a, evecs_a = np.linalg.eigh(A)
    eigs_b, evecs_b = np.linalg.eigh(B)
    H = A + B
    eigs_h, evecs_h = np.linalg.eigh(H)
    
    def fast_error(a, b):
        """Fast error computation using precomputed eigendecompositions."""
        dim = A.shape[0]
        total_error = 0.0
        n_times = len(times)
        
        for t in times:
            # Exact propagator
            U_exact = evecs_h @ np.diag(np.exp(-1j * t * eigs_h)) @ evecs_h.conj().T
            
            # Product formula
            U_prod = np.eye(dim, dtype=complex)
            for k in range(12):
                exp_b = evecs_b @ np.diag(np.exp(-1j * b[k] * t * eigs_b)) @ evecs_b.conj().T
                exp_a = evecs_a @ np.diag(np.exp(-1j * a[k] * t * eigs_a)) @ evecs_a.conj().T
                U_prod = exp_b @ exp_a @ U_prod
            
            diff = U_prod - U_exact
            try:
                s = np.linalg.svd(diff, compute_uv=False)
                total_error += s[0]
            except:
                total_error += 1e6
        
        return total_error / n_times
    
    # Parameterize to enforce sum constraints:
    # a = 1/12 + delta_a where sum(delta_a) = 0
    # b = 1/12 + delta_b where sum(delta_b) = 0
    # Use 11 free parameters for each (last is determined)
    
    def params_to_coeffs(params):
        """Convert 22 parameters to 12 a's and 12 b's with sum=1 each."""
        da = params[:11]
        db = params[11:22]
        
        a = [1.0 / 12.0] * 12
        b = [1.0 / 12.0] * 12
        
        for i in range(11):
            a[i] += da[i]
            b[i] += db[i]
        # Last element absorbs the negative sum
        a[11] -= sum(da)
        b[11] -= sum(db)
        
        # Clip to [-50, 50] and renormalize if needed
        a = np.array(a)
        b = np.array(b)
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        # Ensure sums are exactly 1
        a_sum = sum(a)
        b_sum = sum(b)
        if abs(a_sum) > 1e-15:
            a = a / a_sum
        if abs(b_sum) > 1e-15:
            b = b / b_sum
        
        return list(a), list(b)
    
    def objective(params):
        a, b = params_to_coeffs(params)
        return fast_error(a, b)
    
    # Initial guess: symmetric 4th-order Suzuki-Trotter inspired
    # Use a simple symmetric pattern as starting point
    # c1 = 1/(2*(1 - 2^(1/3)))
    c1 = 1.0 / (2.0 * (1.0 - 2.0 ** (1.0 / 3.0)))
    c2 = (c1 - 1.0) / 2.0
    d1 = 1.0 - 2.0 * c1
    
    # Build a 12-stage symmetric pattern from 4th-order coefficients
    # The 4th-order formula: exp(c2 B) exp(c1 A) exp(d1 B) exp(c1 A) exp(c2 B)
    # In our product form exp(b_k B) exp(a_k A), we need to map this.
    
    # Let's use a simpler symmetric initialization
    # Start with uniform and let optimization find the structure
    x0 = np.zeros(22)
    
    # Run Nelder-Mead
    from scipy.optimize import minimize
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Give Nelder-Mead a reasonable budget
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
        
        # Check time
        if time.time() > deadline - 20:
            a, b = params_to_coeffs(best_x)
            return a, b
        
        # Try a few more local optimizations from different starts
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            perturbation = np.random.randn(22) * 0.1
            x_trial = best_x + perturbation
            # Re-parameterize to maintain constraints
            a_t, b_t = params_to_coeffs(x_trial)
            
            result2 = minimize(
                objective,
                x_trial,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        
    except Exception as e:
        # Fallback to initial guess
        pass
    
    a, b = params_to_coeffs(best_x)
    
    # Final validation
    a = np.array(a)
    b = np.array(b)
    a = np.clip(a, -50, 50)
    b = np.clip(b, -50, 50)
    a = a / sum(a)
    b = b / sum(b)
    
    return list(a), list(b)
