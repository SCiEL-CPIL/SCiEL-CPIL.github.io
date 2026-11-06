import numpy as np
import time
from scipy.optimize import minimize

def _build_matrices():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(mats):
        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    # XXZ interaction: (sx_i sx_j + sy_i sy_j)/2 + anis * sz_i sz_j
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    # Wait, for 6 sites with open boundary: bonds are (0,1),(1,2),(2,3),(3,4),(4,5)
    # Odd bonds: (0,1), (2,3), (4,5)
    # Even bonds: (1,2), (3,4)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    def bond_term(i, j, anis):
        """XXZ bond term between sites i and j."""
        terms = []
        # sx_i sx_j / 2
        mats = [I2] * N
        mats[i] = sx
        mats[j] = sx
        terms.append(0.5 * kron_all(mats))
        # sy_i sy_j / 2
        mats = [I2] * N
        mats[i] = sy
        mats[j] = sy
        terms.append(0.5 * kron_all(mats))
        # anis * sz_i sz_j
        mats = [I2] * N
        mats[i] = sz
        mats[j] = sz
        terms.append(anis * kron_all(mats))
        return sum(terms)
    
    # Odd bonds for A: (0,1), (2,3), (4,5)
    for i, j in [(0,1), (2,3), (4,5)]:
        A += bond_term(i, j, anis)
    
    # Even bonds for B: (1,2), (3,4)
    for i, j in [(1,2), (3,4)]:
        B += bond_term(i, j, anis)
    
    # Uniform field: -field * sum sz_i, split half to A and half to B
    for site in range(N):
        mats = [I2] * N
        mats[site] = sz
        field_term = -field * kron_all(mats)
        A += 0.5 * field_term
        B += 0.5 * field_term
    
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error."""
    errors = []
    for t in times:
        U = np.eye(2**6, dtype=complex)
        for k in range(len(a)):
            U = np.linalg.multi_dot([np.exp(-1j * b[k] * t * B), np.exp(-1j * a[k] * t * A), U])
        exact = np.exp(-1j * t * (A + B))
        err = np.linalg.norm(U - exact, 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    A, B = _build_matrices()
    times = [0.25, 0.5, 1.0]
    N = 64
    
    # Precompute matrix exponentials for speed
    # We'll cache exp(-i * x * A) and exp(-i * x * B) for various x
    
    # Initial guess: symmetric Strang-like
    # 12 stages: use symmetric pattern
    a0 = [0.0] * stages
    b0 = [0.0] * stages
    # Simple: equal weights
    a0 = [1.0/stages] * stages
    b0 = [1.0/stages] * stages
    
    best_a = a0[:]
    best_b = b0[:]
    best_err = _compute_error(best_a, best_b, A, B, times)
    
    # Try a few hand-crafted symmetric high-order schemes first
    # 6th order symmetric 12-factor scheme
    # Use composition of Strang splittings
    # c = 1/(2*(1-2^(1/3)))
    c = 1.0 / (2.0 * (1.0 - 2.0**(1.0/3.0)))
    d = (1.0 - c) / 2.0
    
    # Suzuki 6th order: S6 = S2(c) * S2(1-2c) * S2(c) where S2 is Strang
    # But our format is different. Let me just use optimization.
    
    # Use Nelder-Mead or similar on the 24 parameters with constraints
    # Actually, let's use a simpler approach: parameterize symmetrically
    
    # For symmetry: a[k] = a[11-k], b[k] = b[11-k] for k=0..11
    # So we only need 6 independent a's and 6 independent b's
    # Plus constraints: sum(a) = 1, sum(b) = 1
    
    # Let me use scipy minimize with bounds and constraints
    
    def objective(params):
        if time.time() > deadline:
            return best_err
        # params: 12 a's and 12 b's
        a = params[:stages]
        b = params[stages:]
        # Enforce symmetry
        for k in range(6):
            a[stages-1-k] = a[k]
            b[stages-1-k] = b[k]
        # Enforce sum constraint by normalizing
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        # Check bounds
        for x in a + b:
            if x < -50 or x > 50:
                return 1e10
        return _compute_error(a, b, A, B, times)
    
    # Initial point: symmetric equal
    x0 = [1.0/stages]*stages + [1.0/stages]*stages
    
    # Use Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if time.time() < deadline - 5.0:
            a_opt = result.x[:stages]
            b_opt = result.x[stages:]
            # Enforce symmetry
            for k in range(6):
                a_opt[stages-1-k] = a_opt[k]
                b_opt[stages-1-k] = b_opt[k]
            sa = sum(a_opt)
            sb = sum(b_opt)
            if abs(sa) > 1e-15 and abs(sb) > 1e-15:
                a_opt = [x/sa for x in a_opt]
                b_opt = [x/sb for x in b_opt]
            # Check bounds
            valid = all(-50 <= x <= 50 for x in a_opt + b_opt)
            if valid:
                err = _compute_error(a_opt, b_opt, A, B, times)
                if err < best_err:
                    best_a = a_opt[:]
                    best_b = b_opt[:]
                    best_err = err
    except Exception:
        pass
    
    # If time remains, try more iterations or different initial points
    if time.time() < deadline - 10.0:
        # Try a 6th-order symmetric scheme as initial guess
        # For 12 (B,A) pairs, use known optimal coefficients
        # Let me try a few different symmetric patterns
        
        # Pattern 1: concentrated in middle
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            
            # Random symmetric initialization
            np.random.seed(trial)
            a_rand = np.random.uniform(-0.5, 1.5, 6)
            b_rand = np.random.uniform(-0.5, 1.5, 6)
            a_full = list(a_rand) + list(a_rand[::-1])
            b_full = list(b_rand) + list(b_rand[::-1])
            
            sa = sum(a_full)
            sb = sum(b_full)
            if abs(sa) < 1e-10 or abs(sb) < 1e-10:
                continue
            a_full = [x/sa for x in a_full]
            b_full = [x/sb for x in b_full]
            
            if any(x < -50 or x > 50 for x in a_full + b_full):
                continue
            
            err = _compute_error(a_full, b_full, A, B, times)
            if err < best_err:
                best_a = a_full[:]
                best_b = b_full[:]
                best_err = err
                
                # Refine with Nelder-Mead
                x0r = a_full + b_full
                try:
                    result = minimize(objective, x0r, method='Nelder-Mead',
                                     options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                    a_r = result.x[:stages]
                    b_r = result.x[stages:]
                    for k in range(6):
                        a_r[stages-1-k] = a_r[k]
                        b_r[stages-1-k] = b_r[k]
                    sa = sum(a_r)
                    sb = sum(b_r)
                    if abs(sa) > 1e-15 and abs(sb) > 1e-15:
                        a_r = [x/sa for x in a_r]
                        b_r = [x/sb for x in b_r]
                    if all(-50 <= x <= 50 for x in a_r + b_r):
                        err_r = _compute_error(a_r, b_r, A, B, times)
                        if err_r < best_err:
                            best_a = a_r[:]
                            best_b = b_r[:]
                            best_err = err_r
                except Exception:
                    pass
    
    # Final check: ensure constraints
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    # Ensure finite
    best_a = [float(x) if np.isfinite(x) else 0.0 for x in best_a]
    best_b = [float(x) if np.isfinite(x) else 0.0 for x in best_b]
    
    return best_a, best_b
