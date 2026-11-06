"""Optimized product formula for Heisenberg chain time evolution."""
import time
import numpy as np

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz)
        B += 0.5 * field * op_on_site(j, sz)
    
    return A, B

def _precompute_eigs(A, B, H):
    """Precompute eigendecompositions of A, B, and H using eigh for Hermitian matrices."""
    A_evals, A_evecs = np.linalg.eigh(A)
    B_evals, B_evecs = np.linalg.eigh(B)
    H_evals, H_evecs = np.linalg.eigh(H)
    
    return (A_evals, A_evecs), (B_evals, B_evecs), (H_evals, H_evecs)

def _expm_from_eigh(evals, evecs, scale):
    """Compute exp(scale * M) from precomputed eigh decomposition."""
    exp_vals = np.exp(scale * evals)
    return (evecs * exp_vals) @ evecs.conj().T

def _objective_fast(a, b, A_eig, B_eig, H_eig, dim):
    """Fast objective using precomputed eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact propagator
        exact = _expm_from_eigh(H_eig[0], H_eig[1], -1j * t)
        
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Applied in order k = 0, 1, ..., 11
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eigh(A_eig[0], A_eig[1], -1j * a[k] * t)
            exp_b = _expm_from_eigh(B_eig[0], B_eig[1], -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)

def coefficients(stages):
    """
    12-stage product formula with symmetric parameterization and Nelder-Mead optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute eigendecompositions
    A_eig, B_eig, H_eig = _precompute_eigs(A, B, H)
    
    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_eig, B_eig, H_eig, dim)
    
    def symmetric_params_to_full(params):
        """
        Convert 10 symmetric parameters to full 12-length arrays a and b.
        params: [a1, a2, a3, a4, a5, b1, b2, b3, b4, b5]
        a6 is derived to satisfy sum(a)=1.
        Symmetry: a_k = a_{11-k} for k=0..11 (0-indexed).
        a[0]=a[11], a[1]=a[10], a[2]=a[9], a[3]=a[8], a[4]=a[7], a[5]=a[6].
        Sum a = 2*(a[0]+a[1]+a[2]+a[3]+a[4]+a[5]) = 1.
        So a[5] = 0.5 - (a[0]+a[1]+a[2]+a[3]+a[4]).
        """
        a_free = params[:5]
        b_free = params[5:10]
        
        a5 = 0.5 - np.sum(a_free)
        b5 = 0.5 - np.sum(b_free)
        
        a_full = np.array([a_free[0], a_free[1], a_free[2], a_free[3], a_free[4], a5,
                           a5, a_free[4], a_free[3], a_free[2], a_free[1], a_free[0]])
        b_full = np.array([b_free[0], b_free[1], b_free[2], b_free[3], b_free[4], b5,
                           b5, b_free[4], b_free[3], b_free[2], b_free[1], b_free[0]])
        
        return a_full, b_full

    def constrained_obj(params):
        a_full, b_full = symmetric_params_to_full(params)
        return eval_score(a_full, b_full)
    
    # Initial candidate: Equal weights
    # a_full = ones(12)/12.
    # a[0]=1/12. Sum of first 5 = 5/12.
    # a[5] = 0.5 - 5/12 = 1/12.
    # So params = [1/12]*5 + [1/12]*5.
    np.random.seed(42)
    best_params = np.ones(10) / 12.0
    best_score = constrained_obj(best_params)
    best_a, best_b = symmetric_params_to_full(best_params)
    
    # Multi-start Nelder-Mead
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            x0 = best_params.copy()
        else:
            x0 = best_params + np.random.normal(0, 0.05, 10)
        
        # Initialize simplex
        n = 10
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            v[i] += 0.01 * (abs(v[i]) + 1.0)
            simplex.append(v)
        
        fvals = [constrained_obj(v) for v in simplex]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 500
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[idx] for idx in order]
            fvals = [fvals[idx] for idx in order]
            
            if abs(fvals[0] - fvals[-1]) < 1e-12:
                break
            
            if fvals[0] < best_score:
                best_score = fvals[0]
                best_params = simplex[0].copy()
                best_a, best_b = symmetric_params_to_full(best_params)
            
            centroid = np.mean(simplex[:-1], axis=0)
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = constrained_obj(x_r)
            
            if f_r < fvals[0]:
                x_e = centroid + gamma * (x_r - centroid)
                f_e = constrained_obj(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                x_c = centroid + rho * (simplex[-1] - centroid)
                f_c = constrained_obj(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    for j in range(1, n + 1):
                        simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                        fvals[j] = constrained_obj(simplex[j])
    
    # Final cleanup
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    return list(best_a), list(best_b)
