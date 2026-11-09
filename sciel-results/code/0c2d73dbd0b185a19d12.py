import numpy as np
import time
import sys

# Deadline guard: evaluator kills after 150s wall-clock. 
# We aim to finish well before. 
_T0 = time.time()
_DEADLINE = 140.0 # 10s safety margin

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) 
    for a 6-site open XXZ Heisenberg chain.
    Anisotropy delta = 0.8, Field h = 0.3.
    """
    N = 6
    delta = 0.8
    field = 0.3
    
    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def site_op(op, site):
        ops = [I] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(i, j):
        """XXZ bond: delta*(Sz_i Sz_j + Sx_i Sx_j + Sy_i Sy_j)"""
        ops_sz = [I] * N
        ops_sz[i] = sz
        ops_sz[j] = sz
        ops_sx = [I] * N
        ops_sx[i] = sx
        ops_sx[j] = sx
        ops_sy = [I] * N
        ops_sy[i] = sy
        ops_sy[j] = sy
        return delta * (kron_all(ops_sz) + kron_all(ops_sx) + kron_all(ops_sy))
    
    # Even bonds: (0,1), (2,3), (4,5)
    B_bonds = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    # Odd bonds: (1,2), (3,4)
    A_bonds = bond_op(1, 2) + bond_op(3, 4)
    
    # Half field on each: h/2 * sum Sz_i
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A = A_bonds + field_op
    B = B_bonds + field_op
    
    return A, B

def _compute_error(a, b, A, B):
    """
    Compute mean spectral norm error at t = 0.25, 0.5, 1.0.
    a, b are lists of length 12.
    """
    times = [0.25, 0.5, 1.0]
    H = A + B
    dim = A.shape[0]
    
    # Pre-compute eigendecompositions for A and B to speed up exponentials
    # Note: A and B are Hermitian, so we use eigh
    eigvals_a, eigvecs_a = np.linalg.eigh(A)
    eigvals_b, eigvecs_b = np.linalg.eigh(B)
    
    # Pre-compute exact propagator components
    eigvals_h, eigvecs_h = np.linalg.eigh(H)
    
    errors = []
    for t in times:
        # Exact: exp(-i t H)
        # U_exact = V * diag(exp(-i t lambda)) * V^dag
        exp_h = np.exp(-1j * t * eigvals_h)
        U_exact = (eigvecs_h * exp_h) @ eigvecs_h.conj().T
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Note: The product is k=1..12. 
        # U = E_b1 E_a1 E_b2 E_a2 ... E_b12 E_a12
        # Since matrix multiplication is associative, we can build it up.
        # To save memory/time, we can apply them sequentially to an identity matrix
        # or just multiply. For 64x64 matrices, sequential multiplication is fine.
        
        U = np.eye(dim, dtype=complex)
        
        # Pre-compute exp(-i t * eigvals) for each unique coefficient? 
        # No, coefficients vary per stage. 
        # But we can precompute exp(-i t * eigvals_a) and exp(-i t * eigvals_b) 
        # and then raise them to the power of the coefficient? 
        # exp(-i a_k t A) = (exp(-i t A))^a_k ? No, only if A is diagonal in the same basis.
        # It is! A is diagonalized by eigvecs_a.
        # So exp(-i a_k t A) = eigvecs_a @ diag(exp(-i a_k t eigvals_a)) @ eigvecs_a^dag
        # We can precompute M_a = eigvecs_a @ diag(exp(-i t eigvals_a)) @ eigvecs_a^dag
        # Then exp(-i a_k t A) = M_a ** a_k ? 
        # Power of a matrix is tricky. Better to just compute the diagonal in the eigenbasis.
        
        # Let's optimize the loop:
        # U = prod (V_b D_b_k V_b^dag) (V_a D_a_k V_a^dag)
        # This is 24 matrix multiplications per time step. 
        # 3 time steps * 24 mults * 64x64 is very fast.
        
        for k in range(12):
            # exp(-i b_k t B)
            exp_vals_b = np.exp(-1j * b[k] * t * eigvals_b)
            E_b = (eigvecs_b * exp_vals_b) @ eigvecs_b.conj().T
            
            # exp(-i a_k t A)
            exp_vals_a = np.exp(-1j * a[k] * t * eigvals_a)
            E_a = (eigvecs_a * exp_vals_a) @ eigvecs_a.conj().T
            
            U = E_b @ E_a @ U
            
        err = np.linalg.norm(U - U_exact, ord=2)
        errors.append(err)
        
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter=2000, tol=1e-6):
    """
    Simple Nelder-Mead implementation.
    f: objective function
    x0: initial guess (list)
    """
    n = len(x0)
    simplex = [x0[:]]
    # Initial simplex
    for i in range(n):
        xi = x0[:]
        # Step size: 5% of magnitude or 0.01 if small
        step = 0.01 if abs(xi[i]) < 0.1 else 0.05 * xi[i]
        if step == 0: step = 0.01
        xi[i] += step
        simplex.append(xi)
        
    fvals = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if _remaining() < 20: # Safety margin for final return
            break
            
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
            
        # Centroid of all but worst
        centroid = np.mean(simplex[:n], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
                    
    # Return best
    best_idx = np.argmin(fvals)
    return simplex[best_idx], fvals[best_idx]

def coefficients(stages):
    """
    Optimize 12-stage product formula coefficients.
    """
    if stages != 12:
        # Fallback: simple equal weights
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Initial guess: Symmetric equal weights
    # a = [1/12]*12, b = [1/12]*12
    # Symmetric params: p0..p5 for a, q0..q5 for b
    # a = [p0, p1, p2, p3, p4, p5, p5, p4, p3, p2, p1, p0]
    # sum(a) = 2*(p0+...+p5) = 1 => p0+...+p5 = 0.5
    
    # Start with uniform: pi = 0.5/6 = 1/12
    p_init = [1.0/12.0] * 6
    q_init = [1.0/12.0] * 6
    x0 = p_init + q_init
    
    def params_to_ab(params):
        p = params[:6]
        q = params[6:12]
        a = [p[0], p[1], p[2], p[3], p[4], p[5], p[5], p[4], p[3], p[2], p[1], p[0]]
        b = [q[0], q[1], q[2], q[3], q[4], q[5], q[5], q[4], q[3], q[2], q[1], q[0]]
        
        # Normalize to ensure sum=1 exactly (handling numerical drift)
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-12:
            a = [x/sa for x in a]
        if abs(sb) > 1e-12:
            b = [x/sb for x in b]
            
        # Clip to [-50, 50] as per constraints
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        
        return a, b

    def objective(params):
        a, b = params_to_ab(params)
        return _compute_error(a, b, A, B)
    
    # Run optimization
    # We allow negative coefficients, which is necessary for 4th order and higher.
    # Nelder-Mead will explore the space.
    
    best_params = x0
    best_err = objective(x0)
    
    # Run Nelder-Mead
    if _remaining() > 30:
        res_params, res_err = _nelder_mead(objective, x0, max_iter=3000)
        if res_err < best_err:
            best_err = res_err
            best_params = res_params
            
    # If we have time, try a second run from a different starting point
    # e.g. slightly perturbed or different initial distribution
    if _remaining() > 60:
        # Try a "bumpy" start: larger middle coefficients
        p_alt = [0.05, 0.1, 0.15, 0.15, 0.1, 0.05] # Sum = 0.6 -> normalized to 0.5
        # Normalize manually for start
        s_p = sum(p_alt)
        p_alt = [x * (0.5/s_p) for x in p_alt]
        
        q_alt = [0.05, 0.1, 0.15, 0.15, 0.1, 0.05]
        s_q = sum(q_alt)
        q_alt = [x * (0.5/s_q) for x in q_alt]
        
        x0_alt = p_alt + q_alt
        res_params2, res_err2 = _nelder_mead(objective, x0_alt, max_iter=2000)
        if res_err2 < best_err:
            best_err = res_err2
            best_params = res_params2
            
    # Finalize
    a, b = params_to_ab(best_params)
    
    return a, b

# Keep a global instance? No, coefficients is called once.
# But we need to make sure we don't rebuild if called multiple times? 
# The task says "coefficients runs in a sandboxed subprocess". 
# It's a single call.

if __name__ == "__main__":
    # Simple self-test if run directly, though not required for evaluation
    a, b = coefficients(12)
    print("a:", a)
    print("b:", b)
    print("sum a:", sum(a))
    print("sum b:", sum(b))
