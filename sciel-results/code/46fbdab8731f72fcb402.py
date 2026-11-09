"""Product-formula coefficient optimizer for 6-site Heisenberg chain."""
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock; use 130s as safety margin.
_DEADLINE = 130.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
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
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    # Odd bonds: (1,2), (3,4)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    # Half field on each: h/2 * sum Sz_i
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A += field_op
    B += field_op
    
    return A, B

def _compute_error(a, b, A, B):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    H = A + B
    errors = []
    for t in times:
        # Exact: exp(-i t H)
        eigvals, eigvecs = np.linalg.eigh(H)
        exact = (eigvecs * np.exp(-1j * t * eigvals)) @ eigvecs.conj().T
        
        # Product formula: prod_k exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B)
            ev_b, vec_b = np.linalg.eigh(B)
            exp_b = (vec_b * np.exp(-1j * b[k] * t * ev_b)) @ vec_b.conj().T
            # exp(-i a_k t A)
            ev_a, vec_a = np.linalg.eigh(A)
            exp_a = (vec_a * np.exp(-1j * a[k] * t * ev_a)) @ vec_a.conj().T
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    return np.mean(errors)

def _try_known_coefficients(A, B):
    """Try several known high-order product formula coefficient sets."""
    candidates = []
    
    # 5th order 12-stage coefficients (symmetric)
    # From literature: for 5th order, we need specific coefficients
    # Let's use a few reasonable symmetric patterns
    
    # Pattern 1: Equal weights
    n = 12
    a1 = [1.0/n] * n
    b1 = [1.0/n] * n
    candidates.append((a1, b1))
    
    # Pattern 2: Symmetric with larger middle
    # Try to match 4th order
    # For 4th order symmetric 12-stage:
    # This is complex, let's try a few heuristics
    
    # Pattern 3: Based on 4th order Suzuki-Trotter
    # For 4th order, the coefficients for a symmetric 2m-stage formula
    # m=6, so 12 stages
    # The 4th order formula has specific alpha and beta values
    # Let's use known 4th order coefficients scaled to 12 stages
    
    # Actually, let's just try optimization
    return candidates

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    if stages != 12:
        # Fallback for other stage counts
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Try known coefficient sets first
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error(best_a, best_b, A, B)
    
    # Try symmetric 4th-order-like coefficients
    # For a symmetric 12-stage formula, use parameters
    # Let's parameterize: a = [p, q, r, s, t, u, u, t, s, r, q, p]
    # b = [v, w, x, y, z, w2, w2, z, y, x, w, v]
    # with sum constraints
    
    # Let's do a simple grid search over a few symmetric patterns
    # Pattern: a_k = 1/12 + c*(k - 6.5)^2 / 42.25 for some c
    # This gives symmetric with larger middle or edges
    
    if _remaining() > 30:
        # Try a few symmetric perturbations
        for c in [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]:
            a_test = []
            for k in range(12):
                val = 1.0/12 + c * (k - 5.5)**2 / 42.25
                a_test.append(val)
            # Normalize
            s = sum(a_test)
            a_test = [x/s for x in a_test]
            b_test = a_test[:]  # Try symmetric a=b
            err = _compute_error(a_test, b_test, A, B)
            if err < best_err:
                best_err = err
                best_a = a_test[:]
                best_b = b_test[:]
    
    if _remaining() > 20:
        # Local optimization using Nelder-Mead style
        # Parameterize symmetric coefficients
        # a = [p0, p1, p2, p3, p4, p5, p5, p4, p3, p2, p1, p0]
        # b = [q0, q1, q2, q3, q4, q5, q5, q4, q3, q2, q1, q0]
        # Constraint: 2*(p0+...+p5) = 1, 2*(q0+...+q5) = 1
        
        def params_to_ab(params):
            p = params[:6]
            q = params[6:12]
            a = [p[0], p[1], p[2], p[3], p[4], p[5], p[5], p[4], p[3], p[2], p[1], p[0]]
            b = [q[0], q[1], q[2], q[3], q[4], q[5], q[5], q[4], q[3], q[2], q[1], q[0]]
            return a, b
        
        def objective(params):
            a, b = params_to_ab(params)
            # Enforce sum constraint by normalizing
            sa = sum(a)
            sb = sum(b)
            if abs(sa) < 1e-15 or abs(sb) < 1e-15:
                return 1e10
            a = [x/sa for x in a]
            b = [x/sb for x in b]
            # Clip to [-50, 50]
            a = [max(-50, min(50, x)) for x in a]
            b = [max(-50, min(50, x)) for x in b]
            return _compute_error(a, b, A, B)
        
        # Start from best found
        # Convert best_a, best_b to symmetric params
        p0 = (best_a[0] + best_a[11]) / 2
        p1 = (best_a[1] + best_a[10]) / 2
        p2 = (best_a[2] + best_a[9]) / 2
        p3 = (best_a[3] + best_a[8]) / 2
        p4 = (best_a[4] + best_a[7]) / 2
        p5 = (best_a[5] + best_a[6]) / 2
        q0 = (best_b[0] + best_b[11]) / 2
        q1 = (best_b[1] + best_b[10]) / 2
        q2 = (best_b[2] + best_b[9]) / 2
        q3 = (best_b[3] + best_b[8]) / 2
        q4 = (best_b[4] + best_b[7]) / 2
        q5 = (best_b[5] + best_b[6]) / 2
        
        x0 = [p0, p1, p2, p3, p4, p5, q0, q1, q2, q3, q4, q5]
        
        # Simple Nelder-Mead
        n = 12
        simplex = [x0[:]]
        for i in range(n):
            xi = x0[:]
            xi[i] += 0.01 if xi[i] != 0 else 0.01
            simplex.append(xi)
        
        fvals = [objective(x) for x in simplex]
        
        max_iter = 200
        for it in range(max_iter):
            if _remaining() < 10:
                break
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[n])
            f_ref = objective(x_ref)
            
            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[n])
                f_exp = objective(x_exp)
                if f_exp < f_ref:
                    simplex[n] = x_exp
                    fvals[n] = f_exp
                else:
                    simplex[n] = x_ref
                    fvals[n] = f_ref
            elif f_ref < fvals[n-1]:
                simplex[n] = x_ref
                fvals[n] = f_ref
            else:
                # Contraction
                rho = 0.5
                x_con = centroid + rho * (simplex[n] - centroid)
                f_con = objective(x_con)
                if f_con < fvals[n]:
                    simplex[n] = x_con
                    fvals[n] = f_con
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
        
        # Get best
        best_idx = np.argmin(fvals)
        best_params = simplex[best_idx]
        a_opt, b_opt = params_to_ab(best_params)
        sa = sum(a_opt)
        sb = sum(b_opt)
        if abs(sa) > 1e-15:
            a_opt = [x/sa for x in a_opt]
        if abs(sb) > 1e-15:
            b_opt = [x/sb for x in b_opt]
        a_opt = [max(-50, min(50, x)) for x in a_opt]
        b_opt = [max(-50, min(50, x)) for x in b_opt]
        
        err_opt = _compute_error(a_opt, b_opt, A, B)
        if err_opt < best_err:
            best_err = err_opt
            best_a = a_opt
            best_b = b_opt
    
    # Final check: ensure sums are exactly 1
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
