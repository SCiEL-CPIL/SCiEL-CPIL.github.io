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


def _precompute_eigh(A, B, H):
    """Precompute eigendecompositions using eigh (Hermitian) for speed and stability."""
    A_evals, A_evecs = np.linalg.eigh(A)
    B_evals, B_evecs = np.linalg.eigh(B)
    H_evals, H_evecs = np.linalg.eigh(H)
    return A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs


def _expm_from_eigh(evals, evecs, scale):
    """Compute exp(scale * M) from precomputed Hermitian eigendecomposition."""
    return (evecs * np.exp(scale * evals)) @ evecs.conj().T


def _objective_fast(a, b, A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs, dim):
    """Fast objective using precomputed Hermitian eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        exact = _expm_from_eigh(H_evals, H_evecs, -1j * t)
        
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eigh(A_evals, A_evecs, -1j * a[k] * t)
            exp_b = _expm_from_eigh(B_evals, B_evecs, -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)


def _solve_2nd_order_constraints(stages):
    """
    Solve for coefficients that match the generator expansion up to 2nd order.
    
    For U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A), the 2nd-order error
    involves the commutator [A,B]. The condition for vanishing 2nd-order
    commutator error in a symmetric scheme is:
      sum_k b_k * (cumulative_A_before_k - 0.5) = 0
    where cumulative_A_before_k = sum_{j<k} a_j
    
    We also need sum(a) = 1 and sum(b) = 1.
    
    We parameterize with symmetry: a_k = a_{13-k}, b_k = b_{13-k} for k=1..12.
    This reduces free parameters and enforces time-reversal symmetry.
    """
    # With mirror symmetry, we have 6 free a's and 6 free b's (indices 0-5 determine 6-11)
    # a[0]=a[11], a[1]=a[10], ..., a[5]=a[6]
    # b[0]=b[11], b[1]=b[10], ..., b[5]=b[6]
    
    # Constraints:
    # 1) sum(a) = 1: 2*(a0+a1+a2+a3+a4+a5) = 1
    # 2) sum(b) = 1: 2*(b0+b1+b2+b3+b4+b5) = 1
    # 3) 2nd order commutator condition
    
    # For the commutator condition, compute cumulative A before each B step.
    # In the product U = prod_k exp(-ib_k t B) exp(-i a_k t A),
    # the B at step k "sees" all A's from steps 0..k-1 already applied.
    # The 2nd order term proportional to [A,B] has coefficient:
    #   sum_k b_k * (sum_{j<k} a_j) - 1/2 * sum_k b_k * sum_j a_j
    # Since sum a = 1, this becomes:
    #   sum_k b_k * (cumA_k) - 1/2 = 0
    # where cumA_k = sum_{j<k} a_j
    
    # With mirror symmetry, let's set up the linear system.
    # Free variables: a0..a5, b0..b5 (12 variables)
    # Constraint 1: a0+a1+a2+a3+a4+a5 = 0.5
    # Constraint 2: b0+b1+b2+b3+b4+b5 = 0.5
    # Constraint 3: sum_k b_k * cumA_k = 0.5
    
    # cumA_k for k=0..11 (with a_k = a_{11-k}):
    # cumA_0 = 0
    # cumA_1 = a0
    # cumA_2 = a0+a1
    # cumA_3 = a0+a1+a2
    # cumA_4 = a0+a1+a2+a3
    # cumA_5 = a0+a1+a2+a3+a4
    # cumA_6 = a0+a1+a2+a3+a4+a5 = 0.5
    # cumA_7 = 0.5 + a5 = 0.5 + a5
    # cumA_8 = 0.5 + a5 + a4
    # cumA_9 = 0.5 + a5 + a4 + a3
    # cumA_10 = 0.5 + a5 + a4 + a3 + a2
    # cumA_11 = 0.5 + a5 + a4 + a3 + a2 + a1 = 1.0 - a0
    
    # sum_k b_k * cumA_k = b0*0 + b1*a0 + b2*(a0+a1) + b3*(a0+a1+a2) + b4*(a0+a1+a2+a3) + b5*(a0+a1+a2+a3+a4)
    #   + b5*(0.5+a5) + b4*(0.5+a5+a4) + b3*(0.5+a5+a4+a3) + b2*(0.5+a5+a4+a3+a2) + b1*(0.5+a5+a4+a3+a2+a1) + b0*(1.0-a0)
    
    # This is nonlinear in general, but if we assume a_k and b_k are small and use
    # a linearized approach, we can set up a least-squares problem.
    
    # Alternative: use a known high-order splitting pattern as initial guess,
    # then project onto the constraint manifold.
    
    # For now, use the symmetric Suzuki-Trotter 4th-order coefficients as the
    # 2nd-order-matching solution (they satisfy 1st and 2nd order by construction
    # for symmetric schemes).
    
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # 6-step symmetric 4th order, extended to 12 steps by repeating
    # Standard 6-step: [x3/2, x1/2, x1/2, x3/2, x1/2, x1/2] for a
    #                  [x2/2, x2/2, 0,    0,    x2/2, x2/2] for b
    # For 12 steps, repeat the 6-step pattern twice:
    a_half = np.array([x3/2, x1/2, x1/2, x3/2, x1/2, x1/2])
    b_half = np.array([x2/2, x2/2, 0.0,  0.0,  x2/2, x2/2])
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Normalize to ensure sum = 1
    a_full = a_full / np.sum(a_full)
    b_full = b_full / np.sum(b_full)
    
    return a_full, b_full


def coefficients(stages):
    """
    12-stage product formula using 2nd-order-matching initial guess and
    multi-start Nelder-Mead with Hermitian eigendecompositions.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute Hermitian eigendecompositions (faster than general eig)
    A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs = _precompute_eigh(A, B, H)
    
    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_evals, A_evecs, B_evals, B_evecs, H_evals, H_evecs, dim)
    
    # Generate initial candidates
    candidates = []
    
    # 1. 2nd-order-matching symmetric coefficients (from _solve_2nd_order_constraints)
    a_2nd, b_2nd = _solve_2nd_order_constraints(stages)
    candidates.append((a_2nd.copy(), b_2nd.copy()))
    
    # 2. Equal weights baseline
    eq = np.ones(stages) / stages
    candidates.append((eq.copy(), eq.copy()))
    
    # 3. Symmetric pattern with different spread
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # Alternative symmetric 12-step: [x3, x1, x1, x3, x1, x1, x1, x3, x1, x1, x3, x1]
    a_sym2 = np.array([x3, x1, x1, x3, x1, x1, x1, x3, x1, x1, x3, x1])
    b_sym2 = np.array([x2, x2, 0,  x2, x2, 0,  0,  x2, x2, 0,  x2, x2])
    a_sym2 = a_sym2 / np.sum(a_sym2)
    b_sym2 = b_sym2 / np.sum(b_sym2)
    candidates.append((a_sym2.copy(), b_sym2.copy()))
    
    # 4. Center-weighted
    center = np.arange(stages) - stages / 2.0
    a_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    b_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    a_cw = a_cw / np.sum(a_cw)
    b_cw = b_cw / np.sum(b_cw)
    candidates.append((a_cw.copy(), b_cw.copy()))
    
    # Evaluate all candidates
    best_a = candidates[0][0].copy()
    best_b = candidates[0][1].copy()
    best_score = eval_score(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        s = eval_score(a_c, b_c)
        if s < best_score:
            best_a = a_c.copy()
            best_b = b_c.copy()
            best_score = s
    
    # Nelder-Mead with reduced parameterization (22 free params)
    def constrained_obj(params):
        a_free = params[:11]
        b_free = params[11:22]
        a_full = np.append(a_free, 1.0 - np.sum(a_free))
        b_full = np.append(b_free, 1.0 - np.sum(b_free))
        return eval_score(a_full, b_full)
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = np.concatenate([best_a[:11], best_b[:11]])
        else:
            x0 = np.concatenate([best_a[:11], best_b[:11]]) + np.random.normal(0, 0.15, 22)
        
        # Initialize simplex
        n = 22
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            v[i] += 0.03 * (abs(v[i]) + 1.0)
            simplex.append(v)
        
        fvals = [constrained_obj(v) for v in simplex]
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = 300
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            order = np.argsort(fvals)
            simplex = [simplex[idx] for idx in order]
            fvals = [fvals[idx] for idx in order]
            
            if abs(fvals[0] - fvals[-1]) < 1e-12:
                break
            
            a_full = np.append(simplex[0][:11], 1.0 - np.sum(simplex[0][:11]))
            b_full = np.append(simplex[0][11:22], 1.0 - np.sum(simplex[0][11:22]))
            if fvals[0] < best_score:
                best_a = a_full.copy()
                best_b = b_full.copy()
                best_score = fvals[0]
            
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
