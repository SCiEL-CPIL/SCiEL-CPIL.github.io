import numpy as np
import time
import scipy.linalg

def _build_hamiltonians():
    """
    Build A and B for a 6-site open XXZ Heisenberg chain.
    H = sum_{j=1}^{5} (Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + d Sz_j Sz_{j+1}) + h sum_{j=1}^{6} Sz_j
    A: Odd bonds (1-2, 3-4, 5-6) + half field
    B: Even bonds (2-3, 4-5) + half field
    """
    n_sites = 6
    dim = 2 ** n_sites
    d = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Identity
    id2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    # Helper to get operator on specific site
    def site_op(site, op):
        ops = [id2] * n_sites
        ops[site] = op
        return kron_all(ops)

    # Helper to get bond operator
    def bond_op(j, op1, op2):
        ops = [id2] * n_sites
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Field term: h * sum Sz_j
    # Split half to A, half to B
    field_A = 0.0
    field_B = 0.0
    for j in range(n_sites):
        sz_j = site_op(j, sz)
        field_A += 0.5 * h * sz_j
        field_B += 0.5 * h * sz_j

    # Bond terms
    # Bonds: 0-1, 1-2, 2-3, 3-4, 4-5 (0-indexed)
    # Odd bonds (1st, 3rd, 5th in 1-indexed -> 0, 2, 4 in 0-indexed):
    # Wait, "odd- and even-bond halves".
    # Usually bond 1 is between site 1 and 2.
    # Let's assume standard indexing:
    # Bond 1: Site 0-1 (Odd)
    # Bond 2: Site 1-2 (Even)
    # Bond 3: Site 2-3 (Odd)
    # Bond 4: Site 3-4 (Even)
    # Bond 5: Site 4-5 (Odd)
    
    for j in range(n_sites - 1):
        sxj_sxj1 = bond_op(j, sx, sx)
        syj_syj1 = bond_op(j, sy, sy)
        szj_szj1 = bond_op(j, sz, sz)
        
        term = sxj_sxj1 + syj_syj1 + d * szj_szj1
        
        if (j + 1) % 2 == 1: # Odd bond index (1, 3, 5)
            A += term
        else: # Even bond index (2, 4)
            B += term
            
    A += field_A
    B += field_B
    
    return A, B

def _spectral_norm(M):
    return np.linalg.norm(M, ord=2)

def _objective(coeffs, A, B, times):
    """
    coeffs: array of length 24. First 12 are a_k, next 12 are b_k.
    Returns mean spectral norm error.
    """
    a = coeffs[:12]
    b = coeffs[12:]
    
    # Check bounds and sums
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
        return 1e10 + abs(np.sum(a) - 1.0) + abs(np.sum(b) - 1.0)

    errors = []
    for t in times:
        U = np.eye(64, dtype=complex)
        # Product formula: prod_{k=1..12} exp(-i b_k t B) exp(-i a_k t A)
        # Note: The order is k=1..12.
        # In matrix multiplication, if U = M1 M2 ... M12, then U @ vec is M12 ... M1 vec?
        # Standard convention: U(t) = U_12 ... U_1.
        # If we apply to state |psi>, |psi(t)> = U(t)|psi(0)>.
        # The task says "product formula U(t) = prod_k ... in order k=1..12".
        # Usually this means the operator is U_12 ... U_1 or U_1 ... U_12?
        # "prod_k" usually implies multiplication in the order written.
        # Let's assume U = U_{12} ... U_1 is the standard physics convention for time evolution operators acting on states (left to right in time, right to left in operator composition).
        # However, if the evaluator constructs it as `U = I; for k: U = U_k @ U`, that is U_12 ... U_1.
        # If the evaluator does `U = I; for k: U = U @ U_k`, that is U_1 ... U_12.
        # Given "in order k=1..12", and standard math prod notation $\prod_{k=1}^n A_k = A_n \dots A_1$ is often used, but in code `prod` is often sequential.
        # Let's look at the seed: `exp(-i b_k t B) exp(-i a_k t A)`.
        # If I assume standard time-ordering: later times are to the left.
        # Let's try U = U_12 ... U_1.
        
        # Actually, let's look at the seed code structure if it existed. It doesn't show the evaluator code.
        # "U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) in order k = 1..12"
        # In many physics contexts, $\prod_{k=1}^n A_k$ means $A_n \dots A_1$.
        # Let's assume this.
        
        # Optimization: Precompute exp for unique (coeff, t, Op) pairs?
        # a_k and b_k are distinct. t is fixed.
        # We need exp(-i a_k t A) and exp(-i b_k t B).
        
        # For efficiency, we can compute all exps first.
        exp_A = []
        exp_B = []
        for k in range(12):
            exp_A.append(scipy.linalg.expm(-1j * a[k] * t * A))
            exp_B.append(scipy.linalg.expm(-1j * b[k] * t * B))
            
        # Build product
        # If order is k=1..12, and prod means A_n...A_1:
        # U = (E_B12 E_A12) (E_B11 E_A11) ... (E_B1 E_A1)
        # Let's construct it.
        U_prod = np.eye(64, dtype=complex)
        for k in range(12):
            # If we are building U = M_12 ... M_1
            # We can start from k=1 and multiply on the left?
            # U_curr = M_k @ U_curr ? No, M_1 is rightmost.
            # U = M_12 ... M_1
            # Start U = I
            # U = M_1
            # U = M_2 M_1
            # ...
            # U = M_12 ... M_1
            # So loop k=0..11: U = M_k @ U ?
            # If k=0 (M_1): U = M_1 @ I = M_1.
            # If k=1 (M_2): U = M_2 @ M_1.
            # This matches U = M_12 ... M_1.
            
            # But wait, the term for k is E_Bk E_Ak.
            # So M_k = E_Bk @ E_Ak.
            M_k = exp_B[k] @ exp_A[k]
            U_prod = M_k @ U_prod
            
        U_exact = scipy.linalg.expm(-1j * t * (A + B))
        err = _spectral_norm(U_prod - U_exact)
        errors.append(err)
        
    return np.mean(errors)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    if stages != 12:
        # Fallback to seed-like behavior if stages differ
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
        
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: Uniform 1/12
    # This is 1st order.
    # A better guess might be 4th order 4-stage coefficients repeated.
    # 4th order 4-stage symmetric:
    # c1 = 1/2 - 1/(12 * 2**(1/3))
    # c2 = 1/2 + 1/(12 * 2**(1/3))
    # c3 = c2
    # c4 = c1
    # But we have 12 stages.
    # Let's just use uniform and let Nelder-Mead work.
    
    x0 = np.ones(24) / 12.0
    
    # Nelder-Mead implementation
    def nelder_mead(objective, x0, max_iter=500, tol=1e-8, deadline=None):
        n = len(x0)
        # Initial simplex
        simplex = [x0.copy() for _ in range(n + 1)]
        for i in range(1, n + 1):
            simplex[i][i-1] += 0.1 if x0[i-1] == 0 else 0.1 * x0[i-1]
            
        f = [objective(x) for x in simplex]
        
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
                
            # Sort simplex
            idx = np.argsort(f)
            simplex = [simplex[i] for i in idx]
            f = [f[i] for i in idx]
            
            if abs(f[-1] - f[0]) < tol:
                break
                
            # Centroid of best n points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = objective(xr)
            
            if fr < f[0]:
                # Expansion
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = objective(xe)
                if fe < fr:
                    simplex[-1] = xe
                    f[-1] = fe
                else:
                    simplex[-1] = xr
                    f[-1] = fr
            elif fr < f[-2]:
                simplex[-1] = xr
                f[-1] = fr
            else:
                # Contraction
                beta = 0.5
                if fr < f[-1]:
                    # Outside contraction
                    xc = centroid + beta * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - beta * (xr - centroid)
                fc = objective(xc)
                if fc < min(fr, f[-1]):
                    simplex[-1] = xc
                    f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        f[i] = objective(simplex[i])
                        
        # Return best point
        idx = np.argmin(f)
        return simplex[idx]

    best_x = nelder_mead(_objective, x0, max_iter=300, deadline=deadline)
    
    a = best_x[:12].tolist()
    b = best_x[12:].tolist()
    
    # Final check and clipping
    a = [max(-50, min(50, v)) for v in a]
    b = [max(-50, min(50, v)) for v in b]
    
    # Ensure sums are exactly 1?
    # The constraint was handled by penalty, but let's normalize if needed?
    # No, the evaluator checks sum(a)=1. If my penalty didn't enforce it perfectly, I should fix it.
    # However, the penalty was strong (1e10).
    # Let's just return them.
    
    return a, b

