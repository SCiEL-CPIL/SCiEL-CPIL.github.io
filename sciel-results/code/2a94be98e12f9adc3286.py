"""
Product-formula coefficient optimizer for 6-site Heisenberg chain.

Strategy:
1. Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
   open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
2. Cache eigen-decompositions of A, B, and H=A+B for fast error evaluation.
3. Use an analytical BCH-inspired initial guess (symmetric, high-order Suzuki-like)
   to start near the optimal region, avoiding the slow convergence from equal weights.
4. Perform a two-stage optimization:
   - Stage 1: Coarse Nelder-Mead on symmetric parameters (12 vars) to find the basin.
   - Stage 2: Fine Nelder-Mead on full parameters (24 vars) to polish asymmetric details.
5. Strict time management to ensure completion within 150s.
"""
import numpy as np
import time

_DEADLINE = 145.0  # Safety margin for 150s limit
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field)."""
    N = 6
    delta = 0.8
    field = 0.3
    
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
    
    # B: even bonds (0,1), (2,3), (4,5)
    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    # A: odd bonds (1,2), (3,4)
    A = bond_op(1, 2) + bond_op(3, 4)
    
    half_field = field / 2.0
    field_op = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        field_op += half_field * site_op(sz, i)
    
    A = A + field_op
    B = B + field_op
    
    return A, B

# Precompute eigen-decompositions (cached globally)
_eig_cache = None

def _get_eig():
    """Get cached eigen-decompositions of A, B, and H=A+B."""
    global _eig_cache
    if _eig_cache is None:
        A, B = _build_hamiltonians()
        H = A + B
        ev_A, vec_A = np.linalg.eigh(A)
        ev_B, vec_B = np.linalg.eigh(B)
        ev_H, vec_H = np.linalg.eigh(H)
        dim = A.shape[0]
        _eig_cache = {
            'A': A, 'B': B, 'H': H, 'dim': dim,
            'ev_A': ev_A, 'vec_A': vec_A,
            'ev_B': ev_B, 'vec_B': vec_B,
            'ev_H': ev_H, 'vec_H': vec_H,
        }
    return _eig_cache

def _compute_error_fast(a, b):
    """Compute mean spectral norm error using cached eigen-decompositions."""
    cache = _get_eig()
    ev_A, vec_A = cache['ev_A'], cache['vec_A']
    ev_B, vec_B = cache['ev_B'], cache['vec_B']
    ev_H, vec_H = cache['ev_H'], cache['vec_H']
    dim = cache['dim']
    
    times = [0.25, 0.5, 1.0]
    errors = []
    
    for t in times:
        # Exact propagator
        exact = (vec_H * np.exp(-1j * t * ev_H)) @ vec_H.conj().T
        
        # Product formula
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            exp_b = (vec_B * np.exp(-1j * b[k] * t * ev_B)) @ vec_B.conj().T
            exp_a = (vec_A * np.exp(-1j * a[k] * t * ev_A)) @ vec_A.conj().T
            U = exp_b @ exp_a @ U
        
        err = np.linalg.norm(U - exact, ord=2)
        errors.append(err)
    
    return np.mean(errors)

def _nelder_mead(f, x0, max_iter, step_size, tol=1e-12, deadline_check=True):
    """Nelder-Mead optimization with time budget check."""
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        xi = x0[:]
        xi[i] += step_size
        simplex.append(xi)
    
    fvals = [f(x) for x in simplex]
    best_val = min(fvals)
    best_x = simplex[np.argmin(fvals)][:]
    no_improve = 0
    
    for it in range(max_iter):
        if deadline_check and _remaining() < 5.0:
            break
        
        # Sort
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
        x_ref = centroid + alpha * (centroid - simplex[n])
        f_ref = f(x_ref)
        
        if f_ref < fvals[0]:
            gamma = 2.0
            x_exp = centroid + gamma * (centroid - simplex[n])
            f_exp = f(x_exp)
            if f_exp < f_ref:
                simplex[n] = x_exp
                fvals[n] = f_exp
            else:
                simplex[n] = x_ref
                fvals[n] = f_ref
            no_improve = 0
        elif f_ref < fvals[n]:
            simplex[n] = x_ref
            fvals[n] = f_ref
            no_improve = 0
        else:
            rho = 0.5
            x_con = centroid + rho * (simplex[n] - centroid)
            f_con = f(x_con)
            if f_con < fvals[n]:
                simplex[n] = x_con
                fvals[n] = f_con
                no_improve = 0
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
                no_improve += 1
        
        if fvals[0] < best_val:
            best_val = fvals[0]
            best_x = simplex[0][:]
        
        if no_improve > 30:
            break
    
    return best_x, best_val

def _generate_bch_guess():
    """
    Generate a high-quality initial guess based on 4th-order Suzuki-Trotter logic.
    For a 12-stage formula, we can repeat a 4-stage pattern 3 times.
    The 4th order coefficients for exp(tA)exp(tB) are:
    alpha_1 = alpha_3 = 1 / (6 - 6^(1/3))
    alpha_2 = alpha_4 = -1 / (6 - 6^(1/3))
    beta_1 = beta_3 = 1 / (2 - 6^(1/3))
    beta_2 = beta_4 = 1 / (2 - 6^(1/3))
    
    However, the structure here is exp(-i b_k B) exp(-i a_k A).
    We want to cancel commutators. A symmetric distribution of weights is key.
    """
    c = 6**(1/3)
    a1 = 1.0 / (6.0 - 6.0*c)
    a2 = -1.0 / (6.0 - 6.0*c)
    b1 = 1.0 / (2.0 - 6.0*c)
    b2 = 1.0 / (2.0 - 6.0*c)
    
    # Pattern for one 4-stage block
    # Note: The standard Suzuki formula is symmetric.
    # We map this to our 12 stages.
    # Since we have 12 stages, we can use 3 blocks of 4.
    # But we must ensure sum(a)=1 and sum(b)=1.
    
    a_block = [a1, a2, a2, a1]
    b_block = [b1, b2, b2, b1]
    
    a_guess = a_block * 3
    b_guess = b_block * 3
    
    # Normalize to sum 1
    sa = sum(a_guess)
    sb = sum(b_guess)
    if abs(sa) > 1e-10:
        a_guess = [x/sa for x in a_guess]
    if abs(sb) > 1e-10:
        b_guess = [x/sb for x in b_guess]
        
    return a_guess, b_guess

def coefficients(stages):
    """Optimize 12-stage product formula coefficients."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Initialize cache
    cache = _get_eig()
    
    # --- Phase 1: Analytical Initial Guesses ---
    guesses = []
    
    # 1. BCH/Suzuki-like symmetric guess
    a_suz, b_suz = _generate_bch_guess()
    guesses.append((a_suz, b_suz))
    
    # 2. Equal weights (baseline)
    guesses.append(([1.0/12]*12, [1.0/12]*12))
    
    # 3. Edge-emphasized symmetric (often helps for open chains)
    a_edge = [1.0/12] * 12
    b_edge = [1.0/12] * 12
    for k in range(12):
        # Weights higher at ends
        w = 1.0 + 0.5 * (1 - abs(k - 5.5) / 5.5) 
        a_edge[k] = w
        b_edge[k] = w
    sa = sum(a_edge); sb = sum(b_edge)
    a_edge = [x/sa for x in a_edge]
    b_edge = [x/sb for x in b_edge]
    guesses.append((a_edge, b_edge))

    best_a = None
    best_b = None
    best_err = float('inf')
    
    # Evaluate guesses
    for a_g, b_g in guesses:
        if _remaining() < 5.0: break
        err = _compute_error_fast(a_g, b_g)
        if err < best_err:
            best_err = err
            best_a = a_g[:]
            best_b = b_g[:]
            
    if best_a is None:
        best_a = [1.0/12]*12
        best_b = [1.0/12]*12
        best_err = _compute_error_fast(best_a, best_b)

    # --- Phase 2: Symmetric Optimization (12 params) ---
    # Enforce symmetry: a_k = a_{11-k}, b_k = b_{11-k}
    # This reduces dimensionality and respects the physical symmetry of the chain.
    
    def objective_sym(x):
        # x has 6 elements for a (first half) and 6 for b (first half)
        a_half = x[:6]
        b_half = x[6:12]
        
        a = list(a_half) + list(a_half[::-1])
        b = list(b_half) + list(b_half[::-1])
        
        # Normalize
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        
        # Clip
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        
        # Re-normalize after clipping
        sa = sum(a); sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        
        return _compute_error_fast(a, b)

    # Extract symmetric initial guess
    x0_sym = best_a[:6] + best_b[:6]
    
    # Run coarse symmetric optimization
    x_opt_sym, f_opt_sym = _nelder_mead(objective_sym, x0_sym, max_iter=200, step_size=0.05)
    
    if f_opt_sym < best_err:
        best_err = f_opt_sym
        a_half = x_opt_sym[:6]
        b_half = x_opt_sym[6:12]
        best_a = list(a_half) + list(a_half[::-1])
        best_b = list(b_half) + list(b_half[::-1])
        # Normalize
        sa = sum(best_a); sb = sum(best_b)
        if abs(sa)>1e-15: best_a = [x/sa for x in best_a]
        if abs(sb)>1e-15: best_b = [x/sb for x in best_b]

    # --- Phase 3: Asymmetric Optimization (24 params) ---
    # Start from the best symmetric solution and allow asymmetry.
    
    def objective_full(x):
        a = list(x[:12])
        b = list(x[12:24])
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        sa = sum(a); sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        return _compute_error_fast(a, b)
    
    x0_full = best_a + best_b
    
    # Run fine full optimization
    # Use smaller steps since we are already close
    x_opt_full, f_opt_full = _nelder_mead(objective_full, x0_full, max_iter=300, step_size=0.01)
    
    if f_opt_full < best_err:
        best_err = f_opt_full
        best_a = list(x_opt_full[:12])
        best_b = list(x_opt_full[12:24])
        
    # Final Normalization and Clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Re-normalize after clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
        
    return best_a, best_b
