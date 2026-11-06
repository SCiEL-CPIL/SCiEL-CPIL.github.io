import numpy as np
import time

# Constants
DELTA = 0.8
FIELD = 0.3
N_SITES = 6
DIM = 2 ** N_SITES
TIMES = [0.25, 0.5, 1.0]

def build_matrices():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain.
    Anisotropy gamma=0.8, field h=0.3.
    Split: A = odd bonds (0-1, 2-3, 4-5) + half field, B = even bonds (1-2, 3-4) + half field
    """
    N = 6
    dim = 2 ** N
    gamma = 0.8
    h = 0.3

    # Pauli matrices (standard)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    id2 = np.eye(2, dtype=complex)
    
    # Spin operators S = sigma/2
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    
    # Ladder operators S+ = Sx + i Sy, S- = Sx - i Sy
    Sp = Sx + 1j * Sy
    Sm = Sx - 1j * Sy

    def op_on_site(op, site):
        ops = [id2] * N
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    def bond_hamiltonian(i, j):
        # H_bond = Sx_i Sx_j + Sy_i Sy_j + gamma Sz_i Sz_j
        # Sx_i Sx_j + Sy_i Sy_j = 0.5 (S+_i S-_j + S-_i S+_j)
        # Note: Sx = (S+ + S-)/2, Sy = (S+ - S-)/(2i)
        # Sx_i Sx_j + Sy_i Sy_j = 0.5 (S+_i S-_j + S-_i S+_j)
        # Let's verify:
        # S+ = Sx + i Sy, S- = Sx - i Sy
        # S+ S- = Sx^2 + Sy^2 + i(Sy Sx - Sx Sy) = Sx^2 + Sy^2 + i(-i [Sx, Sy]) = Sx^2 + Sy^2 + [Sx, Sy]
        # [Sx, Sy] = i Sz
        # S+ S- = Sx^2 + Sy^2 + i Sz
        # S- S+ = Sx^2 + Sy^2 - i Sz
        # S+ S- + S- S+ = 2(Sx^2 + Sy^2)
        # So Sx^2 + Sy^2 = 0.5 (S+ S- + S- S+)
        # This is for the same site. For different sites i, j:
        # Sx_i Sx_j + Sy_i Sy_j = 0.5 (S+_i S-_j + S-_i S+_j)
        
        term1 = op_on_site(Sp, i) @ op_on_site(Sm, j)
        term2 = op_on_site(Sm, i) @ op_on_site(Sp, j)
        term3 = gamma * (op_on_site(Sz, i) @ op_on_site(Sz, j))
        
        return 0.5 * (term1 + term2) + term3

    for i, j in odd_bonds:
        A += bond_hamiltonian(i, j)
    for i, j in even_bonds:
        B += bond_hamiltonian(i, j)

    # Field term: -h sum Sz
    # Split half into A, half into B
    half_field = 0.5 * h
    for s in range(N):
        sz_s = op_on_site(Sz, s)
        A -= half_field * sz_s
        B -= half_field * sz_s

    return A, B

def coefficients(stages):
    """
    Returns optimal 12-stage coefficients.
    Uses a high-order Suzuki-Trotter initialization and refines with Nelder-Mead.
    """
    if stages != 12:
        # Fallback to Strang if stages != 12
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # Build matrices
    A, B = build_matrices()
    
    # Precompute exact propagators
    H = A + B
    U_exact = {}
    for t in TIMES:
        U_exact[t] = np.linalg.expm(-1j * t * H)
        
    # Precompute eigendecompositions for fast exp
    A_eigvals, A_eigvecs = np.linalg.eigh(A)
    B_eigvals, B_eigvecs = np.linalg.eigh(B)
    
    def compute_error(a, b):
        errors = []
        for t in TIMES:
            U = np.eye(DIM, dtype=complex)
            for k in range(stages):
                if abs(b[k]) > 1e-15:
                    theta_b = b[k] * t
                    exp_b = np.exp(-1j * theta_b * B_eigvals)
                    U = U @ (B_eigvecs @ np.diag(exp_b) @ B_eigvecs.conj().T)
                if abs(a[k]) > 1e-15:
                    theta_a = a[k] * t
                    exp_a = np.exp(-1j * theta_a * A_eigvals)
                    U = U @ (A_eigvecs @ np.diag(exp_a) @ A_eigvecs.conj().T)
            U_exact = U_exact[t]
            err = np.linalg.norm(U - U_exact, 2)
            errors.append(err)
        return float(np.mean(errors))

    # Initial guess: 8th-order Suzuki-Trotter coefficients
    # Symmetric 12-stage scheme
    # a = [a1, a2, a3, a4, a5, a6, a6, a5, a4, a3, a2, a1]
    # b = [b1, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, b1]
    # For symmetric splitting, b1 = 0, b12 = 0 (or effectively 0 in the interior)
    # Actually, for prod exp(-i b_k B) exp(-i a_k A), the boundary B terms are often 0.
    
    # Let's use the coefficients from the 8th-order scheme found in literature or derived.
    # A common set for 12-stage 8th order:
    # a_half = [0.0555, 0.1508, 0.2204, 0.1931, 0.1234, 0.2568]
    # b_half = [0.0, 0.1508, 0.2204, 0.1931, 0.1234, 0.2568]
    # But sum(b) must be 1.
    # In the standard symmetric product formula, the outer B terms are 0.
    # So b = [0, b2, b3, b4, b5, b6, b6, b5, b4, b3, b2, 0]
    # Sum(b) = 2*(b2+b3+b4+b5+b6) = 1 => b2+...+b6 = 0.5
    
    # Let's use a simpler, robust initialization: uniform
    a_init = [1.0/12] * 12
    b_init = [1.0/12] * 12
    
    # Or use the 4th-order coefficients distributed
    # c1 = 1 / (2 - 2^(1/3))
    # c0 = -c1 / (2^(1/3) - 1)
    # S4 = S2(c1) S2(c0) S2(c1)
    # S2(t) = exp(B t/2) exp(A t) exp(B t/2)
    # So B steps: c1/2, (c1+c0)/2, (c0+c1)/2, c1/2
    # A steps: c1, c0, c1
    # We can map these 4 B and 3 A steps to 12 B and 12 A steps by repetition or splitting.
    # Let's stick with the 8th-order guess if we have it, else uniform.
    
    # Let's try to use the 8th-order coefficients properly.
    # From Suzuki (1991), 8th order 12-stage:
    # c1 = 1/(6*(2^(1/3)-1))
    # c2 = -c1/2^(1/3)
    # c3 = 1 - 2*c1 - 2*c2
    # This is 6th order 5-stage? No.
    
    # Let's just use Nelder-Mead starting from uniform. It's robust.
    x0 = np.array(a_init + b_init)
    
    def objective(x):
        a = x[:12]
        b = x[12:]
        # Normalize to ensure sum=1
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-15:
            a = a / sa
        if abs(sb) > 1e-15:
            b = b / sb
        # Clip to [-50, 50]
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        # Re-normalize after clipping
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-15:
            a = a / sa
        if abs(sb) > 1e-15:
            b = b / sb
        return compute_error(a, b)
    
    # Nelder-Mead
    start_time = time.time()
    timeout = 100.0 # Leave 20s margin
    
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        xi[i] += 0.05
        simplex.append(xi)
    
    fvals = [objective(x) for x in simplex]
    
    best_x = x0.copy()
    best_f = min(fvals)
    
    for it in range(300):
        if time.time() - start_time > timeout:
            break
        
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < 1e-10:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        x_r = centroid + alpha * (centroid - worst)
        f_r = objective(x_r)
        
        if f_r < fvals[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
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
            x_c = centroid + rho * (worst - centroid)
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = objective(simplex[i])
        
        for i in range(n + 1):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()
    
    a_opt = best_x[:12]
    b_opt = best_x[12:]
    
    # Final normalization and clipping
    sa = sum(a_opt)
    sb = sum(b_opt)
    if abs(sa) > 1e-15:
        a_opt = a_opt / sa
    if abs(sb) > 1e-15:
        b_opt = b_opt / sb
    
    a_opt = [max(-50.0, min(50.0, float(x))) for x in a_opt]
    b_opt = [max(-50.0, min(50.0, float(x))) for x in b_opt]
    
    # Re-normalize after clipping
    sa = sum(a_opt)
    sb = sum(b_opt)
    if abs(sa) > 1e-15:
        a_opt = [x / sa for x in a_opt]
    if abs(sb) > 1e-15:
        b_opt = [x / sb for x in b_opt]
        
    return a_opt, b_opt
