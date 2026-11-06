"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula for a 6-site XXZ Heisenberg chain.

    Strategy:
      1. Build A and B (odd/even bond halves + half field).
      2. Precompute eigendecompositions for fast matrix exponentials.
      3. Start from 4th-order Suzuki-Trotter (duplicated to 12 stages).
      4. Full 24-parameter Nelder-Mead optimization on mean spectral-norm error.
      5. Deadline-guarded: return best found before time limit.

    Returns (a, b): two lists of `stages` finite numbers in [-50,50],
    with sum(a)=sum(b)=1.
    """
    deadline = time.time() + 135  # safety margin well under the 150s kill
    S = stages

    # ------------------------------------------------------------------
    # 1. Build A and B (odd / even bond halves + half field).
    # ------------------------------------------------------------------
    def build_AB():
        N = 6
        dim = 2 ** N
        delta = 0.8
        field = 0.3

        I2 = np.eye(2, dtype=complex)
        sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
        sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
        sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)

        def kron_all(ops):
            out = np.array([[1.0]], dtype=complex)
            for op in ops:
                out = np.kron(out, op)
            return out

        def site_op(op, i):
            ops = [I2] * N
            ops[i] = op
            return kron_all(ops)

        def bond(j, j1):
            return (delta / 4.0) * (site_op(sz, j) @ site_op(sz, j1)) + \
                   (1.0 / 4.0) * (site_op(sx, j) @ site_op(sx, j1) +
                                  site_op(sy, j) @ site_op(sy, j1))

        A = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        for j, j1 in [(0, 1), (2, 3), (4, 5)]:
            A += bond(j, j1)
        for j, j1 in [(1, 2), (3, 4)]:
            B += bond(j, j1)
        F = field / 4.0
        for s in range(N):
            A += F * site_op(sz, s)
            B += F * site_op(sz, s)
        return A, B

    def clamp(x, lo=-50.0, hi=50.0):
        return lo if x < lo else (hi if x > hi else x)

    def normalize(v):
        s = sum(v)
        if abs(s) < 1e-15:
            return [1.0 / S] * S
        return [x / s for x in v]

    def valid(v):
        return all(-50.0 <= x <= 50.0 for x in v)

    # ------------------------------------------------------------------
    # 2. Build matrices and precompute eigendecompositions.
    # ------------------------------------------------------------------
    A = B = dim = None
    A_eig = B_eig = AB_eig = None
    try:
        if time.time() < deadline - 20:
            A, B = build_AB()
            dim = A.shape[0]
            AB = A + B
            # Eigendecompositions for fast expm
            A_vals, A_vecs = np.linalg.eig(A)
            B_vals, B_vecs = np.linalg.eig(B)
            AB_vals, AB_vecs = np.linalg.eig(AB)
            A_eig = (A_vals, A_vecs)
            B_eig = (B_vals, B_vecs)
            AB_eig = (AB_vals, AB_vecs)
    except Exception:
        A = B = dim = None
        A_eig = B_eig = AB_eig = None

    # ------------------------------------------------------------------
    # Fast matrix exponential using precomputed eigendecomposition.
    # ------------------------------------------------------------------
    def expm_fast(c, eig_cache):
        """expm(c * M) where eig_cache = (eigenvalues, eigenvectors) of M."""
        vals, vecs = eig_cache
        # expm(c*M) = V * diag(exp(c*vals)) * V^{-1}
        exp_vals = np.exp(c * vals)
        # V * diag(exp_vals) * V^{-1}
        # = (V * diag(exp_vals)) @ inv(V)
        # More numerically stable: solve V^T x = (V * diag(exp_vals))^T
        # But for 64x64, direct inverse is fine
        V = vecs
        V_inv = np.linalg.inv(V)
        # V @ diag(exp_vals) @ V_inv
        # = (V * exp_vals[None, :]) @ V_inv
        result = (V * exp_vals[None, :]) @ V_inv
        return result

    # ------------------------------------------------------------------
    # Product formula evaluation using fast expm.
    # ------------------------------------------------------------------
    def product_U(a, b, t, A_eig, B_eig, dim):
        U = np.eye(dim, dtype=complex)
        for k in range(S):
            if abs(b[k]) > 1e-15:
                U = expm_fast(-1j * b[k] * t, B_eig) @ U
            if abs(a[k]) > 1e-15:
                U = expm_fast(-1j * a[k] * t, A_eig) @ U
        return U

    def mean_error(a, b, A_eig, B_eig, AB_eig, dim):
        tot = 0.0
        for t in (0.25, 0.5, 1.0):
            U = product_U(a, b, t, A_eig, B_eig, dim)
            Ue = expm_fast(-1j * t, AB_eig)
            tot += np.linalg.norm(U - Ue, 2)
        return tot / 3.0

    # ------------------------------------------------------------------
    # Candidate 1: Strang base (proven structure).
    # ------------------------------------------------------------------
    sub = S // 2
    a1 = [0.5 / sub] * S
    b1 = []
    for _ in range(sub):
        b1 += [1.0 / sub, 0.0]
    while len(a1) < S:
        a1.append(0.0); b1.append(0.0)
    a1 = normalize(a1); b1 = normalize(b1)
    best_a, best_b = a1, b1

    # ------------------------------------------------------------------
    # Candidate 2: 4th-order Suzuki-Trotter, duplicated to 12.
    # ------------------------------------------------------------------
    a2 = b2 = None
    try:
        if S >= 6:
            x = (2.0 + 2.0 ** (1.0 / 3.0)) ** (-1.0)
            y = 0.5 - x
            z = 0.5 - 2.0 * x
            # 6-stage 4th order (B,A,B,A,B,A)
            b6 = [y, x, x, x, x, y]
            a6 = [x, z, z, z, z, x]
            # duplicate to 12 (each sub-step weight halved)
            b2 = [v / 2.0 for v in b6] * 2
            a2 = [v / 2.0 for v in a6] * 2
            b2 = normalize(b2); a2 = normalize(a2)
            if valid(a2) and valid(b2):
                best_a, best_b = a2, b2
    except Exception:
        a2 = b2 = None

    # ------------------------------------------------------------------
    # 3. Nelder-Mead optimization on full 24-parameter space.
    # ------------------------------------------------------------------
    def nelder_mead(objective, x0, max_iter=500, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        """Simple Nelder-Mead simplex optimization."""
        n = len(x0)
        # Initial simplex
        simplex = [list(x0)]
        for i in range(n):
            xi = list(x0)
            xi[i] += 0.01 if x0[i] == 0 else 0.01 * abs(x0[i])
            simplex.append(xi)
        
        # Evaluate all vertices
        fvals = [objective(v) for v in simplex]
        
        for it in range(max_iter):
            # Sort by function value
            order = sorted(range(n + 1), key=lambda i: fvals[i])
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[0] - fvals[-1]) < tol:
                break
            
            # Centroid of all but worst
            centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
            
            # Reflection
            xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fr = objective(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
                fe = objective(xe)
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
                xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = objective(simplex[i])
        
        # Return best
        best_idx = min(range(n + 1), key=lambda i: fvals[i])
        return simplex[best_idx], fvals[best_idx]

    if A is not None and A_eig is not None and time.time() < deadline - 30:
        try:
            # Build objective function that handles constraints
            def objective_flat(x):
                """x is 24-dim: first 12 are a, next 12 are b."""
                a = list(x[:S])
                b = list(x[S:2*S])
                # Normalize
                a = normalize(a)
                b = normalize(b)
                # Check bounds
                if not valid(a) or not valid(b):
                    return 1e6
                return mean_error(a, b, A_eig, B_eig, AB_eig, dim)

            # Start from best candidate
            x0 = best_a + best_b
            x0 = normalize(x0[:S]) + normalize(x0[S:])  # ensure sums are 1
            
            # Run Nelder-Mead with time guard
            start_time = time.time()
            max_iter = 300
            best_x, best_f = None, float('inf')
            
            # Manual Nelder-Mead with time checking
            n = 2 * S
            simplex = [list(x0)]
            for i in range(n):
                xi = list(x0)
                xi[i] += 0.01
                simplex.append(xi)
            
            fvals = [objective_flat(v) for v in simplex]
            best_f = min(fvals)
            best_idx = fvals.index(best_f)
            best_x = simplex[best_idx]
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    break
                
                # Sort
                order = sorted(range(n + 1), key=lambda i: fvals[i])
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break
                
                centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
                
                xr = [centroid[j] + 1.0 * (centroid[j] - simplex[-1][j]) for j in range(n)]
                fr = objective_flat(xr)
                if time.time() > deadline - 15:
                    break
                
                if fr < fvals[0]:
                    xe = [centroid[j] + 2.0 * (xr[j] - centroid[j]) for j in range(n)]
                    fe = objective_flat(xe)
                    if time.time() > deadline - 15:
                        break
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
                    xc = [centroid[j] + 0.5 * (simplex[-1][j] - centroid[j]) for j in range(n)]
                    fc = objective_flat(xc)
                    if time.time() > deadline - 15:
                        break
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                            fvals[i] = objective_flat(simplex[i])
                            if time.time() > deadline - 15:
                                break
                
                # Track best
                cur_best_idx = min(range(n + 1), key=lambda i: fvals[i])
                if fvals[cur_best_idx] < best_f:
                    best_f = fvals[cur_best_idx]
                    best_x = simplex[cur_best_idx]
            
            if best_x is not None:
                a_opt = normalize(list(best_x[:S]))
                b_opt = normalize(list(best_x[S:2*S]))
                if valid(a_opt) and valid(b_opt):
                    best_a, best_b = a_opt, b_opt
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 4. Final local polish with coordinate descent if time remains.
    # ------------------------------------------------------------------
    if A is not None and A_eig is not None and time.time() < deadline - 10:
        try:
            pa, pb = best_a[:], best_b[:]
            perr = mean_error(pa, pb, A_eig, B_eig, AB_eig, dim)
            step = 0.01
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                improved = False
                for k in range(S):
                    for d in (-step, step):
                        ta = pa[:]; ta[k] += d
                        ta = normalize(ta)
                        if valid(ta):
                            e = mean_error(ta, pb, A_eig, B_eig, AB_eig, dim)
                            if e < perr - 1e-12:
                                pa, perr = ta, e
                                improved = True
                    for d in (-step, step):
                        tb = pb[:]; tb[k] += d
                        tb = normalize(tb)
                        if valid(tb):
                            e = mean_error(pa, tb, A_eig, B_eig, AB_eig, dim)
                            if e < perr - 1e-12:
                                pb, perr = tb, e
                                improved = True
                if not improved:
                    step *= 0.7
                    if step < 1e-6:
                        break
            best_a, best_b = pa, pb
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final consistency + bounds enforcement.
    # ------------------------------------------------------------------
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    best_a = [clamp(x) for x in best_a]
    best_b = [clamp(x) for x in best_b]
    # Re-normalise after clamping to restore sum = 1.
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    return best_a, best_b
# EVOLVE-BLOCK-END
