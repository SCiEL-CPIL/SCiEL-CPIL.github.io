"""Optimized product formula design for Heisenberg-chain time evolution."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula for a 6-site XXZ Heisenberg chain.

    Strategy:
      1. Build A, B and precompute eigendecompositions for fast expm.
      2. Generate diverse initial candidates (Strang, Suzuki-Trotter, random).
      3. Run Nelder-Mead optimization on the true spectral-norm error
         with deadline guards.
      4. Return best (a, b) with sum=1 and bounds [-50,50].

    Returns (a, b): two lists of `stages` finite numbers in [-50,50],
    with sum(a)=sum(b)=1.
    """
    deadline = time.time() + 140  # safety margin under 150s kill
    S = stages

    # ------------------------------------------------------------------
    # Build A and B (odd / even bond halves + half field).
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
    # Fast matrix exponential via eigendecomposition (precomputed).
    # ------------------------------------------------------------------
    def fast_expm_eig(eigvals, eigvecs, coeff):
        """Compute exp(coeff * M) given eigendecomposition M = V diag(w) V^{-1}."""
        # exp(coeff * M) = V diag(exp(coeff * w)) V^{-1}
        exp_diag = np.exp(coeff * eigvals)
        # V @ diag(exp_diag) @ V^{-1}
        return (eigvecs * exp_diag) @ np.linalg.inv(eigvecs)

    def build_expm_cache(M, max_coeff):
        """Precompute eigendecomposition of M for fast expm at various coefficients."""
        eigvals, eigvecs = np.linalg.eig(M)
        inv_eigvecs = np.linalg.inv(eigvecs)
        return eigvals, eigvecs, inv_eigvecs

    def expm_fast(coeff, eigvals, eigvecs, inv_eigvecs):
        exp_diag = np.exp(coeff * eigvals)
        return eigvecs @ (exp_diag[:, None] * inv_eigvecs)

    # ------------------------------------------------------------------
    # Fast evaluation of product formula error.
    # ------------------------------------------------------------------
    def eval_coefficients_fast(a, b, A_eig, B_eig, AB_eig, dim):
        """Compute mean spectral norm error over t = 0.25, 0.5, 1.0."""
        A_eigvals, A_eigvecs, A_inv = A_eig
        B_eigvals, B_eigvecs, B_inv = B_eig
        AB_eigvals, AB_eigvecs, AB_inv = AB_eig

        total_err = 0.0
        for t in (0.25, 0.5, 1.0):
            # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
            U = np.eye(dim, dtype=complex)
            for k in range(S):
                bk = b[k]
                ak = a[k]
                if bk != 0.0:
                    U = expm_fast(-1j * bk * t, B_eigvals, B_eigvecs, B_inv) @ U
                if ak != 0.0:
                    U = expm_fast(-1j * ak * t, A_eigvals, A_eigvecs, A_inv) @ U
            # Exact propagator
            Ue = expm_fast(-1j * t, AB_eigvals, AB_eigvecs, AB_inv)
            total_err += np.linalg.norm(U - Ue, 2)
        return total_err / 3.0

    # ------------------------------------------------------------------
    # Nelder-Mead optimizer (manual, no scipy dependency).
    # ------------------------------------------------------------------
    def nelder_mead(f, x0, bounds, max_iter=200, tol=1e-10, deadline=None):
        """Simple Nelder-Mead on 2*S parameters (a_0..a_{S-1}, b_0..b_{S-1})."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Initial simplex
        simplex = [x0[:]]
        for i in range(n):
            pert = x0[:]
            if pert[i] != 0:
                pert[i] *= 1.05
            else:
                pert[i] += 0.05
            simplex.append(pert)

        fx = [f(simplex[i]) for i in range(n + 1)]

        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            # Sort by function value
            order = sorted(range(n + 1), key=lambda i: fx[i])
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]

            if abs(fx[0] - fx[-1]) < tol:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fxr = f(xr)
            if fx[0] <= fxr < fx[-2]:
                simplex[-1] = xr
                fx[-1] = fxr
                continue

            # Expansion
            if fxr < fx[0]:
                xe = centroid + gamma * (xr - centroid)
                fxe = f(xe)
                if fxe < fxr:
                    simplex[-1] = xe
                    fx[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx[-1] = fxr
                continue

            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fxc = f(xc)
            if fxc < fx[-1]:
                simplex[-1] = xc
                fx[-1] = fxc
                continue

            # Shrink
            for i in range(1, n + 1):
                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                fx[i] = f(simplex[i])

        # Return best
        order = sorted(range(n + 1), key=lambda i: fx[i])
        return simplex[order[0]], fx[order[0]]

    # ------------------------------------------------------------------
    # Generate initial candidates.
    # ------------------------------------------------------------------
    def generate_candidates():
        cands = []

        # Candidate 1: Strang-like (B first pattern)
        sub = S // 2
        a1 = [0.5 / sub] * S
        b1 = []
        for _ in range(sub):
            b1 += [1.0 / sub, 0.0]
        while len(a1) < S:
            a1.append(0.0); b1.append(0.0)
        a1 = normalize(a1); b1 = normalize(b1)
        cands.append(a1 + b1)

        # Candidate 2: 6-stage 4th-order Suzuki-Trotter, duplicated to 12
        if S >= 6:
            x = (2.0 + 2.0 ** (1.0 / 3.0)) ** (-1.0)
            y = 0.5 - x
            z = 0.5 - 2.0 * x
            b6 = [y, x, x, x, x, y]
            a6 = [x, z, z, z, z, x]
            b2 = [v / 2.0 for v in b6] * 2
            a2 = [v / 2.0 for v in a6] * 2
            b2 = normalize(b2); a2 = normalize(a2)
            if valid(a2) and valid(b2):
                cands.append(a2 + b2)

        # Candidate 3: Uniform split
        a3 = [1.0 / S] * S
        b3 = [1.0 / S] * S
        cands.append(a3 + b3)

        # Candidate 4: Random perturbations around Strang
        rng = np.random.default_rng(42)
        for trial in range(3):
            a4 = [a1[i] + rng.normal(0, 0.05) for i in range(S)]
            b4 = [b1[i] + rng.normal(0, 0.05) for i in range(S)]
            a4 = normalize(a4); b4 = normalize(b4)
            if valid(a4) and valid(b4):
                cands.append(a4 + b4)

        return cands

    # ------------------------------------------------------------------
    # Main optimization loop.
    # ------------------------------------------------------------------
    A = B = dim = None
    A_eig = B_eig = AB_eig = None

    try:
        if time.time() < deadline - 10:
            A, B = build_AB()
            dim = A.shape[0]
            A_eig = build_expm_cache(A, 1.0)
            B_eig = build_expm_cache(B, 1.0)
            AB_eig = build_expm_cache(A + B, 1.0)
    except Exception:
        A = B = dim = None
        A_eig = B_eig = AB_eig = None

    if A is None:
        # Fallback: return Strang base
        sub = S // 2
        a1 = [0.5 / sub] * S
        b1 = []
        for _ in range(sub):
            b1 += [1.0 / sub, 0.0]
        while len(a1) < S:
            a1.append(0.0); b1.append(0.0)
        a1 = normalize(a1); b1 = normalize(b1)
        return a1, b1

    # Wrap the evaluation with constraint handling
    def make_evaluator(a_b_vec):
        def _eval(vec):
            a = normalize(vec[:S])
            b = normalize(vec[S:])
            if not valid(a) or not valid(b):
                return 1e10
            return eval_coefficients_fast(a, b, A_eig, B_eig, AB_eig, dim)
        return _eval

    # Run Nelder-Mead from multiple starting points
    candidates = generate_candidates()
    best_a, best_b = candidates[0][:S], candidates[0][S:]
    best_err = float('inf')

    for start in candidates:
        if time.time() > deadline - 8:
            break
        eval_fn = make_evaluator(start)
        x_opt, f_opt = nelder_mead(eval_fn, start, None, max_iter=150, tol=1e-10, deadline=deadline - 5)
        a_opt = normalize(x_opt[:S])
        b_opt = normalize(x_opt[S:])
        if valid(a_opt) and valid(b_opt) and f_opt < best_err:
            best_a, best_b, best_err = a_opt, b_opt, f_opt

    # Final consistency + bounds enforcement
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    best_a = [clamp(x) for x in best_a]
    best_b = [clamp(x) for x in best_b]
    best_a = normalize(best_a)
    best_b = normalize(best_b)
    return best_a, best_b
# EVOLVE-BLOCK-END
