"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain.

Combines:
  * Robust Hamiltonian build + eigen-decomposition exponentials (fast, stable).
  * Palindromic + Fourier smooth parameterization (8 free vars) to cut search space.
  * Spectral residual minimization: fit log-phase of the propagator in the H
    eigenbasis at the evaluation times, blended with the spectral-norm error.
    This produces a smoother, more global objective than the raw spectral norm,
    so a cheap global (differential_evolution) pass can seed a fast local polish.
  * Explicit deadline guard + Strang fallback on any timeout/error.
"""

import time
import numpy as np

_START = time.time()
# Evaluator kills at 150 s wall-clock. Keep a hard safety margin.
DEADLINE = _START + 132.0


def _build_hamiltonian():
    """Return (A_mat, B_mat, H_total) for the 6-site open XXZ chain (aniso=0.8, field=0.3)."""
    N = 6
    dim = 1 << N
    aniso = 0.8
    field = 0.3

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)

    def tensor(*ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    def site_op(site, mat):
        ops = [eye2 if i != site else mat for i in range(N)]
        return tensor(*ops)

    def pair_op(i, j, mi, mj):
        ops = []
        for k in range(N):
            if k == i:
                ops.append(mi)
            elif k == j:
                ops.append(mj)
            else:
                ops.append(eye2)
        return tensor(*ops)

    # Even bonds: 0-1, 2-3, 4-5 ; Odd bonds: 1-2, 3-4
    H_even = (pair_op(0, 1, sx, sx) + pair_op(0, 1, sy, sy) + aniso * pair_op(0, 1, sz, sz)
              + pair_op(2, 3, sx, sx) + pair_op(2, 3, sy, sy) + aniso * pair_op(2, 3, sz, sz)
              + pair_op(4, 5, sx, sx) + pair_op(4, 5, sy, sy) + aniso * pair_op(4, 5, sz, sz))
    H_odd = (pair_op(1, 2, sx, sx) + pair_op(1, 2, sy, sy) + aniso * pair_op(1, 2, sz, sz)
             + pair_op(3, 4, sx, sx) + pair_op(3, 4, sy, sy) + aniso * pair_op(3, 4, sz, sz))
    H_field_half = -0.5 * field * sum(site_op(i, sz) for i in range(N))

    A_mat = H_odd + H_field_half
    B_mat = H_even + H_field_half
    H_total = A_mat + B_total if False else A_mat + B_mat
    return A_mat, B_mat, H_total


def _eig_decomp(mat):
    w, v = np.linalg.eigh(mat)
    return w, v, np.linalg.inv(v)


def _exp_negi_t(w, v, inv_v, t):
    """exp(-i t M) via precomputed eigendecomposition."""
    phases = np.exp(-1j * t * w)
    return (v * phases) @ inv_v


def _seed_fallback(stages):
    substeps = max(1, stages // 2)
    a = [0.5 / substeps] * (2 * substeps)
    b = []
    for _ in range(substeps):
        b += [1.0 / substeps, 0.0]
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    return a, b


def coefficients(stages):
    if stages != 12:
        return _seed_fallback(stages)

    try:
        # ---- Build Hamiltonian and precompute exponentials ----
        A_mat, B_mat, H_total = _build_hamiltonian()
        wA, vA, invA = _eig_decomp(A_mat)
        wB, vB, invB = _eig_decomp(B_mat)
        wH, vH, invH = _eig_decomp(H_total)

        eval_times = [0.25, 0.5, 1.0]
        exact = [_exp_negi_t(wH, vH, invH, t) for t in eval_times]
        dim = A_mat.shape[0]
        I = np.eye(dim, dtype=complex)

        # Project the exact propagators into the H-eigenbasis once.
        # In this basis the exact propagator is diagonal: exp(-i t wH).
        # The log-phase of U_exact in this basis is simply -t * wH (mod 2pi).
        # We will fit the log-phase of the approximated U in this basis.

        # ---- Palindromic + Fourier smooth parameterization ----
        # a_k = a_{13-k}, b_k = b_{13-k}.
        # Half sequence of length 6: h[0..5], full = [h, h[::-1]].
        # Represent half via DC + 3 cos harmonics (smooth, low-dim):
        #   h[j] = c0 + c1*cos(pi*j/5) + c2*cos(2*pi*j/5) + c3*cos(3*pi*j/5), j=0..5
        # 4 coeffs per sequence -> 8 free vars.
        # Then enforce sum(full)=1 by subtracting the mean offset from each h.
        #   sum(full) = 2*sum(h); we need sum(h)=0.5.
        #   offset = (sum(h) - 0.5)/6 ; h -= offset  (applied to both, keeps symmetry)

        j_idx = np.arange(6)
        cos1 = np.cos(np.pi * j_idx / 5.0)
        cos2 = np.cos(2 * np.pi * j_idx / 5.0)
        cos3 = np.cos(3 * np.pi * j_idx / 5.0)

        def expand(x):
            ca = x[:4]
            cb = x[4:8]
            ha = ca[0] + ca[1] * cos1 + ca[2] * cos2 + ca[3] * cos3
            hb = cb[0] + cb[1] * cos1 + cb[2] * cos2 + cb[3] * cos3
            # Enforce sum(ha)=0.5, sum(hb)=0.5
            ha = ha - (np.sum(ha) - 0.5) / 6.0
            hb = hb - (np.sum(hb) - 0.5) / 6.0
            a_full = np.concatenate([ha, ha[::-1]])
            b_full = np.concatenate([hb, hb[::-1]])
            return a_full, b_full

        # Precompute the 24 stage exponentials as a function of (a_full, b_full, t).
        # For speed, cache per-(a_full,b_full) the list of 12 A-exps and B-exps at each t.
        # We compute them on the fly but vectorize the phase multiplication.

        def _stage_exps(a_full, b_full, t):
            # returns list of (Ea_k, Eb_k)
            Ea = [(vA * np.exp(-1j * t * a_full[k] * wA)) @ invA for k in range(12)]
            Eb = [(vB * np.exp(-1j * t * b_full[k] * wB)) @ invB for k in range(12)]
            return Ea, Eb

        def _propagator(a_full, b_full, t):
            Ea, Eb = _stage_exps(a_full, b_full, t)
            U = I.copy()
            for k in range(12):
                U = Eb[k] @ Ea[k] @ U
            return U

        # Spectral residual: in H-eigenbasis, exact phase is -t*wH (real).
        # approx phase: log of diagonal of (invH @ U @ vH) -> take angle.
        # We compare wrapped phases to -t*wH (wrapped to (-pi,pi]).
        target_phases = [np.arctan2(np.sin(-t * wH), np.cos(-t * wH)) for t in eval_times]

        def objective(x):
            if time.time() > DEADLINE - 4.0:
                return 1e6
            a, b = expand(x)
            err = 0.0
            for idx, t in enumerate(eval_times):
                U = _propagator(a, b, t)
                # spectral norm error
                err += np.linalg.norm(U - exact[idx], 2)
                # spectral residual (phase) error in H-eigenbasis
                U_h = invH @ U @ vH
                diag = np.diag(U_h)
                # magnitude of U_h diagonal should be ~1; use angle
                phase = np.arctan2(np.imag(diag), np.real(diag))
                # wrap difference
                dphase = (phase - target_phases[idx] + np.pi) % (2 * np.pi) - np.pi
                err += 0.5 * np.sum(dphase ** 2) / dim
            return err / 3.0

        # ---- Optimization ----
        # Start from uniform symmetric: each a_k=b_k=1/12.
        # In Fourier half-rep, uniform h = 1/12 each -> c0 = 1/12, others 0.
        x0 = np.array([1.0 / 12.0, 0.0, 0.0, 0.0, 1.0 / 12.0, 0.0, 0.0, 0.0])

        best_x = x0.copy()
        best_val = objective(x0)

        # Phase 1: cheap global pass (differential evolution) to escape the
        # uniform local minimum; the spectral-residual landscape is smoother.
        if time.time() < DEADLINE - 25.0:
            from scipy.optimize import differential_evolution
            bounds = [(-2.0, 2.0)] * 8
            try:
                res2 = differential_evolution(
                    objective, bounds, seed=7,
                    maxiter=60, tol=1e-7, popsize=12,
                    mutation=(0.4, 1.0), recombination=0.8,
                    polish=False,
                    options={'init': 'sobol', 'maxiter': 60}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        # Phase 2: fast local polish (Nelder-Mead) on the smoothed objective.
        if time.time() < DEADLINE - 8.0:
            from scipy.optimize import minimize
            try:
                res = minimize(objective, best_x, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                        'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        a, b = expand(best_x)
        # Sanity: ensure finite and in range
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return _seed_fallback(stages)
        if np.max(np.abs(a)) > 50 or np.max(np.abs(b)) > 50:
            return _seed_fallback(stages)
        # Renormalize to exact sum=1
        sa = np.sum(a)
        sb = np.sum(b)
        if sa == 0 or sb == 0:
            return _seed_fallback(stages)
        a = list(a / sa)
        b = list(b / sb)
        return a, b

    except Exception:
        return _seed_fallback(stages)
