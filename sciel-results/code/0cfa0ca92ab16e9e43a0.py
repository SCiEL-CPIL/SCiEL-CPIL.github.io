"""Program for the second autocorrelation inequality task.

We parameterize the nonnegative heights as a superposition of shifted
top-hat basis functions.  The autoconvolution supremum is computed
analytically from the pairwise overlap integrals of the basis functions,
and we optimize the continuous amplitudes and positions with gradient
descent on the analytical ratio.
"""

import numpy as np

N = 50

# Grid on [-1/4, 1/4]
A = -0.25
B = 0.25
GRID = np.linspace(A, B, N)
DX = (B - A) / (N - 1)


def _autoconv_supremum(heights):
    """Analytical supremum of the autoconvolution of a piecewise-constant
    (step) signal defined by heights on the uniform grid.

    The signal is constant on each sub-interval [x_i, x_{i+1}).  The
    autoconvolution is piecewise quadratic and its maximum occurs at a
    breakpoint.  We evaluate it exactly at all relevant breakpoints.
    """
    h = np.asarray(heights, dtype=float)
    n = len(h)
    xs = GRID

    # Interval boundaries: n+1 of them (extend by half a cell at each end).
    # The signal is h[i] on [xs[i], xs[i+1]) for i in 0..n-2, and we extend
    # with h[0] before xs[0] and h[-1] after xs[-1] by half a cell so the
    # convolution support is captured.  Simpler: build the piecewise signal
    # on its own support [xs[0], xs[-1]] and convolve.
    #
    # We build the step signal on support [xs[0], xs[-1]] (length L = (n-1)*dx).
    # The autoconvolution support is [2*xs[0], 2*xs[-1]].
    #
    # Represent the signal by its "staircase" of constant pieces.  For a
    # piecewise-constant f with values v_k on intervals [t_k, t_{k+1}),
    # the autoconvolution at a point can be computed by summing over pairs of
    # intervals.  The maximum is attained at a breakpoint (a t_k or a
    # difference of breakpoints).  We evaluate exactly at all breakpoints of
    # the form (t_j - t_k) + 2*xs[0] ... this is O(n^2) which is fine.

    t = xs  # n breakpoints of the signal pieces; piece i is on [t[i], t[i+1]]
    # We treat the signal as: value v_i on [t[i], t[i+1]] for i = 0..n-2.
    # Extend the endpoints: define t_ext with n+1 points and v_ext with n values.
    t_ext = xs
    v = h  # length n, but the last value only matters if we use n pieces.

    # Use n-1 pieces to keep it clean: piece i on [xs[i], xs[i+1]], i=0..n-2.
    m = n - 1
    tb = xs  # m+1 = n boundaries
    vb = h[:m]  # m values

    # Autoconvolution (correlation) g(s) = \int f(x) f(x+s) dx.
    # For piecewise constant, g is piecewise quadratic; max at a breakpoint
    # s = tb[j] - tb[k].
    # Evaluate g at all such s by summing over pairs of pieces.
    # g(s) = sum_{i,k} vb[i]*vb[k] * length( [tb[i],tb[i+1]] ∩ [tb[k]-s, tb[k+1]-s] )
    # where the intersection length of two intervals is max(0, min(e1,e2)-max(b1,b2)).

    # Candidate s values.
    s_vals = np.array([tb[j] - tb[k] for j in range(m + 1) for k in range(m + 1)])
    # Also include s = 0 explicitly (already there).
    best = 0.0
    for s in s_vals:
        total = 0.0
        for i in range(m):
            bi, ei = tb[i], tb[i + 1]
            vi = vb[i]
            if vi == 0.0:
                continue
            for k in range(m):
                vk = vb[k]
                if vk == 0.0:
                    continue
                # interval for f(x+s): x in [tb[k]-s, tb[k+1]-s]
                bk = tb[k] - s
                ek = tb[k + 1] - s
                lo = max(bi, bk)
                hi = min(ei, ek)
                if hi > lo:
                    total += vi * vk * (hi - lo)
        if total > best:
            best = total
    return best


def _ratio_and_grad(heights):
    """Return (ratio, grad) for the ratio R = ||h||_2^2 / (||h||_1 * sup|autoconv|).

    Note: heights are nonnegative so ||h||_1 = sum(h), and the autoconvolution
    is nonnegative (it's a correlation of a nonnegative function), so the
    supremum of the autoconvolution equals the supremum of its absolute value.
    """
    h = np.asarray(heights, dtype=float)
    h = np.maximum(h, 0.0)
    l2sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 <= 1e-15:
        return 0.0, np.zeros_like(h)
    supc = _autoconv_supremum(h)
    if supc <= 1e-15:
        return 0.0, np.zeros_like(h)
    R = l2sq / (l1 * supc)
    # Gradient of R w.r.t. h (ignoring nonnegativity clipping; we project after).
    # R = l2sq / (l1 * supc)
    # dR = (dl2sq * l1*supc - l2sq * d(l1*supc)) / (l1*supc)^2
    # dl2sq = 2 h
    # dl1 = 1 (vector of ones)
    # dsupc: supremum is a max over linear-ish (piecewise) functions; subgradient.
    # For a smooth approximation we treat supc as differentiable via the active
    # breakpoint.  We compute a subgradient: find the active s*, then
    # d supc = sum_{i,k} (vi*vk) * d/ dh of overlap length at s*.
    # This is involved; we use a finite-difference-free analytic subgradient.
    #
    # Simpler and robust: use a small numerical gradient for the supc term
    # only (cheap, N=50) and analytic for l2sq, l1.
    # Actually we do a full analytic subgradient for supc at the active s*.

    # Find active s* (the s that achieves the supremum).
    m = N - 1
    tb = GRID
    vb = h[:m]
    s_vals = np.array([tb[j] - tb[k] for j in range(m + 1) for k in range(m + 1)])
    # Compute g(s) for all s to find the argmax (reuse the loop but vectorized-ish).
    # We'll recompute g for all s and track the max and its s*.
    g_all = np.zeros(len(s_vals))
    for idx, s in enumerate(s_vals):
        total = 0.0
        for i in range(m):
            bi, ei = tb[i], tb[i + 1]
            vi = vb[i]
            if vi == 0.0:
                continue
            for k in range(m):
                vk = vb[k]
                if vk == 0.0:
                    continue
                bk = tb[k] - s
                ek = tb[k + 1] - s
                lo = max(bi, bk)
                hi = min(ei, ek)
                if hi > lo:
                    total += vi * vk * (hi - lo)
        g_all[idx] = total
    imax = int(np.argmax(g_all))
    s_star = s_vals[imax]

    # Subgradient of supc w.r.t. h: for the active s*, supc = sum_{i,k} h[i]*h[k]*L_{ik}(s*)
    # where L_{ik} is the overlap length (independent of h).  So
    # d supc / d h[i] = sum_k h[k] * L_{ik} + sum_k h[k] * L_{ki}  (symmetric)
    # = sum_k h[k]*(L_{ik} + L_{ki}) ... but L is symmetric in (i,k) for fixed s?
    # overlap([i],[k] shifted by s) vs overlap([k],[i] shifted by s): not equal in general.
    # g(s) = sum_i sum_k h[i] h[k] L_{ik}(s) where L_{ik}(s) = len(piece_i ∩ (piece_k - s)).
    # d g / d h[i] = sum_k h[k] L_{ik} + sum_k h[k] L_{ki}.
    grad_supc = np.zeros(N)
    for i in range(m):
        bi, ei = tb[i], tb[i + 1]
        for k in range(m):
            bk = tb[k] - s_star
            ek = tb[k + 1] - s_star
            lo = max(bi, bk)
            hi = min(ei, ek)
            Lik = (hi - lo) if hi > lo else 0.0
            # symmetric term
            lo2 = max(tb[k], tb[i] - s_star)
            hi2 = min(tb[k + 1], tb[i + 1] - s_star)
            Lki = (hi2 - lo2) if hi2 > lo2 else 0.0
            grad_supc[i] += h[k] * (Lik + Lki)
    # Note: h has length N but only first m pieces are used; last height h[m]
    # is unused in the piecewise model (we used m = N-1 pieces).  To keep the
    # model consistent, we actually should use all N heights.  Let's redefine:
    # use N pieces by extending the grid by one half-cell at the right.
    # For simplicity and correctness we recompute with N pieces below.

    # Redo with N pieces: boundaries tb2 of length N+1.
    tb2 = np.concatenate([GRID, [GRID[-1] + DX]])
    vb2 = h  # N values, piece i on [tb2[i], tb2[i+1]]
    m2 = N
    # recompute g at s* with N pieces for the subgradient
    grad_supc2 = np.zeros(N)
    for i in range(m2):
        bi, ei = tb2[i], tb2[i + 1]
        for k in range(m2):
            bk = tb2[k] - s_star
            ek = tb2[k + 1] - s_star
            lo = max(bi, bk)
            hi = min(ei, ek)
            Lik = (hi - lo) if hi > lo else 0.0
            lo2 = max(tb2[k], tb2[i] - s_star)
            hi2 = min(tb2[k + 1], tb2[i + 1] - s_star)
            Lki = (hi2 - lo2) if hi2 > lo2 else 0.0
            grad_supc2[i] += h[k] * (Lik + Lki)

    # Combine gradient of R.
    # R = l2sq / (l1 * supc)
    # dR/dh_i = (2 h_i * l1*supc - l2sq * ( (1*supc) + l1 * d supc/dh_i )) / (l1*supc)^2
    denom = (l1 * supc) ** 2
    dl1 = 1.0
    grad = np.zeros(N)
    for i in range(N):
        dnum = 2.0 * h[i] * (l1 * supc) - l2sq * (dl1 * supc + l1 * grad_supc2[i])
        grad[i] = dnum / denom
    return R, grad


def _autoconv_supremum_N(heights):
    """Supremum of autoconvolution using N pieces (grid extended by half cell)."""
    h = np.asarray(heights, dtype=float)
    h = np.maximum(h, 0.0)
    tb2 = np.concatenate([GRID, [GRID[-1] + DX]])
    m2 = N
    s_vals = np.array([tb2[j] - tb2[k] for j in range(m2 + 1) for k in range(m2 + 1)])
    best = 0.0
    for s in s_vals:
        total = 0.0
        for i in range(m2):
            bi, ei = tb2[i], tb2[i + 1]
            vi = h[i]
            if vi == 0.0:
                continue
            for k in range(m2):
                vk = h[k]
                if vk == 0.0:
                    continue
                bk = tb2[k] - s
                ek = tb2[k + 1] - s
                lo = max(bi, bk)
                hi = min(ei, ek)
                if hi > lo:
                    total += vi * vk * (hi - lo)
        if total > best:
            best = total
    return best


def _ratio(heights):
    h = np.maximum(np.asarray(heights, dtype=float), 0.0)
    l2sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 <= 1e-15:
        return 0.0
    supc = _autoconv_supremum_N(h)
    if supc <= 1e-15:
        return 0.0
    return l2sq / (l1 * supc)


def _optimize():
    """Optimize the nonnegative heights to maximize the ratio via gradient ascent."""
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_R = _ratio(best_h)

    # Multiple random restarts with different initializations.
    for restart in range(4):
        if restart == 0:
            h = np.ones(N)
        elif restart == 1:
            h = np.ones(N)
            # sparse: a few large bumps
            h[:] = 0.0
            for _ in range(6):
                idx = rng.integers(0, N)
                w = rng.integers(1, 5)
                for j in range(idx, min(N, idx + w)):
                    h[j] = rng.uniform(0.5, 2.0)
        elif restart == 2:
            # smooth random
            h = np.abs(np.sin(np.linspace(0, 3 * np.pi, N)) + 0.3 * rng.standard_normal(N))
            h = np.maximum(h, 0.0)
        else:
            # peaky: one wide plateau
            h = np.zeros(N)
            c = rng.integers(5, N - 5)
            w = rng.integers(8, 20)
            for j in range(max(0, c - w // 2), min(N, c + w // 2)):
                h[j] = 1.0

        # Gradient ascent with adaptive step and projection to nonnegativity.
        lr = 0.05
        for it in range(300):
            R, grad = _ratio_and_grad(h)
            if not np.isfinite(R):
                break
            # normalize gradient scale
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            step = lr * h / gnorm  # scale-aware
            h_new = h + step
            h_new = np.maximum(h_new, 0.0)
            R_new = _ratio(h_new)
            if R_new > R + 1e-12:
                h = h_new
                if R_new > best_R:
                    best_R = R_new
                    best_h = h.copy()
                lr = min(lr * 1.05, 0.5)
            else:
                lr *= 0.7
                if lr < 1e-5:
                    break
    return best_h, best_R


def construct():
    """Return a sequence of 50 nonnegative step heights."""
    h, R = _optimize()
    # Safety: ensure finite, nonnegative.
    h = np.where(np.isfinite(h), h, 0.0)
    h = np.maximum(h, 0.0)
    # Normalize scale (the ratio is scale-invariant, but keep reasonable magnitudes).
    s = float(np.sum(h))
    if s > 0:
        h = h * (N / s)  # make sum = N, similar to seed
    return [float(x) for x in h]
