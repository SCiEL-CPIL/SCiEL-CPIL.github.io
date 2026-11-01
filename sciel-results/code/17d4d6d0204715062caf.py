"""Combined analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _precompute_constants():
    """Precompute all constant arrays needed for bound and gradient evaluation."""
    # Hermite polynomial coefficients: H_n(x) = sum_{k=0}^{floor(n/2)} h_{n,k} x^{n-2k}
    # We store p_k as coefficient of x^{2k} in P(x) = sum c_n H_n(x)
    # H0(x) = 1
    # H4(x) = x^4 - 6x^2 + 3
    # H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    # p_k = sum over n of c_n * (coefficient of x^{2k} in H_n(x))
    # H0: coeff of x^0 = 1, others 0
    # H4: coeff of x^0 = 3, x^2 = -6, x^4 = 1
    # H8: coeff of x^0 = 105, x^2 = -420, x^4 = 210, x^6 = -28, x^8 = 1
    # H12: coeff of x^0 = 945, x^2 = -7560, x^4 = 13860, x^6 = -6930, x^8 = 1155, x^10 = -66, x^12 = 1
    
    # dp_k/da_c matrix: rows are k=0..6, columns are c0, c4, c8
    # Plus contribution from da12/da_c * (dp_k/da12)
    
    # Base dp_k/da_c (without a12 contribution):
    # k=0: [1, 3, 105]
    # k=1: [0, -6, -420]
    # k=2: [0, 1, 210]
    # k=3: [0, 0, -28]
    # k=4: [0, 0, 1]
    # k=5: [0, 0, 0]
    # k=6: [0, 0, 0]
    
    base_dp = np.array([
        [1.0, 3.0, 105.0],
        [0.0, -6.0, -420.0],
        [0.0, 1.0, 210.0],
        [0.0, 0.0, -28.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    
    # dp_k/da12:
    # k=0: 945
    # k=1: -7560
    # k=2: 13860
    # k=3: -6930
    # k=4: 1155
    # k=5: -66
    # k=6: 1
    dp_da12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    # da12/da_c: a12 = -(a0 + 3a4 + 105a8)/945
    da12_da_c = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # Full dp_k/da_c = base_dp[:,c] + dp_da12 * da12_da_c[c]
    full_dp = np.zeros((7, 3))
    for c in range(3):
        full_dp[:, c] = base_dp[:, c] + dp_da12 * da12_da_c[c]
    
    # I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
    # = Gamma(n + 1/2) * 2 / (sqrt(pi) * 4^n / 2) ... let me just compute directly
    # I_n = (2n-1)!! * sqrt(pi) / 2^n for n >= 1, I_0 = sqrt(pi)
    # I_n = (2n)! / (2^{2n} n!) * sqrt(pi)
    
    sp = np.sqrt(np.pi)
    I = np.zeros(14)
    I[0] = sp
    for n in range(1, 14):
        # I_n = (2n)! / (4^n * n!) * sqrt(pi)
        # Use log for numerical stability
        log_val = 0.0
        for m in range(1, 2*n+1):
            log_val += np.log(m)
        log_val -= n * np.log(4.0)
        for m in range(1, n+1):
            log_val -= np.log(m)
        log_val += 0.5 * np.log(np.pi)
        I[n] = np.exp(log_val)
    
    # Precompute S[k][i] = sum_j p_j I[k+i+j] -- but p depends on params, so we can't precompute fully
    # Instead, precompute the structure: for each k and i, the sum over j of I[k+i+j]
    # Actually, we need: dm_k = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
    # Let T[k][i] = sum_j I[k+i+j] for j=0..6, but weighted by p_j which varies
    # So we can't fully precompute. But we can precompute I array.
    
    return I, full_dp, da12_da_c


_I, _full_dp, _da12_da_c = _precompute_constants()


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
    # p_k = coefficient of x^{2k} in P(x)
    p0 = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p1 = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p2 = a4 + 210.0*a8 + 13860.0*a12
    p3 = -28.0*a8 - 6930.0*a12
    p4 = a8 + 1155.0*a12
    p5 = -66.0*a12
    p6 = a12
    
    p = np.array([p0, p1, p2, p3, p4, p5, p6])
    
    I = _I
    
    # m_k = integral x^{2k} |P|^2 e^{-x^2} dx = sum_{i,j} p_i p_j I[k+i+j]
    # for k = 0, 1, 2 (i.e., m0, m2, m4)
    m = np.zeros(3)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    m[k] += p[i] * p[j] * I[idx]
    
    # For Fourier transform: G(xi) involves Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = np.zeros(7)
    for k in range(7):
        q[k] = p[k] * ((-1.0)**k)
    
    # |G|^2 = 2*pi * |Q|^2 * e^{-xi^2}
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 14:
                    n[k] += 2.0*np.pi * q[i] * q[j] * I[idx]
    
    m0, m2, m4 = m[0], m[1], m[2]
    n0, n2, n4 = n[0], n[1], n[2]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _full_dp[:, c]
        
        # dm_k/da_c = 2 * sum_i dp_i * (sum_j p_j I[k+i+j])
        dm = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dpi = dp[i]
                if dpi == 0.0:
                    continue
                inner = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        inner += p[j] * I[idx]
                s += dpi * inner
            dm[k] = 2.0 * s
        
        # dq_k/da_c = dp_k * (-1)^k
        dq = np.zeros(7)
        for k in range(7):
            dq[k] = dp[k] * ((-1.0)**k)
        
        # dn_k/da_c = 2 * 2*pi * sum_i dq_i * (sum_j q_j I[k+i+j])
        dn = np.zeros(3)
        for k in range(3):
            s = 0.0
            for i in range(7):
                dqi = dq[i]
                if dqi == 0.0:
                    continue
                inner = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 14:
                        inner += q[j] * I[idx]
                s += dqi * inner
            dn[k] = 4.0 * np.pi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([0.0, 0.0, 0.0])
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns three coefficients (c0, c4, c8) of the Hermite combination.
    The degree-12 weight is fixed so g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180  # Safety margin
    
    # Initial guess from prior work
    c0_init, c4_init, c8_init = -113/100, 1/25, 1/3240
    x0 = np.array([c0_init, c4_init, c8_init])
    
    # Multiple starting points
    starts = [
        x0.copy(),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.06, 0.0006]),
    ]
    
    rng = np.random.default_rng(42)
    for i in range(30):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B with analytic gradient from multiple starts
    for start in starts:
        if time.time() > deadline - 40:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with Nelder-Mead from best
    if best_params is not None and time.time() < deadline - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: More random restarts with L-BFGS-B if time permits
    if time.time() < deadline - 30:
        for i in range(50):
            if time.time() > deadline - 15:
                break
            x_rand = rng.normal(0, 1.0, 3)
            try:
                res = minimize(_objective_grad, x_rand, method='L-BFGS-B',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
        
        # Final polish
        if best_params is not None and time.time() < deadline - 5:
            try:
                res = minimize(_objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 200000, 'xatol': 1e-18, 'fatol': 1e-20})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    if best_params is not None:
        return best_params
    return x0

# EVOLVE-BLOCK-END
