import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize

# ---- Constants from the task contract ----
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT
N_CASES = 9

_DEADLINE = 305.0
_START_TIME = time.time()

# Time grid - midpoint sampling
_t = (np.arange(N_STEPS) + 0.5) * DT

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)

    H_I = 0.5 * (a + a_dag)
    coh_I = coh(H_I)

    H_Q = 0.5j * (a - a_dag)
    coh_Q = coh(H_Q)

    H_det = n
    coh_det = coh(H_det)

    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))

    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]

    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)

    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_I, coh_Q, coh_det, base_inc, test, target

(_base_coh, _coh_I, _coh_Q, _coh_det, _base_inc,
 _test_states, _target_states) = _build()

def _valid(I, Q):
    amp = np.sqrt(I * I + Q * Q)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    dI = np.diff(I)
    dQ = np.diff(Q)
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if np.max(slew) > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    return True

def _score(I, Q):
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _coh_det
                L += sc_val * (I[i] * _coh_I + Q[i] * _coh_Q)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _pulse_from_params(params):
    """
    params layout (4 Gaussians per channel):
    I: A1, s1, c1, A2, s2, c2, A3, s3, c3, A4, s4, c4
    Q: B1, t1, u1, B2, t2, u2, B3, t3, u3, B4, t4, u4
    where B*Gauss(t; t_k, u_k) is the Q component.
    """
    n_g = 4
    I_params = params[:n_g * 3]
    Q_params = params[n_g * 3:]

    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)

    for k in range(n_g):
        Ak = I_params[3*k]
        sk = I_params[3*k+1]
        ck = I_params[3*k+2]
        sk = max(1.5, min(12.0, sk))
        ck = max(6.0, min(34.0, ck))
        g = np.exp(-0.5 * ((_t - ck) / sk)**2)
        I += Ak * g

    for k in range(n_g):
        Bk = Q_params[3*k]
        tk = Q_params[3*k+1]
        uk = Q_params[3*k+2]
        uk = max(1.5, min(12.0, uk))
        tk = max(6.0, min(34.0, tk))
        g = np.exp(-0.5 * ((_t - tk) / uk)**2)
        Q += Bk * g

    # Enforce amplitude bound by scaling if needed
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / max_amp
        I *= scale
        Q *= scale

    if not _valid(I, Q):
        return None
    return I, Q

def _make_drag_initial_guess():
    """
    Create a physically-informed initial guess using second-order DRAG.
    
    For a Gaussian pulse I(t) = A*exp(-t^2/(2*sigma^2)) centered at t_c:
    - First-order DRAG: Q1(t) = alpha1 * dI/dt
    - Second-order DRAG: Q2(t) = alpha2 * d^2I/dt^2
    
    The coefficients are derived from perturbation theory for the 3-level system.
    For chi = -2*pi*0.25:
    alpha1 = -chi * sigma / (2 * A)  (standard DRAG)
    alpha2 = chi^2 * sigma^3 / (8 * A^2)  (second-order correction)
    """
    A = 0.25
    sigma = 6.0
    t_c = 20.0

    # First Gaussian for I
    g1 = np.exp(-0.5 * ((_t - t_c) / sigma)**2)
    I = A * g1

    # DRAG corrections
    alpha1 = -ANHARMONICITY * sigma / (2.0 * A)
    alpha2 = (ANHARMONICITY**2) * sigma**3 / (8.0 * A**2)

    dI = -A * g1 * (_t - t_c) / sigma**2
    d2I = A * g1 * (1.0 - (_t - t_c)**2 / sigma**2) / sigma**2

    Q = alpha1 * dI + alpha2 * d2I

    # Add a second smaller Gaussian to I for robustness
    A2 = 0.10
    sigma2 = 5.0
    t_c2 = 22.0
    g2 = np.exp(-0.5 * ((_t - t_c2) / sigma2)**2)
    I += A2 * g2
    dI2 = -A2 * g2 * (_t - t_c2) / sigma2**2
    alpha1_2 = -ANHARMONICITY * sigma2 / (2.0 * A2)
    Q += alpha1_2 * dI2

    # Enforce constraints
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / max_amp
        I *= scale
        Q *= scale

    if not _valid(I, Q):
        # Fallback: simple constant pulse
        I = np.ones(N_STEPS) * 0.2
        Q = np.zeros(N_STEPS)
        ramp = 8
        for i in range(ramp):
            I[i] *= (i + 1) / ramp
            I[-1-i] *= (i + 1) / ramp
        if not _valid(I, Q):
            I = np.ones(N_STEPS) * 0.15
            Q = np.zeros(N_STEPS)

    # Convert to parameter form for 4-Gaussian basis
    # Approximate I with 2 Gaussians, Q with 2 Gaussians
    params = np.zeros(24)
    # I: 2 main Gaussians
    params[0] = A       # A1
    params[1] = sigma   # s1
    params[2] = t_c     # c1
    params[3] = A2      # A2
    params[4] = sigma2  # s2
    params[5] = t_c2    # c2
    # Q: approximate with 2 Gaussians (DRAG is roughly odd around center)
    # Q is roughly proportional to (t-t_c)*exp(...), which can be approximated
    # by difference of two Gaussians or we just fit
    # For initial guess, use the actual Q values as a rough guide
    # Place Q Gaussians to approximate the DRAG shape
    params[6] = alpha1 * A / sigma   # B1 (approx amplitude of Q)
    params[7] = sigma * 0.8          # t1
    params[8] = t_c - 2.0            # u1
    params[9] = -alpha1 * A / sigma  # B2
    params[10] = sigma * 0.8         # t2
    params[11] = t_c + 2.0           # u2

    return params

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    def objective(params):
        if elapsed() > _DEADLINE - 3:
            return 1000.0
        pulse_data = _pulse_from_params(params)
        if pulse_data is None:
            return 10.0
        I, Q = pulse_data
        return _score(I, Q)

    # Create initial guess
    init_params = _make_drag_initial_guess()
    init_score = objective(init_params)

    best_score = init_score
    best_params = init_params.copy()
    best_pulse = _pulse_from_params(init_params)

    # Differential Evolution for global search
    bounds = []
    for k in range(4):
        bounds.append((-0.35, 0.35))   # Amplitude
        bounds.append((2.0, 10.0))     # Sigma
        bounds.append((10.0, 30.0))    # Center
    for k in range(4):
        bounds.append((-0.35, 0.35))   # Amplitude
        bounds.append((2.0, 10.0))     # Sigma
        bounds.append((10.0, 30.0))    # Center

    # Run DE with time budget
    de_time_budget = 180.0  # seconds for DE
    de_start = time.time()

    def de_callback(xk, fun):
        nonlocal best_score, best_params, best_pulse
        if fun < best_score:
            best_score = fun
            best_params = xk.copy()
            pulse_data = _pulse_from_params(xk)
            if pulse_data is not None:
                best_pulse = pulse_data
        return elapsed() > _DEADLINE - 30

    try:
        # Use DE with moderate population for speed
        res = scipy.optimize.differential_evolution(
            objective, bounds,
            seed=42,
            maxiter=60,
            tol=1e-6,
            popsize=12,
            mutation=(0.3, 0.9),
            recombination=0.8,
            polish=False,
            callback=de_callback,
            update_interval=1
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
            pulse_data = _pulse_from_params(res.x)
            if pulse_data is not None:
                best_pulse = pulse_data
    except Exception:
        pass

    # Local refinement with Nelder-Mead
    if elapsed() < _DEADLINE - 20 and best_params is not None:
        try:
            res2 = scipy.optimize.minimize(
                objective, best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
                pulse_data = _pulse_from_params(res2.x)
                if pulse_data is not None:
                    best_pulse = pulse_data
        except Exception:
            pass

    # Final validity check
    if best_pulse is None:
        I = np.ones(N_STEPS) * 0.2
        Q = np.zeros(N_STEPS)
        ramp = 8
        for i in range(ramp):
            I[i] *= (i + 1) / ramp
            I[-1-i] *= (i + 1) / ramp
        if not _valid(I, Q):
            I = np.ones(N_STEPS) * 0.15
            Q = np.zeros(N_STEPS)
        best_pulse = (I, Q)

    I, Q = best_pulse
    return np.column_stack([I, Q])
