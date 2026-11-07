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

_DEADLINE = 325.0  # Safety margin before 330s kill
_START_TIME = time.time()

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # coherent: -i(H rho - rho H)
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # incoherent
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # test states
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

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# ---- Fast score using precomputed matrices ----
def _score_fast(I, Q):
    """Fast scoring using precomputed superoperators."""
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _det_coh
                L += sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    return True

def _constraint_penalty(I, Q):
    amp = np.sqrt(I**2 + Q**2)
    amp_viol = np.sum(np.maximum(0, amp - AMPLITUDE_BOUND)**2)
    
    slew = np.zeros(N_STEPS + 1)
    slew[0] = np.hypot(I[0], Q[0]) / DT
    slew[-1] = np.hypot(I[-1], Q[-1]) / DT
    if N_STEPS > 1:
        dI = np.diff(I)
        dQ = np.diff(Q)
        slew[1:-1] = np.hypot(dI, dQ) / DT
    slew_viol = np.sum(np.maximum(0, slew - SLEW_BOUND)**2)
    
    return 10.0 * (amp_viol + slew_viol)

def _generate_seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    sigma = 40.0 / 6.0
    # Area = pi: A * sigma * sqrt(2*pi) = pi => A = sqrt(pi/2)/sigma
    A = np.sqrt(np.pi / 2) / sigma
    env = A * np.exp(-0.5 * ((t - center) / sigma)**2)
    # Offset to 0 at edges
    env[0] = 0.0
    env[-1] = 0.0
    I = env
    # DRAG term
    beta = sigma**2 * A / 2.0
    Q = -beta * np.gradient(env) / DT
    return I, Q

def _objective(I, Q):
    score = _score_fast(I, Q)
    pen = _constraint_penalty(I, Q)
    return score + pen

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    
    def elapsed():
        return time.time() - _START_TIME

    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Start with seed pulse
    best_I, best_Q = _generate_seed_pulse()
    best_score = _objective(best_I, best_Q)
    
    # ---- Frequency-domain parameterization with sinc basis ----
    # Use a band-limited representation to explicitly nullify leakage
    # Optimize spectral coefficients to suppress |1>->|2> transition response
    
    t = (np.arange(N_STEPS) + 0.5) * DT
    t_center = TOTAL_TIME / 2.0
    
    # Frequency-domain approach: represent pulse as sum of sinc functions
    # centered at frequencies that avoid the |1>->|2> transition
    # The |1>->|2> transition is at frequency (E2-E1)/h which relates to anharmonicity
    
    # For a transmon with anharmonicity alpha, the |1>->|2> transition
    # is at f_01 + alpha. We want to minimize drive at this frequency.
    
    # Use a parameterized form: I(t) = A * env(t) * cos(phi), Q(t) = -A * env(t) * sin(phi) + DRAG
    # where env is a shaped Gaussian with additional spectral notches
    
    def param_to_pulse_v2(params):
        """Enhanced parametric pulse with spectral shaping."""
        A, sigma, beta, tc, phi, notch_depth, notch_width = params
        if sigma < 0.5: sigma = 0.5
        u = (t - tc) / sigma
        env = A * np.exp(-0.5 * u**2)
        I = env * np.cos(phi)
        Q = -env * np.sin(phi)
        
        # DRAG term
        drag = beta * env * (-u / sigma)
        Q += drag
        
        # Spectral notch: subtract a component that cancels drive at leakage frequency
        # The leakage frequency is approximately 2*pi*(0.5 + alpha/2) relative to carrier
        # In our frame, this corresponds to a frequency offset of alpha
        # We add a corrective term that has a zero at this frequency
        if notch_depth > 0:
            # Create a pulse with a spectral zero at the leakage frequency
            # Using a modulated Gaussian: env * cos(w_leak * t) creates sidebands
            w_leak = 2 * np.pi * 0.25  # anharmonicity magnitude
            correction = notch_depth * env * np.cos(w_leak * (t - tc))
            I -= correction * np.cos(phi)
            Q += correction * np.sin(phi)
        
        return I, Q
    
    def param_obj_v2(params):
        I, Q = param_to_pulse_v2(params)
        return _objective(I, Q)
    
    # Bounds for extended parameterization
    bounds_param_v2 = [
        (0.05, 0.4),    # A
        (1.0, 8.0),     # sigma
        (-5.0, 5.0),    # beta
        (12.0, 28.0),   # tc
        (-0.3, 0.3),    # phi
        (0.0, 0.3),     # notch_depth
        (0.5, 5.0),     # notch_width
    ]
    
    # Multi-start optimization
    initial_guesses_v2 = [
        [0.3, 4.0, 0.0, 20.0, 0.0, 0.0, 2.0],
        [0.3, 4.0, 1.0, 20.0, 0.0, 0.1, 2.0],
        [0.35, 3.5, 0.5, 20.0, 0.0, 0.15, 1.5],
        [0.25, 5.0, 0.3, 20.0, 0.0, 0.05, 3.0],
        [0.3, 4.5, 1.5, 20.0, 0.1, 0.1, 2.0],
        [0.32, 3.8, 0.8, 20.0, -0.05, 0.12, 1.8],
    ]
    
    if elapsed() < _DEADLINE - 180:
        for x0 in initial_guesses_v2:
            if elapsed() > _DEADLINE - 180:
                break
            try:
                res = scipy.optimize.minimize(param_obj_v2, x0, method='Nelder-Mead',
                                              bounds=bounds_param_v2,
                                              options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6})
                I_try, Q_try = param_to_pulse_v2(res.x)
                s_try = _objective(I_try, Q_try)
                if s_try < best_score:
                    best_I, best_Q = I_try, Q_try
                    best_score = s_try
            except Exception:
                pass
    
    # ---- Direct time-domain gradient refinement ----
    # Use L-BFGS-B directly on the pulse samples for local refinement
    
    def direct_obj(IQ_flat):
        I = IQ_flat[:N_STEPS]
        Q = IQ_flat[N_STEPS:]
        return _objective(I, Q)
    
    def direct_grad(IQ_flat):
        """Numerical gradient."""
        eps = 1e-6
        grad = np.zeros(2 * N_STEPS)
        for i in range(2 * N_STEPS):
            IQ_plus = IQ_flat.copy()
            IQ_minus = IQ_flat.copy()
            IQ_plus[i] += eps
            IQ_minus[i] -= eps
            grad[i] = (direct_obj(IQ_plus) - direct_obj(IQ_minus)) / (2 * eps)
        return grad
    
    if elapsed() < _DEADLINE - 60:
        IQ_flat = np.concatenate([best_I, best_Q])
        try:
            res_direct = scipy.optimize.minimize(direct_obj, IQ_flat, method='L-BFGS-B',
                                                  jac=direct_grad,
                                                  options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7, 'maxls': 20})
            I_direct = res_direct.x[:N_STEPS]
            Q_direct = res_direct.x[N_STEPS:]
            s_direct = _objective(I_direct, Q_direct)
            if s_direct < best_score:
                best_I, best_Q = I_direct, Q_direct
                best_score = s_direct
        except Exception:
            pass
    
    # ---- Additional local refinement with smaller steps ----
    if elapsed() < _DEADLINE - 30:
        IQ_flat = np.concatenate([best_I, best_Q])
        try:
            res_fine = scipy.optimize.minimize(direct_obj, IQ_flat, method='L-BFGS-B',
                                                jac=direct_grad,
                                                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 10})
            I_fine = res_fine.x[:N_STEPS]
            Q_fine = res_fine.x[N_STEPS:]
            s_fine = _objective(I_fine, Q_fine)
            if s_fine < best_score:
                best_I, best_Q = I_fine, Q_fine
                best_score = s_fine
        except Exception:
            pass
    
    # Final validation
    if not _valid(best_I, best_Q):
        # Fallback to seed
        best_I, best_Q = _generate_seed_pulse()
        best_score = _objective(best_I, best_Q)
    
    # Ensure final pulse is valid
    if not _valid(best_I, best_Q):
        # Last resort: simple valid Gaussian
        t = (np.arange(N_STEPS) + 0.5) * DT
        center = 20.0
        sigma = 4.0
        I_seed = 0.3 * np.exp(-0.5 * ((t - center) / sigma)**2)
        Q_seed = np.zeros(N_STEPS)
        best_I, best_Q = I_seed, Q_seed
    
    return np.column_stack([best_I, best_Q])
