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

_DEADLINE = 310.0  # Safety margin before 330s kill
_START_TIME = time.time()

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
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

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

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# Precompute the 9 case superoperator bases (without drive terms)
_case_bases = []
for di_val in DE_TUNINGS:
    for sc_val in SCALES:
        base = _base_coh + _base_inc + di_val * _det_coh
        _case_bases.append((base, sc_val))

# ---- Legendre basis for parameterization ----
_t_grid = np.linspace(0, 1, N_STEPS)
_x_grid = 2 * _t_grid - 1
_N_LEG = 12
_legendre_basis = np.array([np.polynomial.legendre.legval(_x_grid, [1 if i == k else 0 for i in range(_N_LEG)]) for k in range(_N_LEG)]).T

def _coeffs_to_pulse(coeffs):
    I_coeffs = coeffs[:_N_LEG]
    Q_coeffs = coeffs[_N_LEG:]
    I = _legendre_basis @ I_coeffs
    Q = _legendre_basis @ Q_coeffs
    return I, Q

def _pulse_to_coeffs(I, Q):
    I_coeffs = np.linalg.lstsq(_legendre_basis, I, rcond=None)[0]
    Q_coeffs = np.linalg.lstsq(_legendre_basis, Q, rcond=None)[0]
    return np.concatenate([I_coeffs, Q_coeffs])

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

def _score_fast(I, Q):
    """Compute score using precomputed case bases."""
    infidelities = np.empty(N_CASES)
    k = 0
    for base, sc_val in _case_bases:
        S = np.eye(9, dtype=complex)
        for i in range(N_STEPS):
            L = base + sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
            S = expm(L * DT) @ S
        f = np.empty(6)
        for j in range(6):
            rf = S @ _test_states[j]
            fj = np.real(np.vdot(_target_states[j], rf))
            f[j] = fj if fj > 0 else 0.0
        infidelities[k] = 1.0 - np.mean(f)
        k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _score(I, Q):
    return _score_fast(I, Q)

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

def _objective(coeffs):
    I, Q = _coeffs_to_pulse(coeffs)
    score = _score_fast(I, Q)
    pen = _constraint_penalty(I, Q)
    return score + pen

def _generate_seed_pulse():
    """Generate the seed DRAG-style Gaussian pulse."""
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = 20.0
    sigma = 40.0 / 6.0  # ~6.67 ns
    
    # Gaussian envelope with area pi
    A = np.pi / (sigma * np.sqrt(2 * np.pi))
    env = A * np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset to make edges ~0
    edge_val = env[0]
    env -= edge_val * 0.5  # slight offset
    
    # DRAG beta for second-order correction
    beta = sigma**2 * A / 2.0
    denv = env * (-(t - center) / sigma**2)
    
    I = env.copy()
    Q = beta * denv
    
    return I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    
    def elapsed():
        return time.time() - _START_TIME

    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # 1. Generate initial seed pulse
    I_seed, Q_seed = _generate_seed_pulse()
    
    # Ensure validity of seed
    if not _valid(I_seed, Q_seed):
        max_amp = np.max(np.sqrt(I_seed**2 + Q_seed**2))
        if max_amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / max_amp * 0.95
            I_seed *= scale
            Q_seed *= scale
        max_slew = 0
        if np.hypot(I_seed[0], Q_seed[0]) / DT > SLEW_BOUND:
            max_slew = max(max_slew, np.hypot(I_seed[0], Q_seed[0]) / DT)
        if np.hypot(I_seed[-1], Q_seed[-1]) / DT > SLEW_BOUND:
            max_slew = max(max_slew, np.hypot(I_seed[-1], Q_seed[-1]) / DT)
        for i in range(N_STEPS - 1):
            s = np.hypot(I_seed[i+1] - I_seed[i], Q_seed[i+1] - Q_seed[i]) / DT
            if s > max_slew:
                max_slew = s
        if max_slew > SLEW_BOUND:
            scale = SLEW_BOUND / max_slew * 0.95
            I_seed *= scale
            Q_seed *= scale
    
    best_I, best_Q = I_seed.copy(), Q_seed.copy()
    best_score = _score_fast(best_I, best_Q)
    best_coeffs = _pulse_to_coeffs(best_I, best_Q)
    
    # 2. L-BFGS-B optimization in Legendre coefficient space
    if elapsed() < _DEADLINE - 60:
        try:
            bounds = [(-0.5, 0.5)] * (2 * _N_LEG)
            res = scipy.optimize.minimize(_objective, best_coeffs, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50})
            I_opt, Q_opt = _coeffs_to_pulse(res.x)
            if _valid(I_opt, Q_opt):
                s_opt = _score_fast(I_opt, Q_opt)
                if s_opt < best_score:
                    best_I, best_Q = I_opt, Q_opt
                    best_score = s_opt
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # 3. Second round with different initial guess (shifted center, different sigma)
    if elapsed() < _DEADLINE - 40:
        t = (np.arange(N_STEPS) + 0.5) * DT
        for tc in [18.0, 22.0]:
            for sig in [5.0, 7.0]:
                A = np.pi / (sig * np.sqrt(2 * np.pi))
                env = A * np.exp(-0.5 * ((t - tc) / sig)**2)
                edge_val = env[0]
                env -= edge_val * 0.5
                beta = sig**2 * A / 2.0
                denv = env * (-(t - tc) / sig**2)
                I_try = env.copy()
                Q_try = beta * denv
                
                if not _valid(I_try, Q_try):
                    max_amp = np.max(np.sqrt(I_try**2 + Q_try**2))
                    if max_amp > AMPLITUDE_BOUND:
                        scale = AMPLITUDE_BOUND / max_amp * 0.95
                        I_try *= scale
                        Q_try *= scale
                    max_slew = 0
                    if np.hypot(I_try[0], Q_try[0]) / DT > SLEW_BOUND:
                        max_slew = max(max_slew, np.hypot(I_try[0], Q_try[0]) / DT)
                    if np.hypot(I_try[-1], Q_try[-1]) / DT > SLEW_BOUND:
                        max_slew = max(max_slew, np.hypot(I_try[-1], Q_try[-1]) / DT)
                    for i in range(N_STEPS - 1):
                        s = np.hypot(I_try[i+1] - I_try[i], Q_try[i+1] - Q_try[i]) / DT
                        if s > max_slew:
                            max_slew = s
                    if max_slew > SLEW_BOUND:
                        scale = SLEW_BOUND / max_slew * 0.95
                        I_try *= scale
                        Q_try *= scale
                
                s_try = _score_fast(I_try, Q_try)
                if s_try < best_score:
                    best_I, best_Q = I_try.copy(), Q_try.copy()
                    best_score = s_try
                    best_coeffs = _pulse_to_coeffs(I_try, Q_try)
    
    # 4. Fine-tune with tighter tolerances
    if elapsed() < _DEADLINE - 20:
        try:
            bounds = [(-0.5, 0.5)] * (2 * _N_LEG)
            res2 = scipy.optimize.minimize(_objective, best_coeffs, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 80})
            I_fine, Q_fine = _coeffs_to_pulse(res2.x)
            if _valid(I_fine, Q_fine):
                s_fine = _score_fast(I_fine, Q_fine)
                if s_fine < best_score:
                    best_I, best_Q = I_fine, Q_fine
                    best_score = s_fine
        except Exception:
            pass
    
    # 5. Direct step-level refinement for last few steps (slew constraint is tight)
    if elapsed() < _DEADLINE - 5:
        I_direct = best_I.copy()
        Q_direct = best_Q.copy()
        # Try small perturbations on boundary steps
        for trial in range(50):
            I_trial = best_I.copy()
            Q_trial = best_Q.copy()
            # Perturb a random subset of steps
            idx = np.random.randint(0, N_STEPS, size=5)
            I_trial[idx] += np.random.normal(0, 0.005, size=5)
            Q_trial[idx] += np.random.normal(0, 0.005, size=5)
            if _valid(I_trial, Q_trial):
                s_trial = _score_fast(I_trial, Q_trial)
                if s_trial < best_score:
                    best_I, best_Q = I_trial, Q_trial
                    best_score = s_trial
    
    # Final validation
    if not _valid(best_I, best_Q):
        t = (np.arange(N_STEPS) + 0.5) * DT
        center = 20.0
        sigma = 4.0
        I_seed2 = 0.3 * np.exp(-0.5 * ((t - center) / sigma)**2)
        Q_seed2 = np.zeros(N_STEPS)
        best_I, best_Q = I_seed2, Q_seed2
    
    return np.column_stack([best_I, best_Q])
