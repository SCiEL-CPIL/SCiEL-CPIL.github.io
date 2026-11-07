import numpy as np
import time
from scipy.linalg import expm

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _precompute_super_terms(a, adag, n):
    I3 = np.eye(3, dtype=complex)
    anharmonicity = -2 * np.pi * 0.25
    H0 = anharmonicity * 0.5 * (n @ (n - I3))
    H0_super = -1j * (np.kron(H0, I3) - np.kron(I3, H0.T))
    
    drive_I = 0.5 * (a + adag)
    drive_Q = 0.5j * (a - adag)
    drive_I_super = -1j * (np.kron(drive_I, I3) - np.kron(I3, drive_I.T))
    drive_Q_super = -1j * (np.kron(drive_Q, I3) - np.kron(I3, drive_Q.T))
    
    det_super = -1j * (np.kron(n, I3) - np.kron(I3, n.T))
    
    L1 = a / np.sqrt(25000.0)
    L2 = n * np.sqrt(2.0 / 40000.0)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    diss_super = L1_super + L2_super
    
    return H0_super, drive_I_super, drive_Q_super, det_super, diss_super

def _legendre_basis(t, n_terms):
    x = 2.0 * t / 40.0 - 1.0
    basis = np.zeros((len(t), n_terms), dtype=float)
    if n_terms > 0:
        basis[:, 0] = 1.0
    if n_terms > 1:
        basis[:, 1] = x
    for k in range(2, n_terms):
        basis[:, k] = ((2*k - 1) * x * basis[:, k-1] - (k - 1) * basis[:, k-2]) / k
    return basis

def _pulse_from_legendre(coeffs):
    t = np.arange(80) * 0.5
    n_I = 8
    n_Q = 8
    B_I = _legendre_basis(t, n_I)
    B_Q = _legendre_basis(t, n_Q)
    
    I = B_I @ coeffs[:n_I]
    Q = B_Q @ coeffs[n_I:n_I+n_Q]
    
    I[0] = 0.0
    I[-1] = 0.0
    Q[0] = 0.0
    Q[-1] = 0.0
    
    IQ = np.column_stack([I, Q])
    return IQ

def _simulate_pulse_fast(IQ, detuning, scale, super_terms):
    H0_super, drive_I_super, drive_Q_super, det_super, diss_super = super_terms
    dt = 0.5
    
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [1/np.sqrt(2), 1j/np.sqrt(2), 0],
        [1/np.sqrt(2), -1j/np.sqrt(2), 0]
    ], dtype=complex)
    
    supers = []
    for i in range(80):
        I_val = IQ[i, 0]
        Q_val = IQ[i, 1]
        L = H0_super + detuning * det_super + scale * (I_val * drive_I_super + Q_val * drive_Q_super) + diss_super
        supers.append(expm(L * dt))
    
    fids = []
    for k in range(6):
        psi = states[k]
        rho = np.outer(psi, psi.conj()).flatten()
        for s in supers:
            rho = s @ rho
        rho = rho.reshape(3, 3)
        psi_t = X_gate @ psi
        rho_t = np.outer(psi_t, psi_t.conj())
        fids.append(np.real(np.trace(rho_t @ rho)))
    return np.mean(fids)

def _score_fast(IQ, super_terms):
    infs = []
    for det in [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]:
        for sc in [0.97, 1.0, 1.03]:
            fid = _simulate_pulse_fast(IQ, det, sc, super_terms)
            infs.append(1 - fid)
    return np.mean(infs) + np.max(infs)

def _valid(IQ):
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        if np.linalg.norm(IQ[i] - prev) / 0.5 > 0.12 + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) / 0.5 > 0.12 + 1e-12:
        return False
    return True

def _enforce_constraints(IQ):
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4):
        scale = 0.4 / np.max(amps) * 0.999
        IQ = IQ * scale
    
    max_d = 0.12 * 0.5 * 0.999
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        diff = np.linalg.norm(IQ[i] - prev)
        if diff > max_d:
            if diff > 1e-15:
                IQ[i] = prev + (max_d / diff) * (IQ[i] - prev)
            else:
                IQ[i] = prev.copy()
    
    if np.linalg.norm(IQ[-1]) > max_d:
        if np.linalg.norm(IQ[-1]) > 1e-15:
            IQ[-1] *= max_d / np.linalg.norm(IQ[-1])
        else:
            IQ[-1] = np.array([0.0, 0.0])
    
    return IQ

def _make_seed_pulse():
    t = np.arange(80) * 0.5
    sigma = 40.0 / 6.0
    env = np.exp(-(t - 20) ** 2 / (2 * sigma ** 2))
    env = np.maximum(env - env[0] - env[-1], 0.0)
    area = np.sum(env) * 0.5
    if area > 1e-10:
        env *= np.pi / area
    I = env
    Q = -0.3 * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    return _enforce_constraints(IQ)

def _fit_legendre(IQ):
    t = np.arange(80) * 0.5
    n_I = 8
    n_Q = 8
    B_I = _legendre_basis(t, n_I)
    B_Q = _legendre_basis(t, n_Q)
    
    I = IQ[:, 0]
    Q = IQ[:, 1]
    coeffs_I, _, _, _ = np.linalg.lstsq(B_I, I, rcond=None)
    coeffs_Q, _, _, _ = np.linalg.lstsq(B_Q, Q, rcond=None)
    return np.concatenate([coeffs_I, coeffs_Q])

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = 310.0
    
    a, adag, n = _build_matrices()
    super_terms = _precompute_super_terms(a, adag, n)
    
    seed_IQ = _make_seed_pulse()
    best_IQ = seed_IQ.copy()
    best_score = _score_fast(best_IQ, super_terms)
    
    coeffs0 = _fit_legendre(seed_IQ)
    current_coeffs = coeffs0.copy()
    current_IQ = _pulse_from_legendre(current_coeffs)
    current_IQ = _enforce_constraints(current_IQ)
    if not _valid(current_IQ):
        current_IQ = seed_IQ.copy()
        current_coeffs = _fit_legendre(current_IQ)
    
    # Simulated Annealing
    T = 1e-4
    T_min = 1e-7
    alpha = 0.995
    rng = np.random.default_rng(42)
    
    steps = 0
    while time.time() < deadline - 10:
        if T < T_min:
            break
        
        # Propose new coefficients
        sigma = 0.05 * (T / T_min + 0.1)
        new_coeffs = current_coeffs + rng.normal(0, sigma, 16)
        
        new_IQ = _pulse_from_legendre(new_coeffs)
        new_IQ = _enforce_constraints(new_IQ)
        
        if not _valid(new_IQ):
            T *= alpha
            continue
        
        new_score = _score_fast(new_IQ, super_terms)
        
        # Acceptance criterion
        if new_score < current_score:
            current_coeffs = new_coeffs
            current_IQ = new_IQ
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_IQ = new_IQ.copy()
        else:
            delta_E = new_score - current_score
            if rng.random() < np.exp(-delta_E / T):
                current_coeffs = new_coeffs
                current_IQ = new_IQ
                current_score = new_score
        
        T *= alpha
        steps += 1
    
    # Final refinement with L-BFGS-B from the best point found
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        
        def obj_cb(c):
            if time.time() > deadline - 5:
                return 1e6
            IQ = _pulse_from_legendre(c)
            IQ = _enforce_constraints(IQ)
            if not _valid(IQ):
                return 1e6
            return _score_fast(IQ, super_terms)
        
        try:
            res = minimize(obj_cb, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-9})
            if res.fun < best_score:
                best_score = res.fun
                best_IQ = _pulse_from_legendre(res.x)
                best_IQ = _enforce_constraints(best_IQ)
                if not _valid(best_IQ):
                    best_IQ = best_IQ # keep old best if invalid
        except Exception:
            pass
    
    if not _valid(best_IQ):
        best_IQ = _enforce_constraints(best_IQ)
    
    return best_IQ
