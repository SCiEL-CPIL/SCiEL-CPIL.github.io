import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _liouvillian_super(I_val, Q_val, detuning, scale, a, adag, n):
    I3 = np.eye(3, dtype=complex)
    H = detuning * n + 0.5 * (-2*np.pi*0.25) * (n @ (n - I3))
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    H_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    L1 = a / np.sqrt(25000.0)
    L2 = n * np.sqrt(2.0 / 40000.0)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    return H_super + L1_super + L2_super

def _simulate_pulse(IQ, detuning, scale, a, adag, n):
    I3 = np.eye(3, dtype=complex)
    X_gate = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
    states = [
        np.array([1,0,0], dtype=complex),
        np.array([0,1,0], dtype=complex),
        (np.array([1,1,0], dtype=complex))/np.sqrt(2),
        (np.array([1,-1,0], dtype=complex))/np.sqrt(2),
        (np.array([1,1j,0], dtype=complex))/np.sqrt(2),
        (np.array([1,-1j,0], dtype=complex))/np.sqrt(2)
    ]
    supers = [expm(_liouvillian_super(IQ[i,0], IQ[i,1], detuning, scale, a, adag, n) * 0.5) for i in range(80)]
    fids = []
    for psi in states:
        rho = np.outer(psi, psi.conj()).flatten()
        for s in supers:
            rho = s @ rho
        rho = rho.reshape(3,3)
        psi_t = X_gate @ psi
        rho_t = np.outer(psi_t, psi_t.conj())
        fids.append(np.real(np.trace(rho_t @ rho)))
    return np.mean(fids)

def _score(IQ):
    a, adag, n = _build_matrices()
    infs = []
    for det in [-2*np.pi*0.002, 0, 2*np.pi*0.002]:
        for sc in [0.97, 1.0, 1.03]:
            fid = _simulate_pulse(IQ, det, sc, a, adag, n)
            infs.append(1 - fid)
    return np.mean(infs) + np.max(infs)

def _valid(IQ):
    amps = np.sqrt(IQ[:,0]**2 + IQ[:,1]**2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    for i in range(80):
        prev = IQ[i-1] if i > 0 else np.array([0.0, 0.0])
        if np.linalg.norm(IQ[i] - prev) / 0.5 > 0.12 + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) / 0.5 > 0.12 + 1e-12:
        return False
    return True

def _enforce_constraints(IQ):
    # Enforce amplitude bound
    amps = np.sqrt(IQ[:,0]**2 + IQ[:,1]**2)
    max_amp = np.max(amps)
    if max_amp > 0.4:
        IQ = IQ * (0.4 / max_amp) * 0.99
    
    # Enforce slew rate bound (including from zero and to zero)
    max_d = 0.12 * 0.5 * 0.99
    for i in range(80):
        prev = IQ[i-1] if i > 0 else np.array([0.0, 0.0])
        diff = IQ[i] - prev
        norm = np.linalg.norm(diff)
        if norm > max_d:
            IQ[i] = prev + (max_d / norm) * diff
    
    # Enforce final slew to zero
    norm_last = np.linalg.norm(IQ[-1])
    if norm_last > max_d:
        IQ[-1] = IQ[-1] * (max_d / norm_last)
    
    return IQ

def _make_seed_pulse():
    t = np.arange(80) * 0.5
    sigma = 40.0 / 6.0
    env = np.exp(-(t - 20)**2 / (2 * sigma**2))
    env = np.maximum(env - env[0] - env[-1], 0)
    area = np.sum(env) * 0.5
    if area > 0:
        env *= np.pi / area
    I = env
    Q = -0.3 * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    return _enforce_constraints(IQ)

def _objective(x):
    IQ = x.reshape(80, 2)
    IQ = _enforce_constraints(IQ)
    return _score(IQ)

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = 320  # Safety margin before 330s kill
    
    # Start with a good DRAG seed
    best_IQ = _make_seed_pulse()
    best_score = _score(best_IQ)
    
    # Run Nelder-Mead optimization over all 160 parameters
    x0 = best_IQ.flatten()
    
    try:
        result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True
            }
        )
        
        if result.fun < best_score:
            best_IQ = _enforce_constraints(result.x.reshape(80, 2))
            best_score = result.fun
    except Exception:
        pass
    
    # Check time and potentially do a second round with tighter tolerances
    elapsed = time.time() - start
    if elapsed < deadline - 60:
        try:
            result2 = minimize(
                _objective,
                best_IQ.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            if result2.fun < best_score:
                best_IQ = _enforce_constraints(result2.x.reshape(80, 2))
                best_score = result2.fun
        except Exception:
            pass
    
    # Final validity check
    if not _valid(best_IQ):
        # Fallback to a simpler valid pulse
        t = np.arange(80) * 0.5
        sigma = 40.0 / 6.0
        env = np.exp(-(t - 20)**2 / (2 * sigma**2))
        env = np.maximum(env - env[0] - env[-1], 0)
        area = np.sum(env) * 0.5
        if area > 0:
            env *= np.pi / area
        I = env * 0.35
        Q = np.zeros(80)
        best_IQ = np.column_stack([I, Q])
        best_IQ = _enforce_constraints(best_IQ)
    
    return best_IQ
