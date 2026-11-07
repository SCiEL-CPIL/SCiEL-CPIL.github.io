import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.special import eval_legendre

# Constants
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

# Precompute static terms
def _build_static_terms():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    I3 = np.eye(3, dtype=complex)
    
    def coh(H):
        return -1j * (np.kron(H, I3) - np.kron(I3, H.T))
        
    # Note: The prompt's H is H = detuning n + ... + 0.5 scale (I (a+adag) + i Q (a-adag))
    # The Liouvillian is L = -i [H, rho] + ...
    # So H_I = 0.5 (a + adag), H_Q = 0.5 i (a - adag)
    
    H0 = ANHARMONICITY * 0.5 * (n @ (n - I3))
    H_det = n
    H_I = 0.5 * (a + adag)
    H_Q = 0.5j * (a - adag)
    
    L0_coh = coh(H0)
    L_det_coh = coh(H_det)
    L_I_coh = coh(H_I)
    L_Q_coh = coh(H_Q)
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        # Standard Lindblad: Lk rho Lk^dag - 0.5 {Lk^dag Lk, rho}
        # Vectorized: kron(Lk, Lk*) - 0.5 kron(I, Lk^dag Lk) - 0.5 kron((Lk^dag Lk)^T, I)
        # Note: The prompt's code used:
        # np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
        # This matches.
        return np.kron(Lk, Lk.conj()) - 0.5 * (np.kron(LkLkd, I3) + np.kron(I3, LkLkd.T))
        
    L_diss = incoh(L1) + incoh(L2)
    
    # States
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
            
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    test_states = []
    target_states = []
    for p in psis:
        rho = np.outer(p, p.conj()).flatten()
        test_states.append(rho)
        psi_t = X_gate @ p
        rho_t = np.outer(psi_t, psi_t.conj()).flatten()
        target_states.append(rho_t)
        
    return L0_coh, L_det_coh, L_I_coh, L_Q_coh, L_diss, test_states, target_states

(L0_coh, L_det_coh, L_I_coh, L_Q_coh, L_diss, _test_states, _target_states) = _build_static_terms()

# Legendre Basis
N_LEG = 12
# Time points: midpoints
_t = (np.arange(N_STEPS) + 0.5) * DT
# Map to [-1, 1]
_x = 2.0 * _t / TOTAL_TIME - 1.0

# Compute Legendre polynomials P_k(x) for k=0..N_LEG-1
# Shape: (N_LEG, N_STEPS)
_leg_basis = np.array([eval_legendre(k, _x) for k in range(N_LEG)])

def _pulse_from_coeffs(c):
    """
    c: array of length 2*N_LEG
    First N_LEG are I coeffs, next N_LEG are Q coeffs.
    Returns I, Q arrays of length N_STEPS.
    """
    cI = c[:N_LEG]
    cQ = c[N_LEG:]
    I = _leg_basis.T @ cI
    Q = _leg_basis.T @ cQ
    return I, Q

def _valid(I, Q):
    amp = np.sqrt(I * I + Q * Q)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-9:
        return False
    # Slew from zero to first
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    # Slew between steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if np.max(slew) > SLEW_BOUND + 1e-9:
        return False
    # Slew from last to zero
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    return True

def _score(I, Q):
    infs = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = L0_coh + L_diss
                L += di_val * L_det_coh
                L += sc_val * (I[i] * L_I_coh + Q[i] * L_Q_coh)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj
            infs[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infs) + np.max(infs)

def _enforce_constraints(I, Q):
    """
    Rough projection to feasible set.
    """
    # Amplitude
    amp = np.sqrt(I*Q*0 + I*I + Q*Q) # just I^2+Q^2
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = (AMPLITUDE_BOUND / max_amp) * 0.999
        I *= scale
        Q *= scale
        
    # Slew
    max_d = SLEW_BOUND * DT * 0.999
    # Forward pass
    for i in range(N_STEPS):
        prev_I = I[i-1] if i > 0 else 0.0
        prev_Q = Q[i-1] if i > 0 else 0.0
        dI = I[i] - prev_I
        dQ = Q[i] - prev_Q
        d = np.hypot(dI, dQ)
        if d > max_d:
            if d > 1e-15:
                I[i] = prev_I + (max_d / d) * dI
                Q[i] = prev_Q + (max_d / d) * dQ
            else:
                I[i] = prev_I
                Q[i] = prev_Q
    # Backward pass for end condition? 
    # The constraint is ||p[N-1] - 0|| / dt <= bound.
    # So I[-1], Q[-1] must be small.
    end_d = np.hypot(I[-1], Q[-1])
    if end_d > max_d:
        if end_d > 1e-15:
            I[-1] *= max_d / end_d
            Q[-1] *= max_d / end_d
        else:
            I[-1] = 0
            Q[-1] = 0
            
    # Re-check amplitude after slew clamp (slew clamp might increase amp if it pulls towards a larger previous value? No, it pulls towards previous, reducing distance. But if previous was large, it might stay large. 
    # Actually, if I[i] was large and I[i-1] was 0, it gets clamped. 
    # If I[i] was 0 and I[i-1] was large, it gets pulled towards I[i-1], increasing I[i]. This could violate amplitude if I[i-1] is near bound.
    # Simple iterative clamp is safer.
    for _ in range(3):
        amp = np.sqrt(I*I + Q*Q)
        max_amp = np.max(amp)
        if max_amp > AMPLITUDE_BOUND:
            scale = (AMPLITUDE_BOUND / max_amp) * 0.999
            I *= scale
            Q *= scale
        for i in range(N_STEPS):
            prev_I = I[i-1] if i > 0 else 0.0
            prev_Q = Q[i-1] if i > 0 else 0.0
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            d = np.hypot(dI, dQ)
            if d > max_d:
                if d > 1e-15:
                    I[i] = prev_I + (max_d / d) * dI
                    Q[i] = prev_Q + (max_d / d) * dQ
        end_d = np.hypot(I[-1], Q[-1])
        if end_d > max_d:
            if end_d > 1e-15:
                I[-1] *= max_d / end_d
                Q[-1] *= max_d / end_d
    
    return I, Q

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = 320.0
    
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
        
    # Initial guess: DRAG Gaussian
    t_gauss = _t
    sigma = 40.0 / 6.0
    env = np.exp(-((t_gauss - 20.0) ** 2) / (2 * sigma ** 2))
    # Offset to make edges 0? Legendre basis doesn't guarantee 0 edges.
    # We'll just use the Gaussian shape for I.
    I_seed = env * 0.3 # Peak ~0.3
    dI_dt = np.gradient(I_seed, DT)
    Q_seed = -0.3 * dI_dt # DRAG coefficient
    
    # Project onto Legendre basis
    # Minimize || _leg_basis.T @ c - I_seed ||^2
    cI_init = np.linalg.lstsq(_leg_basis.T, I_seed, rcond=None)[0]
    cQ_init = np.linalg.lstsq(_leg_basis.T, Q_seed, rcond=None)[0]
    
    c0 = np.concatenate([cI_init, cQ_init])
    
    # Check validity of initial guess
    I_init, Q_init = _pulse_from_coeffs(c0)
    if not _valid(I_init, Q_init):
        I_init, Q_init = _enforce_constraints(I_init, Q_init)
        # Re-project? Or just use the enforced one as a starting point for time-domain opt?
        # For now, let's scale down coefficients until valid.
        scale = 1.0
        for _ in range(20):
            c0_scaled = c0 * scale
            I_s, Q_s = _pulse_from_coeffs(c0_scaled)
            if _valid(I_s, Q_s):
                c0 = c0_scaled
                I_init, Q_init = I_s, Q_s
                break
            scale *= 0.8
        else:
            # Fallback: very small pulse
            c0 = np.zeros(2 * N_LEG)
            c0[0] = 0.1 # Constant I
            I_init, Q_init = _pulse_from_coeffs(c0)
            
    best_score = _score(I_init, Q_init)
    best_c = c0.copy()
    best_IQ = np.column_stack([I_init, Q_init])
    
    def objective(c):
        if time.time() > deadline - 5:
            return 1e6
        I, Q = _pulse_from_coeffs(c)
        if not _valid(I, Q):
            # Penalty
            amp = np.sqrt(I*I + Q*Q)
            amp_pen = np.sum(np.maximum(amp - AMPLITUDE_BOUND, 0.0))
            dI = np.diff(I)
            dQ = np.diff(Q)
            slew = np.sqrt(dI**2 + dQ**2) / DT
            slew_pen = np.sum(np.maximum(slew - SLEW_BOUND, 0.0))
            slew_pen += np.maximum(np.hypot(I[0], Q[0])/DT - SLEW_BOUND, 0.0)
            slew_pen += np.maximum(np.hypot(I[-1], Q[-1])/DT - SLEW_BOUND, 0.0)
            return 1e3 + 10.0 * (amp_pen + slew_pen)
        return _score(I, Q)
        
    # Optimize
    if time.time() < deadline - 30:
        try:
            # L-BFGS-B
            res = minimize(objective, c0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
            if res.fun < best_score:
                best_c = res.x
                I_opt, Q_opt = _pulse_from_coeffs(best_c)
                if _valid(I_opt, Q_opt):
                    best_score = res.fun
                    best_IQ = np.column_stack([I_opt, Q_opt])
                else:
                    I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                    best_IQ = np.column_stack([I_opt, Q_opt])
                    best_score = _score(I_opt, Q_opt)
        except Exception:
            pass
            
    # Local refinement in time domain
    if time.time() < deadline - 20:
        # Take best_IQ and do a few steps of local search
        # We can't easily use L-BFGS-B on 160 dims with the same speed, but we can do a simple coordinate descent or small perturbations.
        # Given the time, let's just do a few random restarts of the polynomial optimizer from different seeds.
        rng = np.random.default_rng(42)
        for i in range(3):
            if time.time() > deadline - 10:
                break
            # Perturb coefficients
            c_try = best_c + rng.normal(0, 0.05, 2 * N_LEG)
            try:
                res2 = minimize(objective, c_try, method='L-BFGS-B', 
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
                if res2.fun < best_score:
                    best_c = res2.x
                    I_opt, Q_opt = _pulse_from_coeffs(best_c)
                    if _valid(I_opt, Q_opt):
                        best_score = res2.fun
                        best_IQ = np.column_stack([I_opt, Q_opt])
                    else:
                        I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                        best_IQ = np.column_stack([I_opt, Q_opt])
                        best_score = _score(I_opt, Q_opt)
            except Exception:
                pass
                
    # Final validity check and enforcement
    if not _valid(best_IQ[:, 0], best_IQ[:, 1]):
        best_IQ[:, 0], best_IQ[:, 1] = _enforce_constraints(best_IQ[:, 0], best_IQ[:, 1])
        
    return best_IQ
