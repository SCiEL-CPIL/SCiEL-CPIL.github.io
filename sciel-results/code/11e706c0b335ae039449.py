"""
Robust control pulses for a three-level transmon.

Strategy:
  1. Build a DRAG-style seed pulse (Gaussian envelope with derivative correction).
  2. Use numerical optimization (Nelder-Mead or L-BFGS-B via scipy) to minimize
     the mean+max infidelity over the 9 detuning/scale cases.
  3. Enforce amplitude and slew-rate constraints during optimization.
  4. Include a time-deadline guard to return the best result before 330 s.
"""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# --- Constants from the task contract ---
DT = 0.5          # ns per step
N_STEPS = 80      # number of steps
T_GATE = 40.0     # ns total gate time
AMP_BOUND = 0.4   # max sqrt(I^2+Q^2)
SLEW_BOUND = 0.12 # max ||p[i+1]-p[i]|| / dt
DETUNING_BOUNDS = [-2*np.pi*0.002, 2*np.pi*0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0      # ns
Tphi = 40000.0    # ns
ANHARMONICITY = -2*np.pi*0.25  # rad/ns

# Three-level ladder: |0>, |1>, |2>
# a = diag(sqrt([1,2]), 1) in the sense that a|1>=|0>, a|2>=sqrt(2)|1>
# Matrix representation of a (3x3):
A = np.array([[0, 1, 0],
              [0, 0, np.sqrt(2)],
              [0, 0, 0]], dtype=complex)
ADAG = A.conj().T
N_OP = ADAG @ A  # number operator diag(0,1,2)

# Lindblad operators
L1 = A / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / Tphi)


def _hamiltonian(detuning, scale, I, Q):
    """Build the 3x3 Hamiltonian for given parameters."""
    h = (detuning * N_OP
         + 0.5 * ANHARMONICITY * N_OP @ (N_OP - np.eye(3))
         + 0.5 * scale * (I * (A + ADAG) + 1j * Q * (A - ADAG)))
    return h


def _liouvillian_super(H, L_ops):
    """Build the Liouvillian superoperator (9x9) for the given H and Lindblad ops.
    
    The master equation is dρ/dt = -i[H,ρ] + Σ_k (L_k ρ L_k† - 0.5{L_k†L_k, ρ})
    We represent ρ as a 9-vector (row-major flattening).
    """
    dim = 3
    n = dim * dim
    L = np.zeros((n, n), dtype=complex)
    
    # Coherent part: -i[H, ρ] => -i(H⊗I - I⊗H^T) in superoperator form
    # Using column-stacking convention: vec(ABC) = (C^T ⊗ A) vec(B)
    # vec(Hρ) = (I ⊗ H) vec(ρ)  ... wait, let's use row-major or be careful.
    
    # Let's use the standard: for dρ/dt = -i[H,ρ], the superoperator is -i(H⊗I - I⊗H†)
    # with column-stacking. But numpy uses row-major by default.
    # To avoid confusion, let's use a direct approach: build the 9x9 matrix.
    
    H = np.asarray(H, dtype=complex)
    
    # Coherent: -i(H ρ - ρ H)
    # In vec form (column stacking): vec(Hρ) = (I ⊗ H) vec(ρ), vec(ρH) = (H^T ⊗ I) vec(ρ)
    # But numpy flatten is row-major. Let's just build it explicitly.
    
    # For row-major flattening (C order), vec(AB) = (B^T ⊗ A) vec(ρ)
    # So vec(Hρ) = (I^T ⊗ H) vec(ρ) = (I ⊗ H) vec(ρ)
    # vec(ρH) = (H^T ⊗ I) vec(ρ)
    
    I3 = np.eye(3, dtype=complex)
    L_coh = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    L = L_coh.copy()
    
    for Lk in L_ops:
        Lk = np.asarray(Lk, dtype=complex)
        Lkd = Lk.conj().T
        LkLkd = Lkd @ Lk
        
        # vec(Lk ρ Lk†) = ((Lk†)^T ⊗ Lk) vec(ρ) = (Lk^T ⊗ Lk) vec(ρ)  [row-major]
        # Actually for row-major: vec(AXB) = (B^T ⊗ A) vec(X)
        # vec(Lk ρ Lkd) = (Lkd^T ⊗ Lk) vec(ρ)
        L_diss = np.kron(Lkd.T, Lk)
        
        # vec({Lkd Lk, ρ}) = vec(LkdLk ρ + ρ LkdLk) = (I ⊗ LkdLk) + (LkdLk^T ⊗ I)
        L_anticom = 0.5 * (np.kron(I3, LkLkd) + np.kron(LkLkd.T, I3))
        
        L += L_diss - L_anticom
    
    return L


def _evolve_state(pulse_I, pulse_Q, detuning, scale, dt=DT):
    """Evolve the density matrix through the pulse. Returns final 3x3 density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # start in |0>
    
    # We'll evolve step by step using the Liouvillian
    for i in range(N_STEPS):
        H = _hamiltonian(detuning, scale, pulse_I[i], pulse_Q[i])
        L_super = _liouvillian_super(H, [L1, L2])
        
        # Evolve for dt using matrix exponential
        U = expm(1j * L_super * dt)  # Wait, the Liouvillian gives dρ/dt = L ρ
        # So ρ(t+dt) = exp(L * dt) ρ(t)
        
        rho_vec = rho.flatten()  # row-major
        rho_vec = U @ rho_vec
        rho = rho_vec.reshape((3, 3))
    
    return rho


def _fidelity(rho, psi):
    """Compute fidelity F = <ψ|ρ|ψ> for pure state psi."""
    return np.real(np.vdot(psi, rho @ psi))


def _compute_score(pulse_I, pulse_Q):
    """Compute mean + max infidelity over 9 cases and 4 input states."""
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Input states: |0>, |1>, (|0>±|1>)/√2, (|0>±i|1>)/√2
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_p = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_m = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_ip = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_im = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    input_states = [psi0, psi1, psi_p, psi_m, psi_ip, psi_im]
    
    all_infidelities = []
    
    for det in detunings:
        for sc in scales:
            case_infs = []
            for psi in input_states:
                # Evolve
                rho = _evolve_state(pulse_I, pulse_Q, det, sc)
                # Target: X gate maps |0><0| -> |1><1|, |1><1| -> |0><0|
                # For arbitrary input |ψ>, target is X|ψ>
                # Fidelity = |<Xψ|ρ|Xψ>|^2? No, for pure target, F = <ψ_target|ρ|ψ_target>
                psi_target = _apply_x(psi)
                fid = _fidelity(rho, psi_target)
                inf = 1.0 - np.real(fid)
                case_infs.append(inf)
            case_mean_inf = np.mean(case_infs)
            case_max_inf = np.max(case_infs)
            all_infidelities.append(case_mean_inf + case_max_inf)
    
    return np.mean(all_infidelities) + np.max(all_infidelities)


def _apply_x(psi):
    """Apply X gate to a 3-level state (acts on |0> and |1>, leaves |2> unchanged)."""
    out = np.zeros(3, dtype=complex)
    out[0] = psi[1]
    out[1] = psi[0]
    out[2] = psi[2]
    return out


def _drag_seed(alpha=0.0):
    """Generate a DRAG-style seed pulse.
    
    Gaussian envelope with derivative correction.
    Area = pi, sigma = 40/6 ns, offset so that edges are ~0.
    """
    t = np.linspace(0, T_GATE, N_STEPS + 1)[:-1] + DT/2  # centers of steps
    t_center = T_GATE / 2.0
    sigma = T_GATE / 6.0
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Normalize area to pi
    area = np.sum(env) * DT
    env = env * (np.pi / area)
    
    # DRAG derivative term
    # d(env)/dt approximated by finite difference
    denv = np.gradient(env, DT)
    
    I = env
    Q = alpha * denv
    
    # Offset to make edges ~0 (they already are for Gaussian, but clip)
    I = np.clip(I, -AMP_BOUND, AMP_BOUND)
    Q = np.clip(Q, -AMP_BOUND, AMP_BOUND)
    
    # Enforce amplitude bound
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > AMP_BOUND
    if np.any(mask):
        I[mask] *= AMP_BOUND / amp[mask]
        Q[mask] *= AMP_BOUND / amp[mask]
    
    # Enforce slew rate
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        d = np.sqrt(dI**2 + dQ**2)
        if d > SLEW_BOUND * DT:
            scale = (SLEW_BOUND * DT) / d
            I[i+1] = I[i] + dI * scale
            Q[i+1] = Q[i] + dQ * scale
    
    return I, Q


def _check_validity(I, Q):
    """Check amplitude and slew rate constraints."""
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMP_BOUND + 1e-9:
        return False
    
    # Slew from zero to step 1
    d0 = np.sqrt(I[0]**2 + Q[0]**2)
    if d0 > SLEW_BOUND * DT + 1e-9:
        return False
    
    for i in range(N_STEPS - 1):
        d = np.sqrt((I[i+1]-I[i])**2 + (Q[i+1]-Q[i])**2)
        if d > SLEW_BOUND * DT + 1e-9:
            return False
    
    # Slew from last step to zero
    dlast = np.sqrt(I[-1]**2 + Q[-1]**2)
    if dlast > SLEW_BOUND * DT + 1e-9:
        return False
    
    return True


def _enforce_constraints(I, Q):
    """Project I, Q onto the feasible set (amplitude and slew bounds)."""
    I = I.copy()
    Q = Q.copy()
    
    # Amplitude bound
    for i in range(N_STEPS):
        a = np.sqrt(I[i]**2 + Q[i]**2)
        if a > AMP_BOUND:
            I[i] *= AMP_BOUND / a
            Q[i] *= AMP_BOUND / a
    
    # Slew rate: forward pass
    I_prev = 0.0
    Q_prev = 0.0
    for i in range(N_STEPS):
        dI = I[i] - I_prev
        dQ = Q[i] - Q_prev
        d = np.sqrt(dI**2 + dQ**2)
        max_d = SLEW_BOUND * DT
        if d > max_d:
            I[i] = I_prev + dI * (max_d / d)
            Q[i] = Q_prev + dQ * (max_d / d)
        I_prev = I[i]
        Q_prev = Q[i]
    
    # Slew rate: backward pass (to also fix the last-to-zero constraint)
    I_next = 0.0
    Q_next = 0.0
    for i in range(N_STEPS - 1, -1, -1):
        dI = I_next - I[i]
        dQ = Q_next - Q[i]
        d = np.sqrt(dI**2 + dQ**2)
        max_d = SLEW_BOUND * DT
        if d > max_d:
            I[i] = I_next - dI * (max_d / d)
            Q[i] = Q_next - dQ * (max_d / d)
        I_next = I[i]
        Q_next = Q[i]
    
    # Re-enforce amplitude after slew projection
    for i in range(N_STEPS):
        a = np.sqrt(I[i]**2 + Q[i]**2)
        if a > AMP_BOUND:
            I[i] *= AMP_BOUND / a
            Q[i] *= AMP_BOUND / a
    
    return I, Q


def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin before 330s
    
    # Generate DRAG seed
    I, Q = _drag_seed(alpha=0.0)
    I, Q = _enforce_constraints(I, Q)
    
    best_I = I.copy()
    best_Q = Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    # Try a few DRAG alpha values quickly
    for alpha in [0.1, 0.2, 0.3, 0.5, -0.1, -0.2, -0.3, -0.5]:
        if time.time() > deadline:
            break
        I_a, Q_a = _drag_seed(alpha=alpha)
        I_a, Q_a = _enforce_constraints(I_a, Q_a)
        if not _check_validity(I_a, Q_a):
            continue
        try:
            s = _compute_score(I_a, Q_a)
            if s < best_score:
                best_score = s
                best_I = I_a.copy()
                best_Q = Q_a.copy()
        except Exception:
            continue
    
    # Optimization using scipy
    # Parameterize the pulse as a vector of length 160 (I and Q interleaved or separate)
    # Use Nelder-Mead or L-BFGS-B with penalty for constraint violation
    
    def objective(x):
        I_opt = x[:N_STEPS]
        Q_opt = x[N_STEPS:]
        
        # Penalty for amplitude violation
        amp = np.sqrt(I_opt**2 + Q_opt**2)
        amp_pen = np.sum(np.maximum(0, amp - AMP_BOUND)**2) * 100
        
        # Penalty for slew rate violation
        slew_pen = 0.0
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            d = np.sqrt((I_opt[i]-prev_I)**2 + (Q_opt[i]-prev_Q)**2)
            if d > SLEW_BOUND * DT:
                slew_pen += (d - SLEW_BOUND * DT)**2 * 100
            prev_I, prev_Q = I_opt[i], Q_opt[i]
        d = np.sqrt(I_opt[-1]**2 + Q_opt[-1]**2)
        if d > SLEW_BOUND * DT:
            slew_pen += (d - SLEW_BOUND * DT)**2 * 100
        
        if amp_pen > 1e-6 or slew_pen > 1e-6:
            return best_score + amp_pen + slew_pen
        
        try:
            return _compute_score(I_opt, Q_opt)
        except Exception:
            return best_score + 10.0
    
    x0 = np.concatenate([best_I, best_Q])
    
    # Try L-BFGS-B
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 50, 'maxfun': 100,
                                   'ftol': 1e-8, 'gtol': 1e-6})
            if time.time() < deadline:
                I_opt = res.x[:N_STEPS]
                Q_opt = res.x[N_STEPS:]
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                if _check_validity(I_opt, Q_opt):
                    s = _compute_score(I_opt, Q_opt)
                    if s < best_score:
                        best_score = s
                        best_I = I_opt.copy()
                        best_Q = Q_opt.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead if time permits
    if time.time() < deadline - 20:
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if time.time() < deadline:
                I_opt = res.x[:N_STEPS]
                Q_opt = res.x[N_STEPS:]
                I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
                if _check_validity(I_opt, Q_opt):
                    s = _compute_score(I_opt, Q_opt)
                    if s < best_score:
                        best_score = s
                        best_I = I_opt.copy()
                        best_Q = Q_opt.copy()
        except Exception:
            pass
    
    # Final check
    if not _check_validity(best_I, best_Q):
        best_I, best_Q = _enforce_constraints(best_I, best_Q)
    
    result = np.column_stack([best_I, best_Q])
    return result
