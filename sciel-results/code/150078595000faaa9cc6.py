import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar
import time

def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 305.0  # 330s wall-clock limit, safety margin

    # ---- constants ----
    alpha = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0
    deltas = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    a_op = np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)
    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    H0_nom = 0.5 * alpha * n_nm1
    driveI = 0.5 * (a_op + a_dag)
    driveQ = 0.5j * (a_op - a_dag)

    # Lindblad terms
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    d = 3
    eye_d = np.eye(d)

    # Precompute incoherent Liouvillian part (constant)
    incoh_L = np.zeros((d * d, d * d), dtype=complex)
    for Lk in [L1, L2]:
        Lkd = Lk.conj().T
        incoh_L += np.kron(Lk.conj(), Lk)
        LkdLk = Lkd @ Lk
        incoh_L += -0.5 * np.kron(eye_d, LkdLk)
        incoh_L += -0.5 * np.kron(LkdLk.T, eye_d)

    def get_H(I, Q, delta, scale):
        return H0_nom + delta * n_op + scale * (I * driveI + Q * driveQ)

    def build_Liouvillian(H):
        coh_L = -1j * (np.kron(H, eye_d) - np.kron(eye_d, H.T))
        return coh_L + incoh_L

    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        if np.max(np.sqrt(np.sum(diffs ** 2, axis=1))) / dt > slew_bound + 1e-9:
            return False
        return True

    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    targets = [
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 0, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1j, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),
    ]

    # Precompute fidelity matrices
    M_mats = []
    for s in range(6):
        M = np.outer(targets[s].conj(), targets[s])
        M_mats.append(M.flatten())
    M_mats = np.array(M_mats)

    rho_vecs_init = np.array([np.outer(states[s], states[s].conj()).flatten() for s in range(6)])

    def propagate_all(I_Q, delta, scale):
        """Propagate all 6 states and return fidelities."""
        rho_vecs = rho_vecs_init.copy()
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            L = build_Liouvillian(H)
            U = expm(L * dt)
            rho_vecs = (U @ rho_vecs.T).T
        fids = np.real(np.sum(M_mats * rho_vecs, axis=1))
        return fids

    def score_case(I_Q, delta, scale):
        res = propagate_all(I_Q, delta, scale)
        infs = 1.0 - res
        return infs.mean() + infs.max()

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        total = 0.0
        for d in deltas:
            for sc in scales:
                total += score_case(I_Q, d, sc)
        return total / 9.0

    # ---- Frequency-domain spectral shaping strategy ----
    # The pulse is defined in frequency domain as a Gaussian envelope with a
    # notch at the 1↔2 transition frequency, then inverse-transformed to time domain.
    # This analytically suppresses |2> leakage.

    # The 1↔2 transition frequency (in rad/ns) for a transmon with anharmonicity alpha:
    # omega_01 ≈ 2*pi*f0, omega_12 = omega_01 + alpha
    # We work in a rotating frame, so the relevant offset for leakage is alpha.
    # In the rotating frame, the drive couples |0>-|1> at frequency omega_d ≈ omega_01.
    # The 1↔2 transition is offset by alpha from the drive frequency.
    # So in the frequency domain of the pulse (relative to the drive), the 1↔2
    # transition appears at frequency alpha (which is negative: -2*pi*0.25).

    def spectral_to_time(spectral_params):
        """Convert spectral parameters to time-domain pulse.
        
        spectral_params: [A, sigma_f, notch_depth, notch_width, phase, q_offset]
        - A: overall amplitude scale
        - sigma_f: spectral bandwidth (rad/ns)
        - notch_depth: depth of the notch at alpha frequency (0 to 1)
        - notch_width: width of the notch (rad/ns)
        - phase: global phase
        - q_offset: constant Q offset
        """
        A, sigma_f, notch_depth, notch_width, phase, q_offset = spectral_params
        
        # Time grid
        t = np.linspace(0, 40, N)
        
        # Build frequency spectrum
        # We want a Gaussian envelope centered at 0 (in the rotating frame)
        # with a notch at frequency alpha (the 1↔2 offset)
        
        # Use a grid of frequencies
        # The pulse duration is 40 ns, so frequency resolution is ~1/(40) rad/ns
        # Use a fine frequency grid for IFFT
        n_freq = 4096
        freq = np.linspace(-n_freq * 0.25, n_freq * 0.25, n_freq)  # rad/ns range
        
        # Gaussian envelope centered at 0
        S = A * np.exp(-freq**2 / (2 * sigma_f**2))
        
        # Apply notch at alpha
        if notch_width > 0:
            notch = 1.0 - notch_depth * np.exp(-(freq - alpha)**2 / (2 * notch_width**2))
            S = S * notch
        
        # Apply phase
        S = S * np.exp(1j * phase)
        
        # Inverse FFT to get time-domain pulse
        # The IFFT gives us the time-domain signal
        I_t = np.real(np.fft.ifft(S)) * n_freq
        Q_t = np.imag(np.fft.ifft(S)) * n_freq
        
        # Normalize to match the desired area
        # For a pure X-pi gate, we need ∫I(t) dt = π
        area = np.trapz(I_t, t)
        if abs(area) > 1e-10:
            scale_factor = np.pi / area
            I_t = I_t * scale_factor
            Q_t = Q_t * scale_factor
        
        # Add Q offset
        Q_t = Q_t + q_offset
        
        # Project onto the time window [0, 40]
        I_t = I_t[:N]
        Q_t = Q_t[:N]
        
        # Enforce constraints by clipping
        I_Q = np.column_stack((I_t, Q_t))
        
        # Clip amplitudes
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        mask = amps > amp_bound
        if np.any(mask):
            I_Q[mask, :] = I_Q[mask, :] / amps[mask, None] * amp_bound
        
        # Enforce slew rate
        for i in range(N - 1):
            diff = np.sqrt((I_Q[i+1, 0] - I_Q[i, 0])**2 + (I_Q[i+1, 1] - I_Q[i, 1])**2)
            max_diff = slew_bound * dt
            if diff > max_diff:
                ratio = max_diff / diff
                I_Q[i+1, :] = I_Q[i, :] + (I_Q[i+1, :] - I_Q[i, :]) * ratio
        
        # Also check from zero to first step
        first_diff = np.sqrt(I_Q[0, 0]**2 + I_Q[0, 1]**2)
        max_first = slew_bound * dt
        if first_diff > max_first:
            ratio = max_first / first_diff
            I_Q[0, :] = I_Q[0, :] * ratio
        
        # Check from last step to zero
        last_diff = np.sqrt(I_Q[-1, 0]**2 + I_Q[-1, 1]**2)
        if last_diff > max_first:
            ratio = max_first / last_diff
            I_Q[-1, :] = I_Q[-1, :] * ratio
        
        return I_Q

    def seed_pulse():
        """DRAG-style Gaussian seed."""
        s = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0)**2 / (2 * s**2))
        return np.column_stack((I, np.zeros(N)))

    best = seed_pulse()
    best_s = score(best)

    # ---- Coarse grid over spectral parameters ----
    # spectral_params: [A, sigma_f, notch_depth, notch_width, phase, q_offset]
    # A: overall amplitude (will be rescaled to get area π)
    # sigma_f: spectral bandwidth
    # notch_depth: 0 to 1
    # notch_width: width of notch at alpha
    # phase: global phase
    # q_offset: Q offset

    A_vals = [0.3, 0.35, 0.4, 0.45]
    sigma_f_vals = [0.3, 0.5, 0.7, 1.0, 1.5]
    notch_depth_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    notch_width_vals = [0.1, 0.2, 0.3, 0.5, 0.8]
    phase_vals = [0.0]
    q_offset_vals = [-0.05, 0.0, 0.05]

    grid_count = 0
    max_grid = 2000  # Limit grid evaluations to stay within time budget

    for A in A_vals:
        for sf in sigma_f_vals:
            for nd in notch_depth_vals:
                for nw in notch_width_vals:
                    for ph in phase_vals:
                        for qo in q_offset_vals:
                            if time.time() > deadline - 80 or grid_count >= max_grid:
                                break
                            I_Q = spectral_to_time([A, sf, nd, nw, ph, qo])
                            if not check_constraints(I_Q):
                                continue
                            grid_count += 1
                            s = score(I_Q)
                            if s < best_s:
                                best_s, best = s, I_Q

    # ---- Refine with Nelder-Mead on spectral parameters ----
    def opt_spectral(p):
        if time.time() > deadline - 40:
            return 1e6
        I_Q = spectral_to_time(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    # Initialize from the best spectral params found in grid
    # We need to track which params gave the best score
    # For now, try a few initializations
    inits = [
        [0.4, 0.5, 0.5, 0.3, 0.0, 0.0],
        [0.35, 0.7, 0.8, 0.2, 0.0, 0.05],
        [0.45, 0.3, 1.0, 0.5, 0.0, -0.05],
        [0.4, 1.0, 0.3, 0.8, 0.0, 0.0],
    ]

    for init in inits:
        if time.time() > deadline - 40:
            break
        try:
            r = minimize(opt_spectral, init, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 0.005, 'fatol': 1e-7})
            if r.fun < best_s:
                I_Q = spectral_to_time(r.x)
                if check_constraints(I_Q):
                    best_s, best = r.fun, I_Q
        except Exception:
            pass

    # ---- Direct 80-step refinement (raw amplitudes) ----
    def opt_direct(x):
        if time.time() > deadline - 10:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    best_x = best.reshape(-1)
    for k in range(3):
        if time.time() > deadline - 10:
            break
        try:
            r = minimize(opt_direct, best_x, method='L-BFGS-B',
                         options={'maxiter': 40, 'ftol': 1e-8, 'gtol': 1e-5,
                                  'maxfun': 400, 'maxls': 20})
            I_Q = r.x.reshape(N, 2)
            if check_constraints(I_Q):
                s = score(I_Q)
                if s < best_s:
                    best_s, best, best_x = s, I_Q, r.x
            else:
                break
        except Exception:
            break

    if not check_constraints(best):
        best = seed_pulse()
    return best
