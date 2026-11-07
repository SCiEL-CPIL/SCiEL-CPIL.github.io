"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import linalg

def pulse(target_gate, hardware_model):
    """
    Generate a robust X_pi pulse for a 3-level transmon.

    Returns (80, 2) array of [I, Q] amplitudes in rad/ns.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")

    dt = 0.5  # ns
    N = 80
    T = N * dt  # 40 ns

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25  # rad/ns
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    # Time guard
    start_time = time.time()
    deadline = start_time + 310.0  # 310s safety margin (330s kill)

    # Time grid
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Hamiltonian / Liouvillian setup
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a
    eye3 = np.eye(3)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)

    def build_liouvillian(detuning, scale, I, Q):
        H = (detuning * n
             + 0.5 * anharmonicity * n @ (n - eye3)
             + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        Hc = H.conj().T
        A = -1j * (np.kron(eye3, H) - np.kron(Hc, eye3))
        for L, Lc in [(L1, L1.conj().T), (L2, L2.conj().T)]:
            LLc = L @ Lc
            A += (np.kron(Lc, L) - 0.5 * np.kron(LLc, eye3)
                  - 0.5 * np.kron(eye3, LLc))
        return A

    # Precompute Liouvillian exponentials for each (detuning, scale) and each step
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # States to test
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]

    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])

    # Precompute rho0 for each state
    rho0s = [np.outer(st, st.conj()).reshape(9, 1) for st in states]
    target_states = [X_gate @ st for st in states]

    def score_pulse_fast(I, Q):
        """Compute mean + max infidelity over 9 cases x 6 states."""
        infids = np.zeros(9)
        case_idx = 0
        for d in detunings:
            for s in scales:
                # Build step propagators for this (detuning, scale)
                Usteps = []
                for k in range(N):
                    A = build_liouvillian(d, s, I[k], Q[k])
                    Usteps.append(linalg.expm(A * dt))
                # Apply to all 6 states
                case_infids = []
                for si in range(6):
                    rho = rho0s[si].copy()
                    for k in range(N):
                        rho = Usteps[k] @ rho
                    rho = rho.reshape(3, 3)
                    tgt = target_states[si]
                    fid = np.real(np.vdot(tgt, rho @ tgt))
                    fid = max(0.0, min(1.0, fid))
                    case_infids.append(1.0 - fid)
                infids[case_idx] = np.mean(case_infids)
                case_idx += 1
        return np.mean(infids) + np.max(infids)

    def check_valid(I, Q):
        amp = np.sqrt(I ** 2 + Q ** 2)
        if np.any(amp > amp_bound + 1e-9):
            return False
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        if slew0 > slew_bound + 1e-9:
            return False
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        if np.any(slews > slew_bound + 1e-9):
            return False
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        if slewN > slew_bound + 1e-9:
            return False
        return True

    def project_constraints(I, Q):
        """Project pulse onto feasible set."""
        # Amplitude constraint
        amp = np.sqrt(I ** 2 + Q ** 2)
        mask = amp > amp_bound
        if np.any(mask):
            factor = amp_bound / amp[mask]
            I[mask] *= factor
            Q[mask] *= factor

        # Slew constraint: iterative smoothing
        for _ in range(50):
            slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
            if slew0 > slew_bound:
                f = slew_bound * dt
                amp0 = np.sqrt(I[0] ** 2 + Q[0] ** 2)
                if amp0 > 0:
                    I[0] *= f / amp0
                    Q[0] *= f / amp0
                else:
                    break

            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
            max_slew = np.max(slews) if slews.size else 0.0
            if max_slew > slew_bound:
                # Scale down all steps proportionally
                scale = slew_bound / max_slew
                # Simple approach: reduce overall amplitude
                I *= 0.9
                Q *= 0.9
                continue

            slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
            if slewN > slew_bound:
                f = slew_bound * dt
                ampN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2)
                if ampN > 0:
                    I[-1] *= f / ampN
                    Q[-1] *= f / ampN
                else:
                    break

            if max_slew <= slew_bound and slew0 <= slew_bound and slewN <= slew_bound:
                break

        return I, Q

    # Generate DRAG seed
    sigma = T / 6.0
    center = T / 2.0
    g = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    edge_val = np.exp(-0.5 * ((0 - center) / sigma) ** 2)
    g = g - edge_val
    g = np.maximum(g, 0.0)
    area = np.sum(g) * dt
    if area > 0:
        g = g * (np.pi / area)

    g_deriv = np.gradient(g, dt)
    # DRAG: Q proportional to derivative of envelope
    # Optimal DRAG parameter: sigma2 = anharmonicity / (2 * drive_amp)
    drive_amp = np.max(g)
    sigma2_drag = abs(anharmonicity) / (2.0 * drive_amp) if drive_amp > 0 else 0.2

    I_seed = g.copy()
    Q_seed = sigma2_drag * g_deriv.copy()

    # Project seed
    I_seed, Q_seed = project_constraints(I_seed, Q_seed)

    # Evaluate seed
    seed_score = score_pulse_fast(I_seed, Q_seed)
    best_I, best_Q = I_seed.copy(), Q_seed.copy()
    best_score = seed_score

    # Nelder-Mead optimization on 160-dim pulse
    # Encode pulse as flat vector
    def encode(I, Q):
        return np.concatenate([I, Q])

    def decode(x):
        I = x[:N].copy()
        Q = x[N:].copy()
        return I, Q

    def objective(x):
        I, Q = decode(x)
        I, Q = project_constraints(I, Q)
        sc = score_pulse_fast(I, Q)
        return sc

    # Use scipy.optimize.minimize with Nelder-Mead
    from scipy.optimize import minimize

    x0 = encode(I_seed, Q_seed)

    # Nelder-Mead parameters
    nm_options = {
        'maxiter': 200,
        'maxfev': 200,
        'xatol': 0.01,
        'fatol': 1e-5,
        'adaptive': True
    }

    # Run optimization with time checks
    result = None
    try:
        result = minimize(objective, x0, method='Nelder-Mead', options=nm_options)
        if result.fun < best_score:
            I_opt, Q_opt = decode(result.x)
            I_opt, Q_opt = project_constraints(I_opt, Q_opt)
            if check_valid(I_opt, Q_opt):
                best_I, best_Q = I_opt, Q_opt
                best_score = result.fun
    except Exception:
        pass

    # If time remains, try a second round with tighter tolerances
    if time.time() < deadline - 60:
        try:
            nm_options2 = {
                'maxiter': 150,
                'maxfev': 150,
                'xatol': 0.005,
                'fatol': 5e-6,
                'adaptive': True
            }
            x_start = encode(best_I, best_Q)
            result2 = minimize(objective, x_start, method='Nelder-Mead', options=nm_options2)
            if result2.fun < best_score:
                I_opt, Q_opt = decode(result2.x)
                I_opt, Q_opt = project_constraints(I_opt, Q_opt)
                if check_valid(I_opt, Q_opt):
                    best_I, best_Q = I_opt, Q_opt
                    best_score = result2.fun
        except Exception:
            pass

    # If time remains, try a third round
    if time.time() < deadline - 30:
        try:
            nm_options3 = {
                'maxiter': 100,
                'maxfev': 100,
                'xatol': 0.002,
                'fatol': 2e-6,
                'adaptive': True
            }
            x_start = encode(best_I, best_Q)
            result3 = minimize(objective, x_start, method='Nelder-Mead', options=nm_options3)
            if result3.fun < best_score:
                I_opt, Q_opt = decode(result3.x)
                I_opt, Q_opt = project_constraints(I_opt, Q_opt)
                if check_valid(I_opt, Q_opt):
                    best_I, best_Q = I_opt, Q_opt
                    best_score = result3.fun
        except Exception:
            pass

    I, Q = best_I, best_Q

    # Final validity check
    amp = np.sqrt(I ** 2 + Q ** 2)
    assert np.all(amp <= amp_bound + 1e-9), "Amplitude constraint violated"

    slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
    assert slew0 <= slew_bound + 1e-9, "Slew rate constraint violated (start)"
    dI = np.diff(I)
    dQ = np.diff(Q)
    slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
    assert np.all(slews <= slew_bound + 1e-9), "Slew rate constraint violated (interior)"
    slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
    assert slewN <= slew_bound + 1e-9, "Slew rate constraint violated (end)"

    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END
