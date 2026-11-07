"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import linalg
from scipy.optimize import minimize

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
    deadline = start_time + 280.0  # safety margin

    # Time grid for constant steps
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Hamiltonian / Liouvillian setup
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)

    eye3 = np.eye(3)
    L1c = L1.conj().T
    L2c = L2.conj().T

    # Lindblad contribution to Liouvillian (9x9)
    Lindblad_part = np.zeros((9, 9), dtype=complex)
    for L, Lc in [(L1, L1c), (L2, L2c)]:
        Lindblad_part += (np.kron(Lc, L) - 0.5 * np.kron(L @ Lc, eye3)
                          - 0.5 * np.kron(eye3, L @ Lc))

    def hamiltonian_part(detuning, scale, I, Q):
        H = (detuning * n
             + 0.5 * anharmonicity * (n @ n - n)
             + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        Hc = H.conj().T
        return -1j * (np.kron(eye3, H) - np.kron(Hc, eye3))

    # States to simulate
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]

    # Pre-compute initial density matrices
    rho0s = np.array([np.outer(st, st.conj()).reshape(9) for st in states])

    # Target states for fidelity
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    target_states = [X_gate @ st for st in states]

    def score_pulse(I, Q):
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]

        fids = []
        for d in detunings:
            for s in scales:
                rho_batch = rho0s.copy()

                for k in range(N):
                    H_part = hamiltonian_part(d, s, I[k], Q[k])
                    L_full = Lindblad_part + H_part
                    U = linalg.expm(L_full * dt)
                    rho_batch = (U @ rho_batch.T).T

                case_fids = []
                for i in range(6):
                    rho_final = rho_batch[i].reshape(3, 3)
                    psi_target = target_states[i]
                    fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
                    case_fids.append(fid)
                fids.append(np.mean(case_fids))

        fids = np.array(fids)
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def make_pulse(I, Q):
        """Apply amplitude and slew constraints."""
        # Enforce amplitude bound
        amp = np.sqrt(I ** 2 + Q ** 2)
        max_amp = np.max(amp) if amp.size else 0.0
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I = I * factor
            Q = Q * factor

        # Enforce slew bound
        def check_slew(I, Q):
            slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
            dI = np.diff(I)
            dQ = np.diff(Q)
            slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
            slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
            m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
            return m <= slew_bound

        if not check_slew(I, Q):
            for _ in range(200):
                I = I * 0.95
                Q = Q * 0.95
                if check_slew(I, Q):
                    break

        return I, Q

    # Generate initial seed pulse using DRAG-style Gaussian
    sigma = T / 6.0
    center = T / 2.0

    def gaussian_env(center_shift=0.0, width_scale=1.0):
        sigma_eff = sigma * width_scale
        g = np.exp(-0.5 * ((t - center - center_shift) / sigma_eff) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma_eff) ** 2)
        g = g - edge_val
        g = np.maximum(g, 0.0)
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        return g

    # Start with a reasonable DRAG pulse
    g = gaussian_env()
    g_deriv = np.gradient(g, dt)
    I0 = g.copy()
    Q0 = 0.3 * g_deriv.copy()
    I0, Q0 = make_pulse(I0, Q0)

    # Convert to flat parameter vector for optimization
    x0 = np.concatenate([I0, Q0])

    def objective(x):
        if time.time() > deadline:
            return float('inf')
        I = x[:N]
        Q = x[N:]
        I, Q = make_pulse(I, Q)
        return score_pulse(I, Q)

    # Gradient-based optimization
    best_I, best_Q = I0, Q0
    best_score = objective(x0)

    # Try L-BFGS-B optimization
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 50, 'maxfun': 200, 'disp': False})
        if result.fun < best_score and time.time() < deadline:
            x_opt = result.x
            I_opt, Q_opt = make_pulse(x_opt[:N], x_opt[N:])
            best_I, best_Q = I_opt, Q_opt
            best_score = result.fun
    except Exception:
        pass

    # If optimization didn't help, try a few manual refinements
    if time.time() < deadline - 30:
        # Try different DRAG alphas with the Gaussian
        for drag_alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            if time.time() > deadline:
                break
            g = gaussian_env()
            g_deriv = np.gradient(g, dt)
            I = g.copy()
            Q = drag_alpha * g_deriv.copy()
            I, Q = make_pulse(I, Q)
            sc = score_pulse(I, Q)
            if sc < best_score:
                best_score = sc
                best_I, best_Q = I, Q

        # Try cosine window
        g_cos = 0.5 * (1 - np.cos(2 * np.pi * t / T))
        area_cos = np.sum(g_cos) * dt
        if area_cos > 0:
            g_cos = g_cos * (np.pi / area_cos)
        for drag_alpha in [0.0, 0.2, 0.4]:
            if time.time() > deadline:
                break
            I = g_cos.copy()
            Q = drag_alpha * np.gradient(g_cos, dt).copy()
            I, Q = make_pulse(I, Q)
            sc = score_pulse(I, Q)
            if sc < best_score:
                best_score = sc
                best_I, best_Q = I, Q

    I, Q = best_I, best_Q

    # Final validity check
    amp = np.sqrt(I ** 2 + Q ** 2)
    assert np.all(amp <= amp_bound + 1e-10), "Amplitude constraint violated"

    def check_slew(I, Q):
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
        return m <= slew_bound

    assert check_slew(I, Q), "Slew rate constraint violated"

    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END
