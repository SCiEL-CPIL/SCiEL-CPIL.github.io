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
    deadline = start_time + 280.0  # 280s safety margin

    # Time grid for constant steps
    t = np.linspace(0, T, N, endpoint=False) + dt / 2.0

    # Base Gaussian envelope (DRAG-style seed)
    sigma = T / 6.0
    center = T / 2.0

    def gaussian_env(center_shift=0.0):
        g = np.exp(-0.5 * ((t - center - center_shift) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma) ** 2)
        g = g - edge_val
        g = np.maximum(g, 0.0)
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        return g

    g = gaussian_env()
    g_deriv = np.gradient(g, dt)

    def make_pulse(sigma2, amp_scale=1.0, center_shift=0.0):
        gg = gaussian_env(center_shift)
        gg_deriv = np.gradient(gg, dt)
        I = amp_scale * gg.copy()
        Q = amp_scale * sigma2 * gg_deriv.copy()

        # Enforce amplitude bound
        amp = np.sqrt(I ** 2 + Q ** 2)
        max_amp = np.max(amp) if amp.size else 0.0
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I *= factor
            Q *= factor

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
            for _ in range(120):
                I *= 0.95
                Q *= 0.95
                if check_slew(I, Q):
                    break

        return I, Q

    # Hamiltonian / Liouvillian setup
    # a = diag(sqrt([1,2]), 1), n = a^dag a
    a = np.diag([1.0, np.sqrt(2.0), 1.0])
    a_dag = a.conj().T
    n = a_dag @ a
    n2 = n @ n

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / Tphi)

    def hamiltonian(detuning, scale, I, Q):
        H = (detuning * n
             + 0.5 * anharmonicity * n @ (n - np.eye(3))
             + 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag)))
        return H

    def liouvillian_step(detuning, scale, I, Q):
        H = hamiltonian(detuning, scale, I, Q)
        Hc = H.conj().T
        L1c = L1.conj().T
        L2c = L2.conj().T
        # Build Liouvillian matrix acting on rho (vectorized column-stacking)
        # L(rho) = -i[H, rho] + D1(rho) + D2(rho)
        # Vectorize with vec(rho) where vec stacks columns; use kron form.
        A = -1j * (np.kron(np.eye(3), H) - np.kron(Hc, np.eye(3)))
        for L, Lc in [(L1, L1c), (L2, L2c)]:
            A += (np.kron(Lc, L) - 0.5 * np.kron(L @ Lc, np.eye(3))
                  - 0.5 * np.kron(np.eye(3), L @ Lc))
        # Exponentiate for dt
        U = linalg.expm(A * dt)
        return U

    # Fidelity for a single state under a given pulse
    def fidelity_for_state(state, detuning, scale, I, Q):
        # Precompute step propagators
        Usteps = [liouvillian_step(detuning, scale, I[k], Q[k]) for k in range(N)]
        rho0 = np.outer(state, state.conj())
        rho = rho0.reshape(9, 1)
        for U in Usteps:
            rho = U @ rho
        rho = rho.reshape(3, 3)
        # Target: X gate
        X = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
        target = X @ state
        fid = np.real(np.vdot(target, X @ state))  # placeholder to avoid unused
        # Fidelity = <psi_target|rho|psi_target>
        fid = np.real(np.vdot(target, rho @ target))
        return max(0.0, min(1.0, fid))

    def score_pulse(I, Q):
        # Cases: detuning in (-2pi*0.002, 0, 2pi*0.002), scale in (0.97, 1, 1.03)
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        states = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
            (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
            (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
            (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        ]
        fids = []
        for d in detunings:
            for s in scales:
                case_fids = []
                for st in states:
                    f = fidelity_for_state(st, d, s, I, Q)
                    case_fids.append(f)
                fids.append(np.mean(case_fids))
        fids = np.array(fids)
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    # Search over DRAG parameter and small shifts/scales
    best_I, best_Q = None, None
    best_score = float('inf')

    sigma2_candidates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    amp_scales = [0.95, 1.0, 1.05]
    center_shifts = [-1.0, 0.0, 1.0]

    for sigma2 in sigma2_candidates:
        for amp_scale in amp_scales:
            for cs in center_shifts:
                I, Q = make_pulse(sigma2, amp_scale, cs)
                sc = score_pulse(I, Q)
                if sc < best_score:
                    best_score = sc
                    best_I, best_Q = I, Q
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if best_I is None:
        # Fallback: plain Gaussian with optimal-ish DRAG
        drive_amp = np.max(g)
        optimal_sigma2 = abs(anharmonicity) / (2 * drive_amp * sigma) if drive_amp > 0 else 0.2
        I, Q = make_pulse(optimal_sigma2, 1.0, 0.0)
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
