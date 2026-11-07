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
    deadline = start_time + 290.0  # 290s safety margin

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

    # States to simulate
    states = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        (np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) + 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
        (np.array([1.0, 0.0, 0.0]) - 1j * np.array([0.0, 1.0, 0.0])) / np.sqrt(2.0),
    ]
    
    rho0s = np.array([np.outer(st, st.conj()).reshape(9) for st in states])
    
    X_gate = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])
    target_states = [X_gate @ st for st in states]

    def hamiltonian_part(detuning, scale, I_val, Q_val):
        H = (detuning * n
             + 0.5 * anharmonicity * (n @ n - n)
             + 0.5 * scale * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag)))
        Hc = H.conj().T
        return -1j * (np.kron(eye3, H) - np.kron(Hc, eye3))

    # Pre-compute Liouvillians for each of the 9 cases for a given step k
    # But I, Q change per step, so we can't pre-compute the full Liouvillian for all steps.
    # However, we can pre-compute the H_part for the drive terms (I, Q) since they are linear.
    # H_drive = 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    # H_static = detuning * n + 0.5 * anharmonicity * (n @ n - n)
    
    # Let's pre-compute the kron terms for the drive part.
    # vec(H_drive rho) - vec(rho H_drive)
    # H_drive is linear in I, Q.
    
    # Actually, let's just compute the full Liouvillian matrix for each step in the scoring loop,
    # but optimize the matrix multiplication.
    # Since N=80 and 9 cases, that's 720 expm calls. expm of a 9x9 matrix is fast.
    # 720 * ~1ms = 0.72s per evaluation.
    # If we do 100 evaluations, that's 72s. This is feasible.

    def score_pulse(I, Q):
        detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        fids = []
        for d in detunings:
            for s in scales:
                rho_batch = rho0s.copy() # (6, 9)
                
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

    def check_slew(I, Q):
        slew0 = np.sqrt(I[0] ** 2 + Q[0] ** 2) / dt
        dI = np.diff(I)
        dQ = np.diff(Q)
        slews = np.sqrt(dI ** 2 + dQ ** 2) / dt
        slewN = np.sqrt(I[-1] ** 2 + Q[-1] ** 2) / dt
        m = max(slew0, np.max(slews) if slews.size else 0.0, slewN)
        return m <= slew_bound

    def make_pulse(params):
        # params: (sigma2, amp_scale, center_shift)
        sigma2, amp_scale, center_shift = params
        
        # Base Gaussian envelope
        sigma = T / 6.0
        center = T / 2.0
        g = np.exp(-0.5 * ((t - center - center_shift) / sigma) ** 2)
        edge_val = np.exp(-0.5 * ((0 - center - center_shift) / sigma) ** 2)
        g = g - edge_val
        g = np.maximum(g, 0.0)
        area = np.sum(g) * dt
        if area > 0:
            g = g * (np.pi / area)
        
        g_deriv = np.gradient(g, dt)
        I = amp_scale * g.copy()
        Q = amp_scale * sigma2 * g_deriv.copy()

        # Enforce amplitude bound
        amp = np.sqrt(I ** 2 + Q ** 2)
        max_amp = np.max(amp) if amp.size else 0.0
        if max_amp > amp_bound:
            factor = amp_bound / max_amp
            I *= factor
            Q *= factor

        # Enforce slew bound
        if not check_slew(I, Q):
            for _ in range(120):
                I *= 0.95
                Q *= 0.95
                if check_slew(I, Q):
                    break

        return I, Q

    # Optimization: Use a local search around the DRAG parameters
    # Initial guess: sigma2 ~ |anharmonicity| / (2 * drive_amp * sigma)
    drive_amp = np.pi / (T * np.sqrt(2 * np.pi) * (T/6))  # Rough estimate of max amp
    # Actually, let's just use the known DRAG formula: sigma2 = |chi| / (2 * Omega * sigma)
    # Omega is the Rabi frequency. Area = pi = Omega * tau_eff.
    # Let's start with a reasonable guess and refine.
    
    best_I, best_Q = None, None
    best_score = float('inf')
    
    # Coarse grid search
    sigma2_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    amp_scales = [0.9, 0.95, 1.0, 1.05, 1.1]
    center_shifts = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for sigma2 in sigma2_candidates:
        for amp_scale in amp_scales:
            for cs in center_shifts:
                if time.time() > deadline - 60.0:
                    break
                I, Q = make_pulse((sigma2, amp_scale, cs))
                if not check_slew(I, Q):
                    continue
                sc = score_pulse(I, Q)
                if sc < best_score:
                    best_score = sc
                    best_I, best_Q = I, Q
            if time.time() > deadline - 60.0:
                break
        if time.time() > deadline - 60.0:
            break

    if best_I is None:
        # Fallback to seed
        I, Q = make_pulse((0.2, 1.0, 0.0))
        best_I, best_Q = I, Q

    # Fine-tuning: Gradient-free local search (Nelder-Mead style)
    # We'll use a simple random perturbation search around the best point.
    # Since we don't have the exact parameters that produced best_I, best_Q,
    # we can try to recover them or just perturb the pulse directly.
    # Perturbing the pulse directly is safer.
    
    current_I, current_Q = best_I.copy(), best_Q.copy()
    current_score = best_score
    
    # Perform a few rounds of local search
    for round_num in range(5):
        if time.time() > deadline - 10.0:
            break
            
        improved = False
        for _ in range(20):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb I and Q
            noise_I = np.random.normal(0, 0.01, N)
            noise_Q = np.random.normal(0, 0.01, N)
            new_I = current_I + noise_I
            new_Q = current_Q + noise_Q
            
            # Check constraints
            amp = np.sqrt(new_I ** 2 + new_Q ** 2)
            if np.any(amp > amp_bound + 1e-10):
                continue
            if not check_slew(new_I, new_Q):
                continue
            
            sc = score_pulse(new_I, new_Q)
            if sc < current_score - 1e-6:
                current_I, current_Q = new_I, new_Q
                current_score = sc
                best_I, best_Q = new_I.copy(), new_Q.copy()
                best_score = sc
                improved = True
        
        if not improved:
            break

    I, Q = best_I, best_Q

    # Final validity check
    amp = np.sqrt(I ** 2 + Q ** 2)
    assert np.all(amp <= amp_bound + 1e-10), "Amplitude constraint violated"
    assert check_slew(I, Q), "Slew rate constraint violated"

    return np.column_stack([I, Q])

# EVOLVE-BLOCK-END
