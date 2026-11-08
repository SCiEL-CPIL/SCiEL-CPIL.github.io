# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, differential_evolution
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    dt = 0.5
    N = 80
    total_time = N * dt

    # 3-level operators
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a
    I3 = np.eye(3)

    # Lindblad operators
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    L1_dag_L1 = L1.conj().T @ L1
    L2_dag_L2 = L2.conj().T @ L2

    L1_term = np.kron(L1, L1.T)
    L2_term = np.kron(L2, L2.T)
    L1_diss_left = np.kron(I3, L1_dag_L1)
    L1_diss_right = np.kron(L1_dag_L1.T, I3)
    L2_diss_left = np.kron(I3, L2_dag_L2)
    L2_diss_right = np.kron(L2_dag_L2.T, I3)

    static_Liouv = L1_term - 0.5 * (L1_diss_left + L1_diss_right) \
                  + L2_term - 0.5 * (L2_diss_left + L2_diss_right)

    an_ham = 0.5 * anharmonicity * n_op @ (n_op - I3)

    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])

    case_static_L = np.zeros((9, 9, 9), dtype=complex)
    case_scale = np.zeros(9)
    case_idx = 0
    for det in detunings:
        for sc in scales:
            H_static = det * n_op + an_ham
            H_left = np.kron(I3, H_static)
            H_right = np.kron(H_static.T, I3)
            case_static_L[case_idx] = -1j * (H_left - H_right) + static_Liouv
            case_scale[case_idx] = sc
            case_idx += 1

    # Precompute drive derivative Liouvillian components
    dHdI_base = 0.5 * (a + a_dag)
    dHdQ_base = 0.5j * (a - a_dag)
    dL_dI_base = -1j * (np.kron(I3, dHdI_base) - np.kron(dHdI_base.T, I3))
    dL_dQ_base = -1j * (np.kron(I3, dHdQ_base) - np.kron(dHdQ_base.T, I3))

    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

    # X_pi target: -i * swap |0> and |1>
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = -1j * s[1]
        t[1] = -1j * s[0]
        t[2] = s[2]
        target_states.append(t)

    # Precompute fidelity gradient matrices
    fid_grad_mats = np.zeros((6, 9), dtype=float)
    for si in range(6):
        psi_t = target_states[si]
        fid_grad_mats[si] = np.outer(np.conj(psi_t), psi_t).real.flatten()

    # Precompute rho_init
    rho_init = np.zeros((6, 9), dtype=complex)
    for si in range(6):
        psi = states[si]
        rho_mat = np.outer(psi, psi.conj())
        rho_init[si] = rho_mat.flatten()

    def check_validity(pulse_IQ):
        amplitudes = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
        if np.any(amplitudes > amplitude_bound + 1e-12):
            return False
        prev = np.zeros(2)
        for i in range(N):
            curr = pulse_IQ[i]
            if np.linalg.norm(curr - prev) / dt > slew_bound + 1e-12:
                return False
            prev = curr
        if np.linalg.norm(prev) / dt > slew_bound + 1e-12:
            return False
        return True

    def make_drag_pulse(alpha=0.0, sigma_ratio=1.0):
        t = np.arange(N) * dt
        sigma = (40.0 / 6.0) * sigma_ratio
        center = total_time / 2.0
        gaussian = np.exp(-0.5 * ((t - center) / sigma)**2)
        edge_val = gaussian[0]
        gaussian -= edge_val
        area = np.sum(gaussian) * dt
        if abs(area) > 1e-15:
            gaussian *= np.pi / area
        I = gaussian.copy()
        if alpha != 0:
            dI_dt = np.gradient(I, dt)
            Q = alpha * dI_dt
        else:
            Q = np.zeros(N)
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
        return pulse_IQ

    def simulate_pulse(pulse_IQ):
        """Simulate all 9 cases x 6 states. Returns rho_final (9,6,9) and U_all (9,N,9,9)."""
        U_all = np.zeros((9, N, 9, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                I_val = pulse_IQ[step, 0]
                Q_val = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)

        rho_final = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                rho_final[ci, si] = vec

        return rho_final, U_all

    def simulate_pulse_with_steps(pulse_IQ):
        """Returns rho_steps (9,N+1,6,9) and U_all (9,N,9,9)."""
        U_all = np.zeros((9, N, 9, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                I_val = pulse_IQ[step, 0]
                Q_val = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (I_val * (a + a_dag) + 1j * Q_val * (a - a_dag))
                H_left = np.kron(I3, H_drive)
                H_right = np.kron(H_drive.T, I3)
                L = case_static_L[ci] - 1j * (H_left - H_right)
                U_all[ci, step] = expm(dt * L)

        rho_steps = np.zeros((9, N + 1, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                rho_steps[ci, 0, si] = vec
                for step in range(N):
                    vec = U_all[ci, step] @ vec
                    rho_steps[ci, step + 1, si] = vec

        return rho_steps, U_all

    def compute_fidelities(rho_final):
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fids[ci, si] = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
        return fids

    def compute_soft_score(fids, beta=50.0):
        infids = 1.0 - fids
        m = np.max(infids)
        soft_max = m + np.log(np.sum(np.exp(beta * (infids - m)))) / beta
        return np.mean(infids) + soft_max

    def compute_adj_gradient(params, beta=50.0):
        """Compute score and gradient using adjoint method."""
        I = params[:N]
        Q = params[N:2*N]
        p = np.column_stack([I, Q])
        if not check_validity(p):
            return 1e10, np.zeros(2*N)

        rho_steps, U_all = simulate_pulse_with_steps(p)
        rho_final = rho_steps[:, N, :, :]
        fids = compute_fidelities(rho_final)
        s = compute_soft_score(fids, beta=beta)

        infids = 1.0 - fids
        m = np.max(infids)
        exp_vals = np.exp(beta * (infids - m))
        sum_exp = np.sum(exp_vals)
        d_s_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * beta)
        d_s_d_fid = -d_s_d_infid.reshape(9, 6)

        grad_params = np.zeros(2*N)
        nodes = np.array([0.5 - 0.5 * np.sqrt(5.0 / 9.0), 0.5, 0.5 + 0.5 * np.sqrt(5.0 / 9.0)])
        weights = np.array([0.3535533905932738, 0.2820947917738781, 0.3535533905932738])

        for ci in range(9):
            sc = case_scale[ci]
            dL_dI = sc * dL_dI_base
            dL_dQ = sc * dL_dQ_base

            lam = np.zeros((6, 9))
            for si in range(6):
                lam[si] = d_s_d_fid[ci, si] * fid_grad_mats[si]

            for step in range(N - 1, -1, -1):
                U_step = U_all[ci, step]
                H_drive = 0.5 * sc * (I[step] * (a + a_dag) + 1j * Q[step] * (a - a_dag))
                L_step = case_static_L[ci] - 1j * (np.kron(I3, H_drive) - np.kron(H_drive.T, I3))

                dS_dU = np.zeros((9, 9), dtype=complex)
                for si in range(6):
                    dS_dU += np.outer(lam[si], rho_steps[ci, step, si])

                dU_dI = np.zeros((9, 9), dtype=complex)
                dU_dQ = np.zeros((9, 9), dtype=complex)
                for k in range(3):
                    sk = nodes[k]
                    wk = weights[k]
                    E1 = expm((1.0 - sk) * dt * L_step)
                    E2 = expm(sk * dt * L_step)
                    dU_dI += wk * dt * (E1 @ dL_dI @ E2)
                    dU_dQ += wk * dt * (E1 @ dL_dQ @ E2)

                grad_params[step] += np.real(np.sum(np.conj(dS_dU) * dU_dI))
                grad_params[N + step] += np.real(np.sum(np.conj(dS_dU) * dU_dQ))

                for si in range(6):
                    lam[si] = U_step.T @ lam[si]

        return s, grad_params

    # Time management
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG sweep for good starting point
    best_pulse = make_drag_pulse(0.0)
    rho_best, _ = simulate_pulse(best_pulse)
    best_score = compute_soft_score(compute_fidelities(rho_best), beta=50.0)

    alphas = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
              -0.05, -0.10, -0.15, -0.20]
    sigma_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    for sr in sigma_ratios:
        if time_remaining() < 60:
            break
        for alpha in alphas:
            if time_remaining() < 60:
                break
            p_test = make_drag_pulse(alpha, sr)
            if check_validity(p_test):
                rho_t, _ = simulate_pulse(p_test)
                s = compute_soft_score(compute_fidelities(rho_t), beta=50.0)
                if s < best_score:
                    best_score = s
                    best_pulse = p_test.copy()

    # Phase 2: Low-dimensional Gaussian basis optimization
    # Represent pulse as: I(t) = sum_k c_k * g_k(t), Q(t) = sum_k d_k * h_k(t)
    # where g_k are Gaussians for I, h_k are derivative-Gaussians for Q (DRAG-like)
    t_arr = np.arange(N) * dt
    center = total_time / 2.0

    # Basis functions: 4 Gaussians for I, 4 for Q
    num_basis = 4
    centers_I = np.array([8.0, 16.0, 24.0, 32.0])  # spread across pulse
    widths_I = np.array([6.0, 5.0, 5.0, 6.0])
    centers_Q = np.array([8.0, 16.0, 24.0, 32.0])
    widths_Q = np.array([6.0, 5.0, 5.0, 6.0])

    def basis_gaussians(t, centers, widths):
        """Return matrix of shape (N, len(centers)) with Gaussian basis values."""
        B = np.zeros((N, len(centers)))
        for k in range(len(centers)):
            B[:, k] = np.exp(-0.5 * ((t - centers[k]) / widths[k])**2)
        return B

    B_I = basis_gaussians(t_arr, centers_I, widths_I)
    B_Q = basis_gaussians(t_arr, centers_Q, widths_Q)

    def basis_params_to_pulse(bparams):
        """Convert basis parameters to pulse. bparams = [c_I(4), c_Q(4)]."""
        c_I = bparams[:num_basis]
        c_Q = bparams[num_basis:2*num_basis]
        I = B_I @ c_I
        Q = B_Q @ c_Q
        # Normalize I area to pi
        area_I = np.sum(I) * dt
        if abs(area_I) > 1e-10:
            I *= np.pi / area_I
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            max_amp = np.max(np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
        return pulse_IQ

    def basis_obj(bparams):
        if time_remaining() < 15:
            return 1e10
        p = basis_params_to_pulse(bparams)
        if not check_validity(p):
            return 1e10
        rho_f, _ = simulate_pulse(p)
        fids = compute_fidelities(rho_f)
        return compute_soft_score(fids, beta=50.0)

    # Initialize basis params from best_pulse by projection
    def pulse_to_basis(pulse_IQ):
        I = pulse_IQ[:, 0]
        Q = pulse_IQ[:, 1]
        # Least squares projection
        c_I, _, _, _ = np.linalg.lstsq(B_I, I, rcond=None)
        c_Q, _, _, _ = np.linalg.lstsq(B_Q, Q, rcond=None)
        return np.concatenate([c_I, c_Q])

    bparams0 = pulse_to_basis(best_pulse)

    if time_remaining() > 120:
        bounds_basis = [(-0.15, 0.15)] * (2 * num_basis)
        try:
            result_basis = differential_evolution(basis_obj, bounds_basis, seed=42,
                                                   maxiter=80, popsize=15,
                                                   tol=1e-8, mutation=(0.5, 1.0),
                                                   recombination=0.7, init='latinhypercube')
            if time_remaining() > 60:
                p_b = basis_params_to_pulse(result_basis.x)
                if check_validity(p_b):
                    rho_b, _ = simulate_pulse(p_b)
                    s_b = compute_soft_score(compute_fidelities(rho_b), beta=50.0)
                    if s_b < best_score:
                        best_score = s_b
                        best_pulse = p_b.copy()
        except Exception:
            pass

        # Refine with L-BFGS-B in basis space
        if time_remaining() > 60:
            def basis_jac(bparams):
                if time_remaining() < 10:
                    return 1e10, np.zeros(2 * num_basis)
                p = basis_params_to_pulse(bparams)
                if not check_validity(p):
                    return 1e10, np.zeros(2 * num_basis)
                rho_f, _ = simulate_pulse(p)
                fids = compute_fidelities(rho_f)
                return compute_soft_score(fids, beta=50.0)

            try:
                result_b2 = minimize(basis_jac, bparams0, method='L-BFGS-B',
                                     bounds=bounds_basis,
                                     options={'maxiter': 50, 'ftol': 1e-14, 'maxfev': 200})
                if time_remaining() > 40:
                    p_b2 = basis_params_to_pulse(result_b2.x)
                    if check_validity(p_b2):
                        rho_b2, _ = simulate_pulse(p_b2)
                        s_b2 = compute_soft_score(compute_fidelities(rho_b2), beta=50.0)
                        if s_b2 < best_score:
                            best_score = s_b2
                            best_pulse = p_b2.copy()
            except Exception:
                pass

    # Phase 3: Raw-value adjoint gradient optimization (fine-tuning)
    if time_remaining() > 50:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def raw_jac(params):
            if time_remaining() < 8:
                return 1e10, np.zeros(2*N)
            return compute_adj_gradient(params, beta=50.0)

        try:
            result3 = minimize(raw_jac, x0, method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-9, 'maxfev': 300})
            if time_remaining() > 20:
                p3 = result3.x.reshape(N, 2)
                if check_validity(p3):
                    rho3, _ = simulate_pulse(p3)
                    s3 = compute_soft_score(compute_fidelities(rho3), beta=50.0)
                    if s3 < best_score:
                        best_score = s3
                        best_pulse = p3.copy()
        except Exception:
            pass

    # Phase 4: Hard-score polish (mean + max infidelity)
    if time_remaining() > 25:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def hard_jac(params):
            if time_remaining() < 5:
                return 1e10, np.zeros(2*N)
            I = params[:N]
            Q = params[N:2*N]
            p = np.column_stack([I, Q])
            if not check_validity(p):
                return 1e10, np.zeros(2*N)
            rho_f, U_all = simulate_pulse(p)
            fids = compute_fidelities(rho_f)
            infids = 1.0 - fids
            score = np.mean(infids) + np.max(infids)
            beta_hard = 500.0
            m = np.max(infids)
            exp_vals = np.exp(beta_hard * (infids - m))
            sum_exp = np.sum(exp_vals)
            d_score_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * beta_hard)
            d_score_d_fid = -d_score_d_infid.reshape(9, 6)
            grad_params = np.zeros(2*N)
            rho_steps, U_all2 = simulate_pulse_with_steps(p)
            nodes = np.array([0.5 - 0.5 * np.sqrt(5.0 / 9.0), 0.5, 0.5 + 0.5 * np.sqrt(5.0 / 9.0)])
            weights = np.array([0.3535533905932738, 0.2820947917738781, 0.3535533905932738])
            for ci in range(9):
                sc = case_scale[ci]
                dL_dI = sc * dL_dI_base
                dL_dQ = sc * dL_dQ_base
                lam = np.zeros((6, 9))
                for si in range(6):
                    lam[si] = d_score_d_fid[ci, si] * fid_grad_mats[si]
                for step in range(N - 1, -1, -1):
                    U_step = U_all2[ci, step]
                    H_drive = 0.5 * sc * (I[step] * (a + a_dag) + 1j * Q[step] * (a - a_dag))
                    L_step = case_static_L[ci] - 1j * (np.kron(I3, H_drive) - np.kron(H_drive.T, I3))
                    dS_dU = np.zeros((9, 9), dtype=complex)
                    for si in range(6):
                        dS_dU += np.outer(lam[si], rho_steps[ci, step, si])
                    dU_dI = np.zeros((9, 9), dtype=complex)
                    dU_dQ = np.zeros((9, 9), dtype=complex)
                    for k in range(3):
                        sk = nodes[k]
                        wk = weights[k]
                        E1 = expm((1.0 - sk) * dt * L_step)
                        E2 = expm(sk * dt * L_step)
                        dU_dI += wk * dt * (E1 @ dL_dI @ E2)
                        dU_dQ += wk * dt * (E1 @ dL_dQ @ E2)
                    grad_params[step] += np.real(np.sum(np.conj(dS_dU) * dU_dI))
                    grad_params[N + step] += np.real(np.sum(np.conj(dS_dU) * dU_dQ))
                    for si in range(6):
                        lam[si] = U_step.T @ lam[si]
            return score, grad_params

        try:
            result4 = minimize(hard_jac, x0, method='L-BFGS-B', jac=True,
                               bounds=bounds,
                               options={'maxiter': 30, 'ftol': 1e-16, 'gtol': 1e-10, 'maxfev': 150})
            if time_remaining() > 5:
                p4 = result4.x.reshape(N, 2)
                if check_validity(p4):
                    rho4, _ = simulate_pulse(p4)
                    fids4 = compute_fidelities(rho4)
                    s_soft4 = compute_soft_score(fids4, beta=50.0)
                    if s_soft4 < best_score:
                        best_score = s_soft4
                        best_pulse = p4.copy()
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
