# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
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

    # Precompute static parts of Liouvillian
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

    # Precompute dL/dI and dL/dQ for each case
    dL_dI_all = np.zeros((9, 9, 9), dtype=complex)
    dL_dQ_all = np.zeros((9, 9, 9), dtype=complex)
    for ci in range(9):
        sc = case_scale[ci]
        dH_dI = 0.5 * sc * (a + a_dag)
        dH_dQ = 0.5 * sc * 1j * (a - a_dag)
        dL_dI_all[ci] = -1j * (np.kron(I3, dH_dI) - np.kron(dH_dI.T, I3))
        dL_dQ_all[ci] = -1j * (np.kron(I3, dH_dQ) - np.kron(dH_dQ.T, I3))

    # Test states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]

    # X_pi target
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = -1j * s[1]
        t[1] = -1j * s[0]
        t[2] = s[2]
        target_states.append(t)

    # Fidelity gradient matrices: d fid / d rho_vec
    fid_grad_mats = []
    for si in range(6):
        psi_t = target_states[si]
        grad_mat = np.outer(np.conj(psi_t), psi_t).real
        fid_grad_mats.append(grad_mat.flatten())

    # Initial density matrices stacked: shape (9, 6, 9)
    rho_init_all = np.zeros((9, 6, 9), dtype=complex)
    for si in range(6):
        psi = states[si]
        rho_mat = np.outer(psi, psi.conj())
        rho_init_all[:, si] = rho_mat.flatten()

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

    def make_drag_pulse(alpha=0.0, sigma_factor=1.0):
        t = np.arange(N) * dt
        sigma = (40.0 / 6.0) * sigma_factor
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
            max_amp = np.max(np.sqrt(I**2 + Q**2))
            if max_amp > amplitude_bound:
                pulse_IQ *= amplitude_bound / max_amp
        return pulse_IQ

    def simulate_all(pulse_IQ):
        """Vectorized simulation: returns rho_final (9,6,9) and U_all (9,N,9,9)."""
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

        # Vectorized state evolution: rho_vec (9, 6, 9)
        rho_vec = rho_init_all.copy()
        for step in range(N):
            for ci in range(9):
                U = U_all[ci, step]
                # rho_vec[ci, si, :] = U @ rho_vec[ci, si, :]
                rho_vec[ci] = np.einsum('ij,kj->ki', U, rho_vec[ci])

        return rho_vec, U_all

    def compute_fidelities(rho_final):
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                rho_mat = rho_final[ci, si].reshape(3, 3)
                psi_t = target_states[si]
                fid = np.real(np.conj(psi_t) @ rho_mat @ psi_t)
                fids[ci, si] = fid
        return fids

    def compute_soft_score(fids, beta=50.0):
        infids = 1.0 - fids
        m = np.max(infids)
        soft_max = m + np.log(np.sum(np.exp(beta * (infids - m)))) / beta
        return np.mean(infids) + soft_max

    def objective_and_grad(params, beta=50.0):
        """Objective with TRUE adjoint gradient."""
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10, np.zeros(2*N)

        rho_final, U_all = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        score = compute_soft_score(fids, beta=beta)

        # d(score)/d(infid[k]) = 1/54 + exp(beta*(infid[k]-m)) / (beta * sum(exp(...)))
        infids = 1.0 - fids
        m = np.max(infids)
        exp_vals = np.exp(beta * (infids - m))
        sum_exp = np.sum(exp_vals)
        d_score_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * beta)
        d_score_d_fid = -d_score_d_infid.reshape(9, 6)

        # grad_rho[ci, si] = d_score_d_fid[ci,si] * fid_grad_mat[si]
        grad_rho = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                grad_rho[ci, si] = d_score_d_fid[ci, si] * fid_grad_mats[si]

        # Adjoint backward pass
        # lambda[ci, si] starts at grad_rho[ci, si]
        # For step from N-1 down to 0:
        #   grad at step uses lambda (which is d_score/d_rho_{step+1})
        #   then lambda[step] = U_all[ci, step].T @ lambda[step+1]
        lambda_state = grad_rho.copy()  # (9, 6, 9)

        grad_params = np.zeros(2 * N)

        for step in range(N - 1, -1, -1):
            I_val = I[step]
            Q_val = Q[step]

            for ci in range(9):
                sc = case_scale[ci]
                U_step = U_all[ci, step]

                # dU/dI ≈ dt * U @ dL_dI (first order)
                dU_dI = dt * U_step @ dL_dI_all[ci]
                dU_dQ = dt * U_step @ dL_dQ_all[ci]

                # d(score)/d(U_step[i,j]) = sum_si lambda[ci,si,i] * rho_step[ci,si,j]
                # We need rho_step (state BEFORE this step)
                # rho_step = U_all[ci, step-1] @ ... @ U_all[ci, 0] @ rho_init[si]
                # But we can get it from: rho_{step+1} = U_step @ rho_step
                # So rho_step = U_step^{-1} @ rho_{step+1}
                # Alternatively, recompute forward. Let's use the stored forward states.
                # Actually, let's just recompute: we need rho at each step.
                # For efficiency in the adjoint, let's precompute rho at all steps.

                # Hmm, this is getting expensive. Let me restructure.
                # Actually, let me just compute rho_step from rho_final by going backwards.
                # rho_{step} = U_step^{-1} @ rho_{step+1}
                # But U_step is 9x9, inversion is cheap.

                # Better approach: precompute rho at all steps in a forward pass.
                # Let me restructure to store rho at each step.

                pass

            # Propagate adjoint backward (for next iteration)
            if step > 0:
                for ci in range(9):
                    U_prev = U_all[ci, step - 1]
                    # lambda[ci, si] = U_prev^T @ lambda[ci, si]
                    # lambda[ci] shape (6, 9), U_prev shape (9, 9)
                    # lambda_new[si, i] = sum_j U_prev[j, i] * lambda[si, j]
                    lambda_state[ci] = np.einsum('ji,si->sj', U_prev, lambda_state[ci])

        # The above approach needs rho at each step. Let me redo with precomputed rho.
        # Actually, let me use a cleaner approach: store all rho states.

        return score, grad_params

    def objective_and_grad_v2(params, beta=50.0):
        """Objective with TRUE adjoint gradient - clean version."""
        I = params[:N]
        Q = params[N:2*N]
        pulse_IQ = np.column_stack([I, Q])
        if not check_validity(pulse_IQ):
            return 1e10, np.zeros(2*N)

        rho_final, U_all = simulate_all(pulse_IQ)
        fids = compute_fidelities(rho_final)
        score = compute_soft_score(fids, beta=beta)

        infids = 1.0 - fids
        m = np.max(infids)
        exp_vals = np.exp(beta * (infids - m))
        sum_exp = np.sum(exp_vals)
        d_score_d_infid = np.ones(54) / 54.0 + exp_vals.flatten() / (sum_exp * beta)
        d_score_d_fid = -d_score_d_infid.reshape(9, 6)

        # grad_rho[ci, si] = d_score_d_fid[ci,si] * fid_grad_mat[si]
        grad_rho = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                grad_rho[ci, si] = d_score_d_fid[ci, si] * fid_grad_mats[si]

        # Precompute rho at each step (forward pass, stored)
        # rho_at_step[ci, step, si, :] = rho vector after step
        # We already have rho_final = rho_at_step[ci, N, si]
        # Let's recompute forward and store
        rho_steps = np.zeros((9, N, 6, 9), dtype=complex)
        rho_vec = rho_init_all.copy()
        for step in range(N):
            for ci in range(9):
                U = U_all[ci, step]
                rho_vec[ci] = np.einsum('ij,kj->ki', U, rho_vec[ci])
                rho_steps[ci, step] = rho_vec[ci]

        # Adjoint backward pass
        lambda_state = grad_rho.copy()  # (9, 6, 9) = d_score/d_rho_{step+1}

        grad_params = np.zeros(2 * N)

        for step in range(N - 1, -1, -1):
            I_val = I[step]
            Q_val = Q[step]

            for ci in range(9):
                sc = case_scale[ci]
                U_step = U_all[ci, step]

                dU_dI = dt * U_step @ dL_dI_all[ci]
                dU_dQ = dt * U_step @ dL_dQ_all[ci]

                # rho_step[ci, si] = rho before this step
                if step == 0:
                    rho_step = rho_init_all[ci]  # (6, 9)
                else:
                    rho_step = rho_steps[ci, step - 1]  # (6, 9)

                # d(score)/d(U_step[i,j]) = sum_si lambda[ci,si,i] * rho_step[ci,si,j]
                # = einsum('si,sj->ij', lambda_state[ci], rho_step)
                d_score_d_U = np.einsum('si,sj->ij', lambda_state[ci], rho_step)

                # d(score)/dI = sum_{i,j} d_score_d_U[i,j] * dU_dI[i,j]
                # For real params: d(score)/dI = real(sum d_score_d_U * conj(dU_dI))
                # Actually: score is real, params are real.
                # d(score)/dI = Re(sum_{i,j} d_score_d_U[i,j] * conj(dU_dI[i,j]))
                grad_I_ci = np.real(np.sum(d_score_d_U * np.conj(dU_dI)))
                grad_Q_ci = np.real(np.sum(d_score_d_U * np.conj(dU_dQ)))

                grad_params[step] += grad_I_ci
                grad_params[N + step] += grad_Q_ci

            # Propagate adjoint: lambda[step] = U_step^T @ lambda[step+1]
            if step > 0:
                for ci in range(9):
                    U_step = U_all[ci, step]
                    # lambda_new[si, i] = sum_j U_step[j, i] * lambda[si, j]
                    lambda_state[ci] = np.einsum('ji,si->sj', U_step, lambda_state[ci])

        return score, grad_params

    # Time management
    start_time = time.time()
    deadline = start_time + 280.0

    def time_remaining():
        return deadline - time.time()

    # Phase 1: DRAG alpha sweep
    pulse_IQ = make_drag_pulse(0.0)
    best_pulse = pulse_IQ.copy()
    f_best = compute_fidelities(simulate_all(best_pulse)[0])
    best_score = compute_soft_score(f_best, beta=50.0)

    alpha_candidates = np.linspace(-0.5, 0.5, 21)
    for alpha in alpha_candidates:
        if time_remaining() < 30:
            break
        p_test = make_drag_pulse(alpha)
        if check_validity(p_test):
            f = compute_fidelities(simulate_all(p_test)[0])
            s = compute_soft_score(f, beta=50.0)
            if s < best_score:
                best_score = s
                best_pulse = p_test.copy()

    # Also try different sigma
    for sf in [0.8, 0.9, 1.1, 1.2]:
        if time_remaining() < 30:
            break
        for alpha in [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]:
            if time_remaining() < 15:
                break
            p_test = make_drag_pulse(alpha, sigma_factor=sf)
            if check_validity(p_test):
                f = compute_fidelities(simulate_all(p_test)[0])
                s = compute_soft_score(f, beta=50.0)
                if s < best_score:
                    best_score = s
                    best_pulse = p_test.copy()

    # Phase 2: L-BFGS-B with adjoint gradient
    if time_remaining() > 60:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2*N)

        def lbfgs_obj(params):
            if time_remaining() < 10:
                return 1e10, np.zeros(2*N)
            return objective_and_grad_v2(params, beta=50.0)

        result = minimize(lbfgs_obj, x0, method='L-BFGS-B', jac=True,
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfev': 500})
        if time_remaining() > 30:
            p_bfgs = result.x.reshape(N, 2)
            if check_validity(p_bfgs):
                f_bfgs = compute_fidelities(simulate_all(p_bfgs)[0])
                s_bfgs = compute_soft_score(f_bfgs, beta=50.0)
                if s_bfgs < best_score:
                    best_score = s_bfgs
                    best_pulse = p_bfgs.copy()

    # Phase 3: Second L-BFGS-B with higher beta for polishing
    if time_remaining() > 40:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2*N)

        def polish_obj(params):
            if time_remaining() < 10:
                return 1e10, np.zeros(2*N)
            return objective_and_grad_v2(params, beta=100.0)

        result2 = minimize(polish_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfev': 400})
        if time_remaining() > 15:
            p_pol = result2.x.reshape(N, 2)
            if check_validity(p_pol):
                f_pol = compute_fidelities(simulate_all(p_pol)[0])
                s_pol = compute_soft_score(f_pol, beta=100.0)
                if s_pol < best_score:
                    best_score = s_pol
                    best_pulse = p_pol.copy()

    # Phase 4: Final L-BFGS-B with even higher beta
    if time_remaining() > 25:
        x0 = best_pulse.flatten()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2*N)

        def final_obj(params):
            if time_remaining() < 8:
                return 1e10, np.zeros(2*N)
            return objective_and_grad_v2(params, beta=200.0)

        result3 = minimize(final_obj, x0, method='L-BFGS-B', jac=True,
                           bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfev': 300})
        if time_remaining() > 8:
            p_fin = result3.x.reshape(N, 2)
            if check_validity(p_fin):
                f_fin = compute_fidelities(simulate_all(p_fin)[0])
                s_fin = compute_soft_score(f_fin, beta=200.0)
                if s_fin < best_score:
                    best_score = s_fin
                    best_pulse = p_fin.copy()

    # Final validity check
    if not check_validity(best_pulse):
        best_pulse = make_drag_pulse(0.0)

    return best_pulse
# EVOLVE-BLOCK-END
