# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # ---------------- Hardware parameters ----------------
    anharmonicity = -2 * np.pi * 0.25
    amplitude_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0

    dt = 0.5
    N = 80
    total_time = N * dt
    t_grid = np.arange(N) * dt
    center = total_time / 2.0

    # ---------------- 3-level operators ----------------
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a            # diag(0,1,2)
    I3 = np.eye(3)

    # Precompute static Liouvillian (dissipation + anharmonicity + detuning per case)
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
    static_Liouv = (L1_term - 0.5 * (L1_diss_left + L1_diss_right)
                    + L2_term - 0.5 * (L2_diss_left + L2_diss_right))
    an_ham = 0.5 * anharmonicity * (n_op @ (n_op - I3))

    detunings = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
    scales = np.array([0.97, 1.0, 1.03])
    case_static_L = np.zeros((9, 9, 9), dtype=complex)
    case_scale = np.zeros(9)
    idx = 0
    for det in detunings:
        for sc in scales:
            H_static = det * n_op + an_ham
            H_left = np.kron(I3, H_static)
            H_right = np.kron(H_static.T, I3)
            case_static_L[idx] = -1j * (H_left - H_right) + static_Liouv
            case_scale[idx] = sc
            idx += 1

    # ---------------- States & targets ----------------
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    target_states = []
    for s in states:
        t = np.zeros(3, dtype=complex)
        t[0] = -1j * s[1]
        t[1] = -1j * s[0]
        t[2] = s[2]
        target_states.append(t)

    # Initial density-matrix vectors (6,9)
    rho_init = np.zeros((6, 9), dtype=complex)
    for si in range(6):
        psi = states[si]
        rho_init[si] = np.outer(psi, psi.conj()).flatten()

    # Precompute fidelity extraction: fid = Re[psi_t^dag rho psi_t]
    # = Re[ sum_ij conj(psi_t_i) rho_ij psi_t_j ] = Re[ v^T vec(rho) ] with v_ij = conj(psi_i) psi_j
    fid_vec = np.zeros((6, 9), dtype=complex)
    for si in range(6):
        p = target_states[si]
        M = np.outer(np.conj(p), p)          # M_ij = conj(p_i) p_j
        fid_vec[si] = M.flatten()

    def check_validity(pulse_IQ):
        amp = np.sqrt(pulse_IQ[:, 0] ** 2 + pulse_IQ[:, 1] ** 2)
        if np.any(amp > amplitude_bound + 1e-12):
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

    # ---------------- Parameterized DRAG pulse ----------------
    # I(t) = A * g(t) * (1 + beta * h(t))  where g is normalized Gaussian, h = small shape
    # Q(t) = -alpha * A * g'(t)  (DRAG)  +  gamma * A * g''(t)
    # params: [A, alpha, sigma, gamma, beta, offset]
    def gauss(t, sigma, off=0.0):
        return np.exp(-0.5 * ((t - center - off) / sigma) ** 2)

    def build_pulse(params):
        A = params[0]
        alpha = params[1]
        sigma = max(params[2], 0.5)
        gamma = params[3]
        beta = params[4]
        off = params[5]
        g = gauss(t_grid, sigma, off)
        # edge offset so that pulse ~0 at edges (helps slew from zero)
        edge = g[0]
        g = g - edge
        # normalize area to pi for the I part (so A controls amplitude scale ~ pi/ (sigma*sqrt(2pi)) )
        area = np.sum(g) * dt
        if abs(area) > 1e-15:
            g = g * (np.pi / area)
        # derivatives
        gp = np.gradient(g, dt)
        gpp = np.gradient(gp, dt)
        # small shape modulation to compensate detuning (higher-order)
        h = ((t_grid - center - off) / sigma) ** 2 - 1.0   # Hermite-like, zero-mean-ish
        I = A * g * (1.0 + beta * h)
        Q = -alpha * A * gp + gamma * A * gpp
        return np.column_stack([I, Q])

    # ---------------- Simulation + adjoint gradient ----------------
    def simulate_and_score(pulse_IQ, soft_beta=50.0):
        """
        Evolve all (case, state) and return (score, U_all, rho_final).
        U_all[ci, step] is the 9x9 step propagator.
        """
        U_all = np.zeros((9, N, 9, 9), dtype=complex)
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                Iv = pulse_IQ[step, 0]
                Qv = pulse_IQ[step, 1]
                H_drive = 0.5 * sc * (Iv * (a + a_dag) + 1j * Qv * (a - a_dag))
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
        fids = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                fids[ci, si] = np.real(np.vdot(fid_vec[si], rho_final[ci, si]))
        infids = 1.0 - fids
        if soft_beta is None:
            score = np.mean(infids) + np.max(infids)
        else:
            m = np.max(infids)
            soft_max = m + np.log(np.sum(np.exp(soft_beta * (infids - m)))) / soft_beta
            score = np.mean(infids) + soft_max
        return score, U_all, rho_final

    def exact_score(pulse_IQ):
        fids = np.zeros((9, 6))
        for ci in range(9):
            sc = case_scale[ci]
            for si in range(6):
                vec = rho_init[si].copy()
                for step in range(N):
                    Iv = pulse_IQ[step, 0]
                    Qv = pulse_IQ[step, 1]
                    H_drive = 0.5 * sc * (Iv * (a + a_dag) + 1j * Qv * (a - a_dag))
                    H_left = np.kron(I3, H_drive)
                    H_right = np.kron(H_drive.T, I3)
                    L = case_static_L[ci] - 1j * (H_left - H_right)
                    vec = expm(dt * L) @ vec
                fids[ci, si] = np.real(np.vdot(fid_vec[si], vec))
        infids = 1.0 - fids
        return np.mean(infids) + np.max(infids)

    def adjoint_grad(pulse_IQ, U_all, rho_final, soft_beta=50.0):
        """
        Analytic gradient of soft score w.r.t. pulse_IQ (N,2).
        Uses chain rule through the step propagators.
        d score / d U[ci,step] (as 9x9) then d U / d pulse via dL/dpulse.
        """
        # Backward: for each case, accumulate adjoint of rho_final
        # score = sum_ci [ mean_si infid_ci,si + softmax over (ci,si) ]
        # d score / d rho_final[ci,si] = - (1/(54)) * fid_vec[si]  (for the mean part)
        #                              + -softmax_weight[ci,si] * fid_vec[si]
        infids = 1.0 - np.zeros((9, 6))
        fids_mat = np.zeros((9, 6))
        for ci in range(9):
            for si in range(6):
                fids_mat[ci, si] = np.real(np.vdot(fid_vec[si], rho_final[ci, si]))
        infids = 1.0 - fids_mat
        m = np.max(infids)
        w = np.exp(soft_beta * (infids - m))
        w = w / np.sum(w)
        # d score / d fids[ci,si] = -1/54 - soft_beta * w[ci,si]
        dfid = -1.0 / 54.0 - soft_beta * w
        # d score / d rho_final[ci,si] (vector) = dfid[ci,si] * conj(fid_vec[si])  (since fid=Re[vdot(fid_vec,rho)])
        # vdot(a,b)=conj(a)·b, d Re[vdot(a,b)]/db = Re[conj(a)] as a real linear functional -> gradient vector = conj(a).conj()?? 
        # Treat complex: score real. d score = Re[ sum d(rho)^* ... ]. We'll compute dL contribution using
        # d score / d vec(rho) as a complex vector g where score += Re[ g^H vec(rho) ] => g = conj(fid_vec)*dfid
        grad_rho = np.zeros((9, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                grad_rho[ci, si] = dfid[ci, si] * np.conj(fid_vec[si])

        # Adjoint propagation backward
        # adj[ci, step] is 9-vector = d score / d vec(rho at start of step)
        # rho after step = U[step] @ rho_before
        # d score / d rho_before = U[step]^H (as acting on vec) ... but U acts on vec via matrix mult:
        # vec_after = U @ vec_before  =>  d score/d vec_before = U^H @ d score/d vec_after  (using standard inner product on vec space)
        # We'll store adj as 9-vectors.
        adj = np.zeros((9, N, 9), dtype=complex)
        for ci in range(9):
            acc = np.zeros((6, 9), dtype=complex)
            for si in range(6):
                acc[si] = grad_rho[ci, si]
            # acc[si] = d score / d vec(rho_final[ci,si])
            for step in range(N - 1, -1, -1):
                # d score / d vec(rho_before) for each si = U[step]^H @ acc[si]
                U = U_all[ci, step]
                new_acc = np.zeros((6, 9), dtype=complex)
                for si in range(6):
                    new_acc[si] = U.conj().T @ acc[si]
                # store adj at this step (d score / d rho at start of step)
                adj[ci, step] = new_acc.sum(axis=0)   # not needed per-step for dU; we need d score/d U
                acc = new_acc
        # Now compute d score / d U[ci,step] (9x9):
        # vec_after = U @ vec_before ; d score = Re[ sum_si acc_after[si]^H (U @ vec_before[si]) ]
        # => d score / d U (as matrix) : dU contribution = Re[ sum_si acc_after[si] vec_before[si]^H ]^T ... 
        # We'll compute dL/dU via: score += Re[ Tr(M^H U) ] where M = sum_si vec_before[si] acc_after[si]^H (as 3x3)
        # => d score / d U_{ij} (real) = Re[ conj(M_{ij}) ]  for the real part; but U is complex and score real,
        # the complex gradient (Wirtinger) is conj(M). We'll use d score/d U = conj(M) as the complex gradient.
        # Then d U / d pulse via dL: U = expm(dt L), dU = dt * integral expm ... approx dt * U @ dL for small dt (first order).
        # d score / d L = U^H @ M  (since d score = Re[ Tr(M^H U) ], dU = dt U dL => d score = Re[ Tr(M^H dt U dL) ] = Re[ dt Tr(dL^H U^H M) ] => d score/dL = dt * U^H M)
        # Then dL/dI, dL/dQ are simple (linear).
        grad_pulse = np.zeros((N, 2))
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                # vec_before at start of step for each si: propagate rho_init forward to start of step
                # We need vec_before[si] = (U[step-1]...U[0]) @ rho_init[si]
                # Recompute forward cumulative (cheap enough: N*N*6*9x9 ~ 80*80*6*81 ~ 3.1M mults per ci -> 28M total, ok)
                # Instead, precompute forward states F[ci, step, si] = vec at start of step
                pass
        # To avoid O(N^2) recompute, precompute forward states once
        F = np.zeros((9, N, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                vec = rho_init[si].copy()
                F[ci, 0, si] = vec
                for step in range(N - 1):
                    vec = U_all[ci, step] @ vec
                    F[ci, step + 1, si] = vec
        for ci in range(9):
            sc = case_scale[ci]
            for step in range(N):
                # M = sum_si vec_before[si] * acc_after[si]^H   (3x3)
                # acc_after[si] = adj[ci, step+1]?? We stored adj[ci,step]=d score/d rho at start of step.
                # We need d score / d rho_after(step) = sum over si of d score/d rho_final propagated back to after step.
                # Let's recompute a clean backward adjoint B[ci, step, si] = d score / d vec(rho after step)
                pass
        # Recompute backward adjoint per (ci,si) cleanly
        B = np.zeros((9, N, 6, 9), dtype=complex)
        for ci in range(9):
            for si in range(6):
                # at final: d score / d vec(rho_final[ci,si]) = grad_rho[ci,si]
                b = grad_rho[ci, si].copy()
                B[ci, N - 1, si] = b   # d score / d vec after last step
                for step in range(N - 1, 0, -1):
                    # vec_after(step) = U[step] @ vec_before(step)
                    # d score / d vec_before(step) = U[step]^H @ b
                    b = U_all[ci, step].conj().T @ b
                    B[ci, step - 1, si] = b   # d score / d vec after step-1
        # Now for each (ci,step): d score / d U[ci,step]:
        # vec_after = U @ vec_before ; d score = Re[ sum_si B_after[si]^H (U @ vec_before[si]) ]
        # = Re[ Tr( (sum_si vec_before[si] B_after[si]^H)^H U ) ]  ... define M = sum_si vec_before[si] B_after[si]^H (3x3)
        # then d score/dU (complex gradient) = conj(M)
        # d score/dL = dt * U^H M  (from dU = dt U dL first order)
        # dL/dI: L = case_static - 1j (kron(I3,Hd) - kron(Hd.T,I3)) ; Hd = 0.5 sc (I (a+adag) + i Q (a-adag))
        # dL/dI = -1j ( kron(I3, dHd/dI) - kron((dHd/dI).T, I3) )
        # dHd/dI = 0.5 sc (a + a_dag) ; dHd/dQ = 0.5 sc * i (a - a_dag)
        dHd_dI = 0.5 * sc * (a + a_dag)
        dHd_dQ = 0.5 * sc * 1j * (a - a_dag)
        dL_dI = -1j * (np.kron(I3, dHd_dI) - np.kron(dHd_dI.T, I3))
        dL_dQ = -1j * (np.kron(I3, dHd_dQ) - np.kron(dHd_dQ.T, I3))
        for step in range(N):
            M = np.zeros((3, 3), dtype=complex)
            for si in range(6):
                vb = F[ci, step, si].reshape(3, 3)
                ba = B[ci, step, si].reshape(3, 3)
                M += vb @ ba.conj().T
            dU = np.conj(M)
            dL = dt * (U_all[ci, step].conj().T @ M)
            dI = np.real(np.vdot(dL, dL_dI))   # Re[ dL^H dL_dI ]
            dQ = np.real(np.vdot(dL, dL_dQ))
            grad_pulse[step, 0] += dI
            grad_pulse[step, 1] += dQ
        return grad_pulse

    def model_objective(params, soft_beta=50.0, need_grad=False):
        if time.time() > deadline - 5:
            return (1e10, np.zeros(6)) if need_grad else 1e10
        p = build_pulse(params)
        if not check_validity(p):
            return (1e10, np.zeros(6)) if need_grad else 1e10
        score, U_all, rho_final = simulate_and_score(p, soft_beta)
        if not need_grad:
            return score
        grad_p = adjoint_grad(p, U_all, rho_final, soft_beta)
        # d score / d params via d p / d params (finite diff on the small model is cheap: 6 params)
        eps = 1e-7
        grad = np.zeros(6)
        for j in range(6):
            pp = params.copy(); pm = params.copy()
            pp[j] += eps; pm[j] -= eps
            p_plus = build_pulse(pp); p_minus = build_pulse(pm)
            sp = 1e10; sm = 1e10
            if check_validity(p_plus):
                sp, _, _ = simulate_and_score(p_plus, soft_beta)
            if check_validity(p_minus):
                sm, _, _ = simulate_and_score(p_minus, soft_beta)
            grad[j] = (sp - sm) / (2 * eps)
        return score, grad

    # ---------------- Deadline guard ----------------
    start_time = time.time()
    deadline = start_time + 300.0   # 300s of 330s kill, 30s safety

    def remaining():
        return deadline - time.time()

    # ---------------- Phase 1: coarse DRAG grid ----------------
    best_pulse = None
    best_score = np.inf
    sigma0 = 40.0 / 6.0
    for A in [0.30, 0.35, 0.40]:
        for alpha in [-0.25, -0.15, -0.08, -0.05, -0.03, 0.0]:
            for sigma in [sigma0 * 0.9, sigma0, sigma0 * 1.1]:
                if remaining() < 20:
                    break
                params = np.array([A, alpha, sigma, 0.0, 0.0, 0.0])
                p = build_pulse(params)
                if not check_validity(p):
                    continue
                s = exact_score(p)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()
                if remaining() < 20:
                    break
            if remaining() < 20:
                break
        if remaining() < 20:
            break

    if best_pulse is None:
        # fallback: simple Gaussian
        g = gauss(t_grid, sigma0)
        g = g - g[0]
        g = g * (np.pi / (np.sum(g) * dt))
        best_pulse = np.column_stack([g, np.zeros(N)])
        if not check_validity(best_pulse):
            best_pulse = best_pulse * (amplitude_bound / np.max(np.sqrt(best_pulse[:, 0] ** 2)))
        best_score = exact_score(best_pulse)

    # ---------------- Phase 2: L-BFGS-B on the 6-param DRAG model ----------------
    if remaining() > 40:
        # initialize from best_pulse by least-squares-ish: just start near a good grid point
        p_init = best_pulse
        # estimate A, alpha, sigma by matching
        A0 = float(np.max(np.abs(p_init[:, 0])))
        alpha0 = -0.10
        params0 = np.array([A0, alpha0, sigma0, 0.0, 0.0, 0.0])
        bounds = [(0.05, 0.42), (-0.5, 0.5), (0.8, 4.0), (-0.3, 0.3), (-0.2, 0.2), (-0.5, 0.5)]

        def obj(params):
            return model_objective(params, soft_beta=50.0, need_grad=False)

        def objg(params):
            return model_objective(params, soft_beta=50.0, need_grad=True)

        res = minimize(objg, params0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': 60, 'maxfev': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if remaining() > 10:
            p = build_pulse(res.x)
            if check_validity(p):
                s = exact_score(p)
                if s < best_score:
                    best_score = s
                    best_pulse = p.copy()

    # ---------------- Phase 3: free-waveform polish (few L-BFGS-B steps) ----------------
    if remaining() > 60:
        x0 = best_pulse.flatten().copy()
        bounds = [(-amplitude_bound, amplitude_bound)] * (2 * N)

        def free_obj(params):
            if time.time() > deadline - 10:
                return 1e10
            p = params.reshape(N, 2)
            if not check_validity(p):
                return 1e10
            s, U_all, rho_final = simulate_and_score(p, soft_beta=50.0)
            return s

        # coarse Nelder-Mead first (robust, no gradient)
        res_nm = minimize(free_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 40, 'maxfev': 60, 'xatol': 1e-4, 'fatol': 1e-7})
        if remaining() > 30:
            p_nm = res_nm.x.reshape(N, 2)
            if check_validity(p_nm):
                s_nm = exact_score(p_nm)
                if s_nm < best_score:
                    best_score = s_nm
                    best_pulse = p_nm.copy()

        # then L-BFGS-B with adjoint gradient on the free waveform
        if remaining() > 30:
            x0 = best_pulse.flatten().copy()

            def free_objg(params):
                if time.time() > deadline - 10:
                    return 1e10, np.zeros(2 * N)
                p = params.reshape(N, 2)
                if not check_validity(p):
                    return 1e10, np.zeros(2 * N)
                s, U_all, rho_final = simulate_and_score(p, soft_beta=50.0)
                g = adjoint_grad(p, U_all, rho_final, soft_beta)
                return s, g.flatten()

            res_lb = minimize(free_objg, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                              options={'maxiter': 25, 'maxfev': 60, 'ftol': 1e-12, 'gtol': 1e-8})
            if remaining() > 10:
                p_lb = res_lb.x.reshape(N, 2)
                if check_validity(p_lb):
                    s_lb = exact_score(p_lb)
                    if s_lb < best_score:
                        best_score = s_lb
                        best_pulse = p_lb.copy()

    # ---------------- Final validity check ----------------
    if not check_validity(best_pulse):
        g = gauss(t_grid, sigma0)
        g = g - g[0]
        g = g * (np.pi / (np.sum(g) * dt))
        best_pulse = np.column_stack([g, np.zeros(N)])
        if not check_validity(best_pulse):
            best_pulse = best_pulse * (amplitude_bound / (np.max(np.sqrt(best_pulse[:, 0] ** 2)) + 1e-15))

    return best_pulse
# EVOLVE-BLOCK-END
