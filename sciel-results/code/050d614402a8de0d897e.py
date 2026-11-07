import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time


def pulse(target_gate, hardware_model):
    start = time.time()
    deadline = start + 322.0

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

    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)

    d = 3
    eye_d = np.eye(d)

    # Incoherent Liouvillian (constant)
    incoh_L = np.zeros((d * d, d * d), dtype=complex)
    for Lk in [L1, L2]:
        Lkd = Lk.conj().T
        incoh_L += np.kron(Lk.conj(), Lk)
        LkdLk = Lkd @ Lk
        incoh_L += -0.5 * np.kron(eye_d, LkdLk)
        incoh_L += -0.5 * np.kron(LkdLk.T, eye_d)

    # Coherent superoperator linear parts (precomputed kron differences)
    coh_base = -1j * (np.kron(H0_nom, eye_d) - np.kron(eye_d, H0_nom.T))
    coh_delta = -1j * (np.kron(n_op, eye_d) - np.kron(eye_d, n_op.T))
    coh_I = -1j * (np.kron(driveI, eye_d) - np.kron(eye_d, driveI.T))
    coh_Q = -1j * (np.kron(driveQ, eye_d) - np.kron(eye_d, driveQ.T))

    # Per-case constant superoperator and drive coefficients
    L_bases = np.zeros((9, d * d, d * d), dtype=complex)
    L_I = np.zeros((9, d * d, d * d), dtype=complex)
    L_Q = np.zeros((9, d * d, d * d), dtype=complex)
    ci = 0
    for d_val in deltas:
        for sc_val in scales:
            L_bases[ci] = coh_base + d_val * coh_delta + incoh_L
            L_I[ci] = sc_val * coh_I
            L_Q[ci] = sc_val * coh_Q
            ci += 1

    # States / targets
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

    # w_s = vec(|t_s><t_s|)  (column-major).  f_s = Re( w_s^T U w_s )
    W = np.array([np.outer(targets[s], targets[s].conj()).ravel() for s in range(6)])
    W2 = W.reshape(6, 1, 9)  # (6,1,9)

    def check_constraints(I_Q):
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        if np.max(np.sqrt(np.sum(diffs ** 2, axis=1))) / dt > slew_bound + 1e-9:
            return False
        return True

    # Fast per-case fidelity evaluation.
    def case_fids(I_Q, ci):
        Lb = L_bases[ci]
        LI = L_I[ci]
        LQ = L_Q[ci]
        f = np.zeros(6)
        for i in range(N):
            L = Lb + I_Q[i, 0] * LI + I_Q[i, 1] * LQ
            U = expm(L * dt)
            f += np.real((W2 @ U @ W).ravel())
        return f / float(N)

    def score_case(I_Q, ci):
        infs = 1.0 - case_fids(I_Q, ci)
        return infs.mean() + infs.max(), infs

    def score(I_Q):
        if not check_constraints(I_Q):
            return 1e6
        total = 0.0
        for ci in range(9):
            total += score_case(I_Q, ci)[0]
        return total / 9.0

    def score_max(I_Q):
        """Soft-max / worst-case objective (focuses on the max term)."""
        if not check_constraints(I_Q):
            return 1e6
        vals = np.array([score_case(I_Q, ci)[0] for ci in range(9)])
        return vals.mean() + vals.max()

    # ---- pulse generators ----
    def gen_drag(p):
        A, sI, eta, sQ, shQ, c, q0 = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        Q = eta * dI + c * np.exp(-(t - tc - shQ) ** 2 / (2 * sQ ** 2)) + q0
        return np.column_stack((I, Q))

    def gen_spectral(p):
        # p = [A, sI, eta, A1, s1, c1, A2, s2, c2]
        A, sI, eta, A1, s1, c1, A2, s2, c2 = p
        t = np.linspace(0, 40, N)
        tc = 20.0
        I = A * np.exp(-(t - tc) ** 2 / (2 * sI ** 2))
        dI = -(t - tc) / sI ** 2 * I
        Q = (eta * dI
             + A1 * np.exp(-(t - tc - c1) ** 2 / (2 * s1 ** 2))
             + A2 * np.exp(-(t - tc - c2) ** 2 / (2 * s2 ** 2)))
        return np.column_stack((I, Q))

    def seed_pulse():
        s = 40.0 / 6.0
        t = np.linspace(0, 40, N)
        I = (np.sqrt(np.pi / 2) / s) * np.exp(-(t - 20.0) ** 2 / (2 * s ** 2))
        return np.column_stack((I, np.zeros(N)))

    best = seed_pulse()
    best_s = score(best)

    # ---- 1. wide DRAG grid (mean+max) ----
    for A in [0.18, 0.22, 0.26, 0.30, 0.34, 0.37, 0.39]:
        for sI in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            for eta in [-4.0, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 4.0]:
                if time.time() > deadline - 150:
                    break
                I_Q = gen_drag([A, sI, eta, 5.0, 0.0, 0.0, 0.0])
                if not check_constraints(I_Q):
                    continue
                s = score(I_Q)
                if s < best_s:
                    best_s, best = s, I_Q

    # ---- 2. spectral-nulling grid (mean+max) ----
    for A in [0.26, 0.30, 0.34]:
        for sI in [5.0, 6.0, 7.0]:
            for eta in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                for A1 in [-0.10, -0.06, 0.0, 0.06, 0.10]:
                    if time.time() > deadline - 120:
                        break
                    I_Q = gen_spectral([A, sI, eta, A1, 3.0, 5.0, -A1, 3.0, -5.0])
                    if not check_constraints(I_Q):
                        continue
                    s = score(I_Q)
                    if s < best_s:
                        best_s, best = s, I_Q

    # ---- 3. Nelder-Mead refine, both objectives ----
    def opt_drag(p):
        if time.time() > deadline - 80:
            return 1e6
        I_Q = gen_drag(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    def opt_drag_max(p):
        if time.time() > deadline - 80:
            return 1e6
        I_Q = gen_drag(p)
        if not check_constraints(I_Q):
            return 1e6
        return score_max(I_Q)

    def opt_spectral(p):
        if time.time() > deadline - 80:
            return 1e6
        I_Q = gen_spectral(p)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    drag_inits = [
        [0.30, 5.0, -1.0, 5.0, 0.0, 0.0, 0.0],
        [0.30, 6.0, -2.0, 5.0, 0.0, 0.0, 0.0],
        [0.34, 4.0, 1.0, 6.0, 0.0, 0.0, 0.0],
        [0.37, 5.5, -1.5, 5.0, 0.0, 0.0, 0.0],
        [0.30, 7.0, -0.5, 5.0, 0.0, 0.0, 0.0],
    ]
    for init in drag_inits:
        if time.time() > deadline - 70:
            break
        for opt in (opt_drag, opt_drag_max):
            if time.time() > deadline - 70:
                break
            try:
                r = minimize(opt, init, method='Nelder-Mead',
                             options={'maxiter': 250, 'xatol': 0.01, 'fatol': 1e-8})
                if r.fun < 1e5:
                    I_Q = gen_drag(r.x)
                    if check_constraints(I_Q):
                        s = score(I_Q)
                        if s < best_s:
                            best_s, best = s, I_Q
            except Exception:
                pass

    spec_inits = [
        [0.30, 5.0, -1.0, 0.05, 3.0, 5.0, -0.05, 3.0, -5.0],
        [0.30, 6.0, -2.0, 0.10, 4.0, 6.0, -0.10, 4.0, -6.0],
        [0.34, 5.0, -1.5, 0.08, 3.5, 4.0, -0.08, 3.5, -4.0],
    ]
    for init in spec_inits:
        if time.time() > deadline - 55:
            break
        try:
            r = minimize(opt_spectral, init, method='Nelder-Mead',
                         options={'maxiter': 250, 'xatol': 0.01, 'fatol': 1e-8})
            if r.fun < 1e5:
                I_Q = gen_spectral(r.x)
                if check_constraints(I_Q):
                    s = score(I_Q)
                    if s < best_s:
                        best_s, best = s, I_Q
        except Exception:
            pass

    # ---- 4. direct 80-step polish (L-BFGS-B), more rounds, both objectives ----
    def opt_direct(x):
        if time.time() > deadline - 12:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    def opt_direct_max(x):
        if time.time() > deadline - 12:
            return 1e6
        I_Q = x.reshape(N, 2)
        if not check_constraints(I_Q):
            return 1e6
        return score_max(I_Q)

    best_x = best.reshape(-1)
    for k in range(12):
        if time.time() > deadline - 12:
            break
        improved = False
        for opt in (opt_direct, opt_direct_max):
            if time.time() > deadline - 12:
                break
            try:
                r = minimize(opt, best_x, method='L-BFGS-B',
                             options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-7,
                                      'maxfun': 1200, 'maxls': 40})
                I_Q = r.x.reshape(N, 2)
                if check_constraints(I_Q):
                    s = score(I_Q)
                    if s < best_s:
                        best_s, best, best_x = s, I_Q, r.x
                        improved = True
            except Exception:
                pass
        if not improved and k >= 4:
            break

    # ---- final feasibility cleanup ----
    if not check_constraints(best):
        I_Q = best.copy()
        amps = np.sqrt(I_Q[:, 0] ** 2 + I_Q[:, 1] ** 2)
        mask = amps > amp_bound
        if np.any(mask):
            sc = amp_bound / amps[mask]
            I_Q[mask, 0] *= sc
            I_Q[mask, 1] *= sc
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        max_step = slew_bound * dt
        for i in range(len(p) - 1):
            diff = p[i + 1] - p[i]
            nd = np.sqrt(np.sum(diff ** 2))
            if nd > max_step:
                p[i + 1] = p[i] + diff * (max_step / nd)
        I_Q = p[1:-1]
        best = I_Q if check_constraints(I_Q) else seed_pulse()

    return best
