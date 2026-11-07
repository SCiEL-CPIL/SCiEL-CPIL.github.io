import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import time

def _make_liouvillian_super(H, a, n_op, T1, Tphi):
    """Build the 9x9 Liouvillian superoperator."""
    dim = 3
    I3 = np.eye(3)
    
    # Coherent: -i[H, rho] => superoperator (I⊗H - H^T⊗I) * (-i)
    L_coh = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative
    a_T1 = a / np.sqrt(T1)
    n_Tphi = n_op * np.sqrt(2.0 / Tphi)
    
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [a_T1, n_Tphi]:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        # Lk rho Lk† => (Lk†)^T ⊗ Lk
        L_diss += np.kron(Lk_dag.T, Lk)
        # -0.5{Lk†Lk, rho} => -0.5[(I⊗Lk†Lk) + (Lk†Lk)^T⊗I]
        L_diss -= 0.5 * (np.kron(I3, Lk_dag_Lk) + np.kron(Lk_dag_Lk.T, I3))
    
    return L_coh + L_diss

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    n_steps = 80
    t = np.arange(n_steps) * dt
    
    # Precompute a, n_op
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n_op = a_dag @ a  # diag(0,1,2)
    
    X_full = np.zeros((3, 3), dtype=complex)
    X_full[0, 1] = 1.0
    X_full[1, 0] = 1.0
    X_full[2, 2] = 1.0
    
    test_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    test_rhos = [np.outer(psi, psi.conj()) for psi in test_states]
    target_rhos = [np.outer(X_full @ psi, (X_full @ psi).conj()) for psi in test_states]
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    def simulate(I, Q, det, sc):
        """Simulate and return final rho."""
        rho_vec = np.zeros(9, dtype=complex)
        rho_vec[0] = 1.0  # |0><0|
        
        for i in range(n_steps):
            H = (det * n_op + 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
                 + 0.5 * sc * (I[i] * (a + a_dag) + 1j * Q[i] * (a - a_dag)))
            L = _make_liouvillian_super(H, a, n_op, T1, Tphi)
            rho_vec = expm(L * dt) @ rho_vec
        
        rho = rho_vec.reshape(3, 3)
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho /= tr
        return rho
    
    def compute_score(I, Q):
        infs = []
        for det in detunings:
            for sc in scales:
                rho_f = simulate(I, Q, det, sc)
                case_infs = []
                for tr in target_rhos:
                    fid = np.real(np.trace(tr @ rho_f))
                    fid = max(0.0, min(1.0, fid))
                    case_infs.append(1.0 - fid)
                infs.append(np.mean(case_infs))
        return np.mean(infs) + np.max(infs)
    
    def make_pulse(A, alpha, sigma):
        t_center = n_steps * dt / 2.0
        I = A * np.exp(-(t - t_center)**2 / (2 * sigma**2))
        dI_dt = -A * (t - t_center) / sigma**2 * np.exp(-(t - t_center)**2 / (2 * sigma**2))
        Q = -alpha * dI_dt
        # Normalize area to pi
        area = np.sum(I) * dt
        if abs(area) > 1e-10:
            I = I * (np.pi / area)
            Q = Q * (np.pi / area)
        return I, Q
    
    def objective(params):
        A, alpha, sigma = params
        I, Q = make_pulse(A, alpha, sigma)
        # Constraint check
        for i in range(n_steps):
            if I[i]**2 + Q[i]**2 > 0.16 + 1e-12:
                return 1e10
        prev = 0.0j
        for i in range(n_steps):
            curr = I[i] + 1j * Q[i]
            if np.abs(curr - prev) > 0.12 * dt + 1e-12:
                return 1e10
            prev = curr
        if np.abs(prev) > 0.12 * dt + 1e-12:
            return 1e10
        return compute_score(I, Q)
    
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin
    
    x0 = [0.15, 0.05, 6.67]
    best_x = x0.copy()
    best_score = objective(x0)
    
    # Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-10},
                      bounds=[(0.01, 0.4), (-0.5, 0.5), (2.0, 20.0)])
        if time.time() < deadline and res.fun < best_score:
            best_score = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Powell
    if time.time() < deadline - 60:
        try:
            res = minimize(objective, best_x, method='Powell',
                          options={'maxiter': 300, 'xtol': 1e-7, 'ftol': 1e-10},
                          bounds=[(0.01, 0.4), (-0.5, 0.5), (2.0, 20.0)])
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
        except Exception:
            pass
    
    A, alpha, sigma = best_x
    I, Q = make_pulse(A, alpha, sigma)
    
    result = np.column_stack([I, Q])
    
    # Final constraint enforcement
    for i in range(n_steps):
        amp = np.sqrt(result[i, 0]**2 + result[i, 1]**2)
        if amp > 0.4:
            s = 0.4 / amp
            result[i, 0] *= s
            result[i, 1] *= s
    
    prev = 0.0j
    for i in range(n_steps):
        curr = result[i, 0] + 1j * result[i, 1]
        delta = curr - prev
        if np.abs(delta) > 0.12 * dt:
            curr = prev + delta * (0.12 * dt / np.abs(delta))
            result[i, 0] = curr.real
            result[i, 1] = curr.imag
        prev = curr
    if np.abs(prev) > 0.12 * dt:
        s = 0.12 * dt / np.abs(prev)
        result[-1, 0] *= s
        result[-1, 1] *= s
    
    return result
