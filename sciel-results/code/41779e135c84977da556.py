"""Robust X_pi control pulse for a three-level transmon using DRAG optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    # Hardware parameters
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    detuning_bounds = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
    scale_bounds = (0.97, 1.03)
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    N = 80
    
    # Build the pulse
    # DRAG-style Gaussian: I(t) = A * exp(-(t-t0)^2/(2*sigma^2))
    # Q(t) = alpha * dI/dt
    # sigma = 40/6 ns as specified
    sigma = 40.0 / 6.0
    t_centers = np.arange(N) * dt + dt / 2
    t0 = 20.0
    
    # Area of I pulse should be pi
    A_I = np.sqrt(2 * np.pi) / sigma
    
    # Initial DRAG parameter estimate
    # Optimal alpha ~ 1/(anharmonicity * sigma^2) for DRAG
    alpha = 1.0 / (abs(anharmonicity) * sigma**2)
    
    def make_pulse(alpha):
        I = A_I * np.exp(-(t_centers - t0)**2 / (2 * sigma**2))
        dI_dt = -A_I * (t_centers - t0) / sigma**2 * np.exp(-(t_centers - t0)**2 / (2 * sigma**2))
        Q = alpha * dI_dt
        return I, Q
    
    def check_constraints(I, Q):
        """Check amplitude and slew constraints."""
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > amp_bound + 1e-12:
            return False, "amp"
        
        # Slew: from zero to step 1
        slew0 = np.sqrt(I[0]**2 + Q[0]**2) / dt
        if slew0 > slew_bound + 1e-12:
            return False, "slew0"
        
        # Between consecutive steps
        for i in range(N - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            slew = np.sqrt(dI**2 + dQ**2) / dt
            if slew > slew_bound + 1e-12:
                return False, "slew"
        
        # From last step to zero
        slewn = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
        if slewn > slew_bound + 1e-12:
            return False, "slewn"
        
        return True, "ok"
    
    def enforce_constraints(I, Q):
        """Scale pulse to satisfy all constraints."""
        # Check amplitude
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
        if max_amp > amp_bound:
            scale = amp_bound / max_amp
            I = I * scale
            Q = Q * scale
        
        # Check slew
        def max_slew(I, Q):
            ms = np.sqrt(I[0]**2 + Q[0]**2) / dt
            for i in range(N - 1):
                dI = I[i+1] - I[i]
                dQ = Q[i+1] - Q[i]
                s = np.sqrt(dI**2 + dQ**2) / dt
                ms = max(ms, s)
            ms = max(ms, np.sqrt(I[-1]**2 + Q[-1]**2) / dt)
            return ms
        
        ms = max_slew(I, Q)
        if ms > slew_bound:
            scale = slew_bound / ms
            I = I * scale
            Q = Q * scale
        
        return I, Q
    
    def build_hamiltonian(detuning, scale, I, Q):
        """Build the 3x3 Hamiltonian matrix."""
        # a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
        a = np.array([[0, np.sqrt(1), 0],
                      [0, 0, np.sqrt(2)],
                      [0, 0, 0]], dtype=complex)
        a_dag = a.conj().T
        n = a_dag @ a  # diag(0, 1, 2)
        
        H = np.zeros((3, 3), dtype=complex)
        # Detuning term: detuning * n
        H += detuning * n
        # Anharmonicity term: 0.5 * anharmonicity * n * (n-1)
        H += 0.5 * anharmonicity * (n @ (n - np.eye(3)))
        # Drive term: 0.5 * scale * (I * (a + a_dag) + i*Q * (a - a_dag))
        H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
        
        return H
    
    def build_liouvillian(H):
        """Build the Liouvillian superoperator for Lindblad evolution."""
        # Gamma_1 = a / sqrt(T1)
        a = np.array([[0, np.sqrt(1), 0],
                      [0, 0, np.sqrt(2)],
                      [0, 0, 0]], dtype=complex)
        Gamma_1 = a / np.sqrt(T1)
        
        # Gamma_phi = n * sqrt(2/Tphi)
        n = np.diag([0, 1, 2]).astype(complex)
        Gamma_phi = n * np.sqrt(2.0 / Tphi)
        
        # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
        # We'll implement this as a function that takes rho and returns L(rho)
        def liouvillian(rho):
            result = -1j * (H @ rho - rho @ H)
            for L in [Gamma_1, Gamma_phi]:
                L_dag = L.conj().T
                result += L @ rho @ L_dag - 0.5 * (L_dag @ L @ rho + rho @ L_dag @ L)
            return result
        
        return liouvillian
    
    def simulate_pulse(I, Q, detuning=0.0, scale=1.0):
        """Simulate the pulse and compute fidelity to X gate."""
        # Initial state: |0> = [1, 0, 0]
        rho = np.zeros((3, 3), dtype=complex)
        rho[0, 0] = 1.0
        
        # Target: X gate on |0>, |1> subspace
        # X|0> = |1>, X|1> = |0>
        
        # Evolve step by step
        for i in range(N):
            H_i = build_hamiltonian(detuning, scale, I[i], Q[i])
            L_i = build_liouvillian(H_i)
            
            # Evolve for dt using matrix exponential approximation
            # For small dt, use first-order: rho_new = rho + dt * L(rho)
            # But for better accuracy, use expm of the Liouvillian superoperator
            # Let's use a simple RK4 for the density matrix
            
            def d_rho(rho):
                return L_i(rho)
            
            # RK4
            k1 = d_rho(rho)
            k2 = d_rho(rho + 0.5 * dt * k1)
            k3 = d_rho(rho + 0.5 * dt * k2)
            k4 = d_rho(rho + dt * k3)
            rho = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Compute fidelity for each test state
        # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
        test_states = [
            np.array([1, 0, 0], dtype=complex),  # |0>
            np.array([0, 1, 0], dtype=complex),  # |1>
            np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
            np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
            np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
            np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
        ]
        
        # X gate action
        X = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 0]], dtype=complex)
        
        fidelities = []
        for psi in test_states:
            # Evolve psi
            psi_vec = psi.copy()
            rho_psi = np.outer(psi_vec, psi_vec.conj())
            
            for i in range(N):
                H_i = build_hamiltonian(detuning, scale, I[i], Q[i])
                L_i = build_liouvillian(H_i)
                
                def d_rho(rho):
                    return L_i(rho)
                
                k1 = d_rho(rho_psi)
                k2 = d_rho(rho_psi + 0.5 * dt * k1)
                k3 = d_rho(rho_psi + 0.5 * dt * k2)
                k4 = d_rho(rho_psi + dt * k3)
                rho_psi = rho_psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Target state
            psi_target = X @ psi
            rho_target = np.outer(psi_target, psi_target.conj())
            
            # Fidelity
            fid = np.real(np.trace(rho_psi @ rho_target))
            fid = np.clip(fid, 0, 1)
            fidelities.append(fid)
        
        return np.mean(fidelities)
    
    # Optimize alpha for DRAG
    def score_for_alpha(alpha):
        I, Q = make_pulse(alpha)
        I, Q = enforce_constraints(I, Q)
        
        # Evaluate over the 9 cases
        detunings = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
        scales = [0.97, 1.0, 1.03]
        
        infidelities = []
        for d in detunings:
            for s in scales:
                fid = simulate_pulse(I, Q, d, s)
                infidelities.append(1 - fid)
        
        return np.mean(infidelities) + np.max(infidelities)
    
    # Grid search for alpha
    alpha_grid = np.linspace(0.0, 0.05, 11)
    best_alpha = alpha_grid[0]
    best_score = np.inf
    
    for a in alpha_grid:
        if _time_remaining() < 60:
            break
        s = score_for_alpha(a)
        if s < best_score:
            best_score = s
            best_alpha = a
    
    # Fine-tune around best alpha
    if _time_remaining() > 30:
        alpha_fine = np.linspace(best_alpha - 0.005, best_alpha + 0.005, 11)
        for a in alpha_fine:
            if _time_remaining() < 30:
                break
            s = score_for_alpha(a)
            if s < best_score:
                best_score = s
                best_alpha = a
    
    # Final pulse
    I, Q = make_pulse(best_alpha)
    I, Q = enforce_constraints(I, Q)
    
    return np.column_stack([I, Q])
# EVOLVE-BLOCK-END
