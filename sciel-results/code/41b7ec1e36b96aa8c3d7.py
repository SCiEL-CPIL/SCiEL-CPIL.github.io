"""Pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _build_operators():
    """Build 3-level transmon operators."""
    # a = diag(sqrt([1,2]), 1) - this means a = |1><0|*1 + |2><1|*sqrt(2)
    # In the basis |0>, |1>, |2>:
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0      # <0|a|1> = 1
    a[1, 2] = math.sqrt(2.0)  # <1|a|2> = sqrt(2)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _hamiltonian(I, Q, detuning, scale, a, adag, n, anharmonicity):
    """Build Hamiltonian for given I, Q, detuning, scale."""
    H = detuning * n + 0.5 * anharmonicity * n @ (n - np.eye(3))
    H += 0.5 * scale * (I * (a + adag) + 1j * Q * (a - adag))
    return H

def _liouvillian_superop(H, a, adag, n, T1, Tphi):
    """Build the Liouvillian superoperator in vectorized form."""
    dim = 3
    # H is 3x3, superoperator is 9x9
    # L(rho) = -i/2 [H, rho] + sum_k (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
    # In vectorized form: vec(L(rho)) = L_super @ vec(rho)
    # -i/2 [H, rho] -> -i/2 (H⊗I - I⊗H^T) in vec convention
    # Actually: vec(AB) = (B^T ⊗ A) vec(C)
    # vec(H rho) = (I ⊗ H) vec(rho)
    # vec(rho H) = (H^T ⊗ I) vec(rho)
    # So vec([H,rho]) = (I⊗H - H^T⊗I) vec(rho)
    
    I3 = np.eye(3)
    H_comm = -0.5j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Lindblad terms
    L1 = a / math.sqrt(T1)
    L2 = n * math.sqrt(2.0 / Tphi)
    
    def lindblad_super(L):
        Ldag = L.conj().T
        LdagL = Ldag @ L
        # vec(L rho L^dag) = (L^T ⊗ L) vec(rho)
        # vec(L^dag L rho) = (I ⊗ L^dag L) vec(rho)
        # vec(rho L^dag L) = ((L^dag L)^T ⊗ I) vec(rho)
        term = np.kron(L.T, L) - 0.5 * np.kron(I3, LdagL) - 0.5 * np.kron(LdagL.T, I3)
        return term
    
    L_super = H_comm + lindblad_super(L1) + lindblad_super(L2)
    return L_super

def _simulate_pulse(pulse_IQ, detuning, scale, a, adag, n, anharmonicity, T1, Tphi, dt=0.5):
    """Simulate the pulse and return the final density matrix for |0> initial state."""
    n_steps = len(pulse_IQ)
    dim = 3
    I3 = np.eye(dim)
    
    # Initial state: |0><0|
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(n_steps):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        H = _hamiltonian(I_val, Q_val, detuning, scale, a, adag, n, anharmonicity)
        L_super = _liouvillian_superop(H, a, adag, n, T1, Tphi)
        
        # Exponentiate: expm(L_super * dt)
        from scipy.linalg import expm
        U = expm(L_super * dt)
        rho_vec = np.array(rho, dtype=complex).flatten()
        rho_vec = U @ rho_vec
        rho = rho_vec.reshape(dim, dim)
    
    return rho

def _fidelity(rho, target_state):
    """Compute fidelity F = <psi|rho|psi> for pure state target."""
    return np.real(np.vdot(target_state, rho @ target_state))

def _compute_score(pulse_IQ, a, adag, n, anharmonicity, T1, Tphi):
    """Compute the scoring metric: mean + max infidelity over 9 cases."""
    detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Target states for X_pi gate: X|0> = |1>, X|1> = |0>
    # So X_pi = -iX (or equivalently, rotation by pi around X)
    # The unitary should be X (up to global phase)
    # X = [[0,1],[1,0]] in the {|0>,|1>} subspace
    
    # Test states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    sqrt2 = math.sqrt(2.0)
    test_states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / sqrt2,
        np.array([1, -1, 0], dtype=complex) / sqrt2,
        np.array([1, 1j, 0], dtype=complex) / sqrt2,
        np.array([1, -1j, 0], dtype=complex) / sqrt2,
    ]
    
    # Target: X gate maps |0>->|1>, |1>->|0>
    # So for each test state, the target output is X @ test_state
    # X = [[0,1,0],[1,0,0],[0,0,0]] (in 3-level basis, |2> stays |2> but we don't populate it)
    X_gate = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)  # identity on |2>
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            for psi in test_states:
                rho = _simulate_pulse(pulse_IQ, det, sc, a, adag, n, anharmonicity, T1, Tphi)
                psi_target = X_gate @ psi
                # Fidelity: |<psi_target|rho|psi>|^2 for pure target
                # But we need mean fidelity to X over the test states
                # infidelity = 1 - mean fidelity
                # For each state: fidelity = |<psi_target|rho|psi>|^2
                val = np.vdot(psi_target, rho @ psi)
                fid = np.abs(val)**2
                infid = 1.0 - fid
                infidelities.append(infid)
    
    infidelities = np.array(infid)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def _seed_pulse():
    """DRAG-style Gaussian seed pulse."""
    n_steps = 80
    dt = 0.5
    T = 40.0
    t = np.arange(n_steps) * dt + dt/2  # midpoints
    
    # Gaussian: sigma = 40/6 ns
    sigma = 40.0 / 6.0
    # Area should be pi (for X_pi gate, the integral of amplitude dt = pi)
    # Gaussian area = amplitude * sigma * sqrt(2*pi)
    # So amplitude = pi / (sigma * sqrt(2*pi))
    amp = np.pi / (sigma * math.sqrt(2 * math.pi))
    
    # Offset to 0 at edges: subtract edge value
    g = amp * np.exp(-0.5 * (t - T/2)**2 / sigma**2)
    edge_val = g[0]
    g -= edge_val
    # Ensure non-negative and within bounds
    g = np.maximum(g, 0)
    g = np.minimum(g, 0.4)
    
    pulse_IQ = np.zeros((n_steps, 2))
    pulse_IQ[:, 0] = g
    return pulse_IQ

def _check_constraints(pulse_IQ, dt=0.5):
    """Check amplitude and slew rate constraints."""
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.max(amps) > 0.4 + 1e-10:
        return False
    # Slew rate: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero to step 1:
    d0 = np.linalg.norm(pulse_IQ[0]) / dt
    if d0 > 0.12 + 1e-10:
        return False
    # From last step to zero:
    dlast = np.linalg.norm(pulse_IQ[-1]) / dt
    if dlast > 0.12 + 1e-10:
        return False
    # Between steps:
    for i in range(len(pulse_IQ) - 1):
        d = np.linalg.norm(pulse_IQ[i+1] - pulse_IQ[i]) / dt
        if d > 0.12 + 1e-10:
            return False
    return True

def _project_to_constraints(pulse_IQ, dt=0.5):
    """Project pulse onto feasible set."""
    pulse_IQ = pulse_IQ.copy()
    # Clip amplitude
    for i in range(len(pulse_IQ)):
        amp = np.linalg.norm(pulse_IQ[i])
        if amp > 0.4:
            pulse_IQ[i] *= 0.4 / amp
    # Enforce slew rate: iterative projection
    for _ in range(10):
        # Forward pass
        prev = np.zeros(2)
        for i in range(len(pulse_IQ)):
            diff = pulse_IQ[i] - prev
            d = np.linalg.norm(diff)
            if d > 0.12 * dt:
                pulse_IQ[i] = prev + diff * (0.12 * dt / d)
            prev = pulse_IQ[i]
        # Backward pass
        prev = np.zeros(2)
        for i in range(len(pulse_IQ) - 1, -1, -1):
            diff = pulse_IQ[i] - prev
            d = np.linalg.norm(diff)
            if d > 0.12 * dt:
                pulse_IQ[i] = prev + diff * (0.12 * dt / d)
            prev = pulse_IQ[i]
        # Re-clip amplitude
        for i in range(len(pulse_IQ)):
            amp = np.linalg.norm(pulse_IQ[i])
            if amp > 0.4:
                pulse_IQ[i] *= 0.4 / amp
    return pulse_IQ

def pulse(target_gate, hardware_model):
    """Returns an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    dt = 0.5
    n_steps = 80
    anharmonicity = -2 * np.pi * 0.25
    T1 = 25000.0
    Tphi = 40000.0
    
    a, adag, n = _build_operators()
    
    # Start with seed pulse
    pulse_IQ = _seed_pulse()
    pulse_IQ = _project_to_constraints(pulse_IQ, dt)
    
    best_pulse = pulse_IQ.copy()
    best_score = _compute_score(pulse_IQ, a, adag, n, anharmonicity, T1, Tphi)
    
    # Simple gradient-free optimization: try small perturbations
    np.random.seed(42)
    step_size = 0.05
    n_attempts = 0
    
    while time.time() < deadline:
        # Try random perturbations
        for _ in range(50):
            if time.time() > deadline:
                break
            n_attempts += 1
            perturbation = np.random.randn(n_steps, 2) * step_size
            candidate = pulse_IQ + perturbation
            candidate = _project_to_constraints(candidate, dt)
            
            # Quick check: only evaluate a subset for speed
            # Use just the nominal case for quick evaluation
            rho_nom = _simulate_pulse(candidate, 0, 1.0, a, adag, n, anharmonicity, T1, Tphi)
            psi0 = np.array([1, 0, 0], dtype=complex)
            psi1 = np.array([0, 1, 0], dtype=complex)
            X_gate = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex)
            
            fid0 = np.abs(np.vdot(X_gate @ psi0, rho_nom @ psi0))**2
            fid1 = np.abs(np.vdot(X_gate @ psi1, rho_nom @ psi1))**2
            quick_infid = 1.0 - 0.5*(fid0 + fid1)
            
            if quick_infid < best_score * 0.5:  # Likely improvement
                full_score = _compute_score(candidate, a, adag, n, anharmonicity, T1, Tphi)
                if full_score < best_score:
                    best_score = full_score
                    best_pulse = candidate.copy()
                    pulse_IQ = candidate.copy()
                    step_size *= 0.8
                else:
                    step_size *= 1.1
            else:
                step_size *= 1.05
        
        step_size = min(step_size, 0.1)
    
    return best_pulse
# EVOLVE-BLOCK-END
