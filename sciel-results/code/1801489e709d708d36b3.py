"""Optimize 3-layer Ry/CZ circuit for open transverse-field Ising chains."""
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _ry_matrix(theta):
    """Ry rotation matrix for angle theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])

def _cz_matrix():
    """CZ gate matrix (4x4)."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)

def _simulate_circuit(angles, n):
    """
    Simulate the circuit: start |0>^n, apply layers of Ry then CZ.
    angles: (layers, n) array
    Returns: statevector (2^n,)
    """
    layers = angles.shape[0]
    # Start state |0>^n
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0

    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            ry = _ry_matrix(angles[layer, i])
            # Apply to qubit i
            psi = _apply_gate_to_qubit(psi, ry, i, n)
        
        # Apply CZ to adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                cz = _cz_matrix()
                psi = _apply_gate_to_qubit(psi, cz, i, n)

    return psi

def _apply_gate_to_qubit(psi, gate, qubit, n):
    """Apply a 2-qubit gate to qubits (qubit, qubit+1) or 1-qubit gate to qubit."""
    gate_dim = gate.shape[0]
    if gate_dim == 2:
        # 1-qubit gate
        return _apply_1q_gate(psi, gate, qubit, n)
    else:
        # 2-qubit gate on (qubit, qubit+1)
        return _apply_2q_gate(psi, gate, qubit, n)

def _apply_1q_gate(psi, gate, qubit, n):
    """Apply 1-qubit gate to qubit."""
    dim = 2**n
    result = np.zeros_like(psi)
    for idx in range(dim):
        bit = (idx >> qubit) & 1
        # The two basis states that differ only in qubit
        idx0 = idx & ~(1 << qubit)
        idx1 = idx0 | (1 << qubit)
        if bit == 0:
            result[idx0] = gate[0, 0] * psi[idx0] + gate[0, 1] * psi[idx1]
            result[idx1] = gate[1, 0] * psi[idx0] + gate[1, 1] * psi[idx1]
        # else: already handled when bit==0 for the pair
    return result

def _apply_2q_gate(psi, gate, qubit, n):
    """Apply 2-qubit gate to qubits (qubit, qubit+1)."""
    dim = 2**n
    result = np.zeros_like(psi)
    q1 = qubit
    q2 = qubit + 1
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        # The four basis states for this pair
        mask = (1 << q1) | (1 << q2)
        base = idx & ~mask
        states = [
            base,
            base | (1 << q1),
            base | (1 << q2),
            base | (1 << q1) | (1 << q2)
        ]
        # Map to gate indices: |b1 b2> -> b1*2 + b2
        gate_idx = b1 * 2 + b2
        for new_b1 in range(2):
            for new_b2 in range(2):
                new_idx = base | (new_b1 << q1) | (new_b2 << q2)
                new_gate_idx = new_b1 * 2 + new_b2
                result[new_idx] += gate[new_gate_idx, gate_idx] * psi[idx]
    return result

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open chain Ising model."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bi = (idx >> i) & 1
            bj = (idx >> (i + 1)) & 1
            z_i = 1 if bi == 0 else -1
            z_j = 1 if bj == 0 else -1
            H[idx, idx] += -coupling * z_i * z_j
    
    # X_i terms
    for i in range(n):
        for idx in range(dim):
            bi = (idx >> i) & 1
            flip_idx = idx ^ (1 << i)
            H[flip_idx, idx] += -field
    
    return H

def _exact_ground_state(n, coupling, field):
    """Compute exact ground state and energy."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_energy = eigvals[0].real
    ground_state = eigvecs[:, 0]
    return ground_energy, ground_state

def _circuit_energy(psi, n, coupling, field):
    """Compute energy <psi|H|psi>."""
    H = _build_hamiltonian(n, coupling, field)
    return np.vdot(psi, H @ psi).real

def _energy_gap_objective(params, n, coupling, field, ground_energy):
    """Objective: mean energy gap per spin."""
    angles = params.reshape(3, n)
    psi = _simulate_circuit(angles, n)
    energy = _circuit_energy(psi, n, coupling, field)
    return (energy - ground_energy) / n

def _mean_field_init(n, coupling, field):
    """
    Variational mean-field / product state initialization.
    For a product state |theta_0>...|theta_{n-1}> where each qubit is
    cos(theta_i/2)|0> + sin(theta_i/2)|1>, we can optimize each theta_i
    independently using the effective field approximation.
    
    For the Ising model, the optimal single-qubit angle for qubit i is
    approximately determined by the transverse field and the effective
    longitudinal field from neighbors.
    
    Simple heuristic: start with theta = arctan(field / effective_z)
    where effective_z accounts for coupling.
    """
    angles = np.zeros((3, n))
    
    # For each qubit, estimate the effective field
    # In a product state, the mean <Z_i> depends on theta_i
    # The energy contribution from Z_i Z_{i+1} is -coupling * <Z_i><Z_{i+1}>
    # The energy contribution from X_i is -field * <X_i>
    
    # For a single qubit in state cos(t/2)|0> + sin(t/2)|1>:
    # <Z> = cos(t), <X> = sin(t)
    # Energy per qubit (approximate): -field*sin(t_i) - coupling*(<Z_i><Z_{i-1}> + <Z_i><Z_{i+1}>)/2
    
    # Optimal t_i satisfies: field*cos(t_i) - coupling*(<Z_{i-1}> + <Z_{i+1}>)/2 * sin(t_i) = 0
    # => tan(t_i) = field / (coupling*(<Z_{i-1}> + <Z_{i+1}>)/2)
    
    # Start with all <Z> = 0 (theta = pi/2)
    z_vals = np.zeros(n)
    for sweep in range(10):
        for i in range(n):
            z_left = z_vals[i-1] if i > 0 else 0.0
            z_right = z_vals[i+1] if i < n-1 else 0.0
            eff_z = coupling * (z_left + z_right) / 2.0
            if abs(eff_z) < 1e-10:
                theta_i = np.pi / 2
            else:
                theta_i = np.arctan2(field, eff_z)
                # Ensure theta in [0, pi] for cos(t/2), sin(t/2) >= 0
                if theta_i < 0:
                    theta_i = np.pi + theta_i
                if theta_i > np.pi:
                    theta_i = 2*np.pi - theta_i
            z_vals[i] = np.cos(theta_i)
            angles[0, i] = theta_i
    
    # For layers 1 and 2, start with small corrections
    angles[1] = 0.1 * angles[0]
    angles[2] = 0.05 * angles[0]
    
    return angles

def _optimize_for_case(n, coupling, field, time_budget):
    """Optimize angles for a specific case."""
    ground_energy, _ = _exact_ground_state(n, coupling, field)
    
    # Initial guess
    angles0 = _mean_field_init(n, coupling, field)
    params0 = angles0.flatten()
    
    best_params = params0.copy()
    best_energy = _energy_gap_objective(params0, n, coupling, field, ground_energy)
    
    # Try multiple optimization methods and initializations
    inits = [params0]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = params0 + rng.normal(0, 0.5, size=params0.shape)
        inits.append(perturbed)
    
    # Also try all zeros and all pi/2
    inits.append(np.zeros(3 * n))
    inits.append(np.full(3 * n, np.pi / 2))
    
    for init in inits:
        if _remaining() < 15:
            break
        
        try:
            result = minimize(
                _energy_gap_objective,
                init,
                args=(n, coupling, field, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Polish with tighter tolerance if time allows
    if _remaining() > 20:
        try:
            result = minimize(
                _energy_gap_objective,
                best_params,
                args=(n, coupling, field, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(3, n)

def angles(n, coupling, field, layers):
    """
    Main entry point. Returns angles (layers, n) in radians.
    """
    # Cache ground state computations
    ground_energy, _ = _exact_ground_state(n, coupling, field)
    
    # Optimize
    result = _optimize_for_case(n, coupling, field, _remaining())
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=np.pi, neginf=-np.pi)
    
    return result
