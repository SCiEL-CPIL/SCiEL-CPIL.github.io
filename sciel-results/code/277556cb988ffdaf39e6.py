"""Quantum information: shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute exact ground energies and circuit simulation helpers
# The evaluator calls angles(n, coupling, field, layers) for specific (n, field) pairs.
# We need to:
# 1. Build the Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
# 2. Find exact ground energy E0
# 3. Define a fast circuit simulation: start |0>^n, apply layers of Ry(theta) with CZ between
# 4. Optimize theta to minimize (E_circuit - E0) / n

# We'll cache results per (n, coupling, field) since the evaluator may call multiple times.
# But actually the evaluator likely calls once per test case. Let's just compute each time.
# Key: keep within 120s (we have 150s wall-clock but should finish well before).

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for open chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            # Z eigenvalue: +1 for |0>, -1 for |1>
            zi = 1.0 if bit_i == 0 else -1.0
            zj = 1.0 if bit_ip1 == 0 else -1.0
            H[state, state] += -coupling * zi * zj
    
    # X_i terms: X flips bit i
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    # Use eigh for symmetric matrices
    eigs = np.linalg.eigvalsh(H)
    return eigs[0]


def _circuit_energy(theta, n, coupling, field):
    """
    Compute energy of the circuit state.
    theta: (layers, n) array of angles
    Circuit: start |0>^n, for each layer: apply Ry(theta[l]) to all qubits, 
             then CZ on adjacent pairs (except after last layer).
    """
    dim = 1 << n
    # State vector
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0  # |0>^n
    
    L = theta.shape[0]
    
    for l in range(L):
        # Apply Ry(theta[l, i]) to each qubit i
        for i in range(n):
            # Ry(theta) = cos(theta/2) |0><0| - sin(theta/2) |0><1| 
        #               - sin(theta/2) |1><0| + cos(theta/2) |1><1|
            # Wait, standard Ry: 
            # Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
            # So for qubit i, we need to apply this to every basis state.
            theta_i = theta[l, i]
            c = np.cos(theta_i / 2.0)
            s = np.sin(theta_i / 2.0)
            
            new_psi = np.zeros_like(psi)
            for state in range(dim):
                bit_i = (state >> i) & 1
                if bit_i == 0:
                    # |0> -> c|0> + s|1>
                    state_0 = state
                    state_1 = state | (1 << i)
                    new_psi[state_0] += c * psi[state]
                    new_psi[state_1] += s * psi[state]
                else:
                    # |1> -> -s|0> + c|1>
                    state_1 = state
                    state_0 = state & ~(1 << i)
                    new_psi[state_0] += -s * psi[state]
                    new_psi[state_1] += c * psi[state]
            psi = new_psi
        
        # Apply CZ on adjacent pairs if not last layer
        if l < L - 1:
            for i in range(n - 1):
                # CZ: |11> -> -|11>
                for state in range(dim):
                    bit_i = (state >> i) & 1
                    bit_ip1 = (state >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        psi[state] *= -1.0
    
    # Compute energy: <psi|H|psi>
    # H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    energy = 0.0
    
    # Z_i Z_{i+1} terms (diagonal)
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zj = 1.0 if bit_ip1 == 0 else -1.0
            energy += -coupling * zi * zj * (np.abs(psi[state]) ** 2)
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            # <state|X|flipped> = 1, so contribution is -field * psi[state]^* * psi[flipped]
            # But we need <psi|X_i|psi> = sum_state psi[state]^* * psi[flipped]
            # Actually X_i |flipped> = |state> if flipped = state ^ (1<<i)
            # <psi|X_i|psi> = sum_state conj(psi[state]) * psi[state ^ (1<<i)]
            # This is complex in general, but for real states it's real.
            # Our circuit produces real states (Ry and CZ with real angles), so psi is real.
            energy += -field * (psi[state] * psi[flipped])
    
    # Note: the X_i sum counts each pair twice? No, X_i is a single operator.
    # <psi|X_i|psi> = sum_s conj(psi_s) psi_{s^(1<<i)}
    # But this sums over all s, so each pair (s, s') is counted twice if we're not careful.
    # Actually no: for a fixed i, the map s -> s^(1<<i) is an involution.
    # sum_s conj(psi_s) psi_{s^(1<<i)} 
    # Let s' = s^(1<<i), then s = s'^(1<<i), so:
    # = sum_{s'} conj(psi_{s'^(1<<i)}) psi_{s'}
    # This is not the same as 2 * sum_{s < s^(1<<i)} ... 
    # Actually it's just one sum. Let me verify with n=1:
    # <psi|X|psi> = conj(psi_0)*psi_1 + conj(psi_1)*psi_0 = 2*Re(conj(psi_0)*psi_1)
    # So yes, it's correct as written.
    
    return energy


def _objective(theta_flat, n, coupling, field, E0):
    """Objective: mean energy gap per spin."""
    theta = theta_flat.reshape(-1, n)
    E_circuit = _circuit_energy(theta, n, coupling, field)
    return (E_circuit - E0) / n


def _physics_init(n, field, layers):
    """
    Physics-informed initialization.
    For the transverse field Ising model, the ground state in the strong-field limit 
    is close to |+>^n, and in the strong-coupling limit it's close to |0>^n or |1>^n.
    
    A reasonable guess: use angles that interpolate. For a single qubit with H = -h X,
    the ground state is at angle pi/2 (|+>). With ZZ coupling, it shifts.
    
    Let's use a simple heuristic: theta ≈ pi/2 * (field / (field + coupling))
    but this is rough. Let's just use pi/2 as a starting point for all.
    """
    theta = np.zeros((layers, n))
    # Start with a guess based on field strength
    # For pure transverse field, optimal is pi/2
    # For pure coupling, optimal is 0
    # Simple interpolation
    base_angle = np.pi / 2.0 * (field / (field + coupling))
    for l in range(layers):
        theta[l, :] = base_angle
    return theta


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Get exact ground energy
    E0 = _exact_ground_energy(n, coupling, field)
    
    # Initial guess
    theta0 = _physics_init(n, field, layers)
    x0 = theta0.flatten()
    
    best_x = x0.copy()
    best_obj = _objective(x0, n, coupling, field, E0)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            _objective, x0, args=(n, coupling, field, E0),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000, 'gtol': 1e-8}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_x.reshape(layers, n)
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        x_rand = np.random.uniform(-np.pi, np.pi, size=(layers, n)).flatten()
        try:
            result = minimize(
                _objective, x_rand, args=(n, coupling, field, E0),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(layers, n)

# EVOLVE-BLOCK-END
