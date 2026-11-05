# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import diags, kron
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
import time

def _build_hamiltonian(n, coupling, field):
    """Build sparse Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    # Build H term by term using kron
    H = np.zeros((dim, dim), dtype=complex)

    # ZZ terms: -coupling * Z_i ⊗ Z_{i+1}
    for i in range(n - 1):
        ops = []
        for q in range(n - 1, -1, -1):
            if q == i or q == i + 1:
                ops.append(Z)
            else:
                ops.append(I)
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += -coupling * term

    # X terms: -field * X_i
    for i in range(n):
        ops = []
        for q in range(n - 1, -1, -1):
            if q == i:
                ops.append(X)
            else:
                ops.append(I)
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += -field * term

    return H

def _apply_ry_vec(state, qubit, theta, n):
    """Apply Ry(theta) to qubit in state vector (qubit 0 = LSB) using vectorized ops."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    axis = n - 1 - qubit
    
    axes = list(range(n))
    axes.remove(axis)
    axes.append(axis)
    state_t = np.transpose(state, axes)
    
    shape = state_t.shape
    state_2d = state_t.reshape(-1, 2)
    
    new_0 = c * state_2d[:, 0] - s * state_2d[:, 1]
    new_1 = s * state_2d[:, 0] + c * state_2d[:, 1]
    
    new_state_2d = np.stack([new_0, new_1], axis=1)
    new_state_t = new_state_2d.reshape(shape)
    
    inv_axes = [0] * n
    for i, a in enumerate(axes):
        inv_axes[a] = i
    new_state = np.transpose(new_state_t, inv_axes)
    
    return new_state

def _apply_cz_vec(state, q1, q2, n):
    """Apply CZ to qubits q1 and q2 (0-indexed from right) using vectorized ops."""
    mask1 = 1 << q1
    mask2 = 1 << q2
    idxs = np.arange(1 << n)
    mask = (idxs & mask1) & (idxs & mask2)
    signs = np.where(mask != 0, -1.0, 1.0)
    return state * signs

def _circuit_state(params, n, layers):
    """Simulate the circuit: start |0>^n, apply layers of Ry then CZ."""
    state = np.zeros(1 << n, dtype=complex)
    state[0] = 1.0  # |0>^n

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            theta = params[layer, q]
            state = _apply_ry_vec(state, q, theta, n)
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz_vec(state, q, q + 1, n)

    return state

def _energy(params, n, coupling, field, H, layers):
    """Compute energy expectation value of circuit state."""
    state = _circuit_state(params, n, layers)
    # <psi|H|psi>
    Hpsi = H @ state
    energy = np.vdot(state, Hpsi).real
    return energy

def _objective(params_flat, n, coupling, field, H, ground_energy, layers):
    """Objective: mean energy gap per spin = (E_circuit - E_ground) / n."""
    params = params_flat.reshape(layers, n)
    e = _energy(params, n, coupling, field, H, layers)
    return (e - ground_energy) / n

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    # Build Hamiltonian
    H = _build_hamiltonian(n, coupling, field)

    # Get exact ground energy
    dim = 1 << n
    if dim <= 64:
        # Use dense eigensolver for small systems
        eigs = np.linalg.eigvalsh(H)
        ground_energy = eigs[0].real
    else:
        ground_energy = eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]

    # Initial guess: Physics inspired
    # Single qubit ground state of -g X is cos(theta/2)|0> + sin(theta/2)|1>
    # tan(theta/2) = 1/g => theta = 2*arctan(1/g)
    theta_physics = 2 * np.arctan(1.0 / field)
    
    # Initialize all layers with this value
    best_params = np.full((layers, n), theta_physics)
    best_obj = _objective(best_params.ravel(), n, coupling, field, H, ground_energy, layers)

    # Try a few other initializations
    initial_guesses = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 2),
        np.full((layers, n), np.pi / 4),
        np.random.uniform(-np.pi, np.pi, (layers, n)),
    ]

    for guess in initial_guesses:
        if time.time() > deadline:
            break
        obj_val = _objective(guess.ravel(), n, coupling, field, H, ground_energy, layers)
        if obj_val < best_obj:
            best_obj = obj_val
            best_params = guess.copy()

    # Optimize from the best initial guess
    result = minimize(
        _objective,
        best_params.ravel(),
        args=(n, coupling, field, H, ground_energy, layers),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
    )

    if result.fun < best_obj:
        best_params = result.x.reshape(layers, n)
        best_obj = result.fun

    # Try a second optimization from a different random start if time allows
    if time.time() < deadline - 5.0:
        guess2 = np.random.uniform(-np.pi, np.pi, (layers, n))
        obj2 = _objective(guess2.ravel(), n, coupling, field, H, ground_energy, layers)
        if obj2 < best_obj:
            result2 = minimize(
                _objective,
                guess2.ravel(),
                args=(n, coupling, field, H, ground_energy, layers),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < best_obj:
                best_params = result2.x.reshape(layers, n)
                best_obj = result2.fun

    # Ensure output is finite
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=np.pi, neginf=-np.pi)

    return best_params
# EVOLVE-BLOCK-END
