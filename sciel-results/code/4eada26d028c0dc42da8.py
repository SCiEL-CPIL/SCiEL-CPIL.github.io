"""Optimize 3-layer Ry/CZ circuit to approximate Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time


def _ry_matrix(theta):
    """Return the 2x2 Ry rotation matrix."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _cz_matrix():
    """Return the 2x2 CZ gate matrix."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=np.complex128)


def _simulate_circuit(n, thetas):
    """
    Simulate the circuit: |0>^n -> Ry(theta1) -> CZ -> Ry(theta2) -> CZ -> Ry(theta3)
    thetas: (layers, n) array of angles
    Returns: statevector of shape (2^n,)
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0>^n

    layers = thetas.shape[0]
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for q in range(n):
            ry = _ry_matrix(thetas[layer, q])
            # Apply single-qubit gate to qubit q
            new_state = np.zeros(dim, dtype=np.complex128)
            for basis in range(dim):
                bit = (basis >> q) & 1
                if bit == 0:
                    # |0> component
                    idx0 = basis
                    idx1 = basis | (1 << q)
                    new_state[idx0] += ry[0, 0] * state[idx0] + ry[0, 1] * state[idx1]
                    new_state[idx1] += ry[1, 0] * state[idx0] + ry[1, 1] * state[idx1]
                # bit == 1 is handled above
            state = new_state

        # Apply CZ between adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                cz = _cz_matrix()
                new_state = np.zeros(dim, dtype=np.complex128)
                for basis in range(dim):
                    bit_q = (basis >> q) & 1
                    bit_q1 = (basis >> (q + 1)) & 1
                    if bit_q == 0 and bit_q1 == 0:
                        idx = basis
                        new_state[idx] += cz[0, 0] * state[idx]
                    elif bit_q == 1 and bit_q1 == 0:
                        idx = basis
                        new_state[idx] += cz[1, 1] * state[idx]
                    elif bit_q == 0 and bit_q1 == 1:
                        idx = basis
                        new_state[idx] += cz[2, 2] * state[idx]
                    elif bit_q == 1 and bit_q1 == 1:
                        idx = basis
                        new_state[idx] += cz[3, 3] * state[idx]
                state = new_state

    return state


def _simulate_circuit_fast(n, thetas):
    """
    Faster simulation using tensor operations for small n.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    layers = thetas.shape[0]
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            new_state = np.zeros(dim, dtype=np.complex128)
            for basis in range(dim):
                bit = (basis >> q) & 1
                if bit == 0:
                    idx0 = basis
                    idx1 = basis | (1 << q)
                    new_state[idx0] = c * state[idx0] - s * state[idx1]
                    new_state[idx1] = s * state[idx0] + c * state[idx1]
            state = new_state

        # Apply CZ gates
        if layer < layers - 1:
            for q in range(n - 1):
                mask_q = 1 << q
                mask_q1 = 1 << (q + 1)
                new_state = state.copy()
                for basis in range(dim):
                    if (basis & mask_q) and (basis & mask_q1):
                        new_state[basis] = -state[basis]
                state = new_state

    return state


def _exact_ground_state(n, coupling, field):
    """
    Compute the exact ground state of H = -coupling*sum Z_i Z_{i+1} - field*sum X_i
    using exact diagonalization.
    """
    dim = 1 << n
    # Build Hamiltonian
    H = np.zeros((dim, dim), dtype=np.complex128)

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for basis in range(dim):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            z_i = 1 - 2 * bit_i
            z_ip1 = 1 - 2 * bit_ip1
            H[basis, basis] += -coupling * z_i * z_ip1

    # X_i terms
    for i in range(n):
        for basis in range(dim):
            bit_i = (basis >> i) & 1
            # X flips bit i
            basis_flip = basis ^ (1 << i)
            H[basis_flip, basis] += -field

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0]

    # Ensure consistent phase: make the largest magnitude component real and positive
    max_idx = np.argmax(np.abs(ground_state))
    phase = np.angle(ground_state[max_idx])
    ground_state = ground_state * np.exp(-1j * phase)

    return ground_state, eigvals[0]


def _mean_field_init(n, coupling, field, layers):
    """
    Initialize angles using a variational mean-field approximation.
    For a product state |psi> = ⊗|θ_i>, the energy is:
    E = -coupling * sum <Z_i Z_{i+1}> - field * sum <X_i>
    
    For a single qubit in state cos(θ/2)|0> + sin(θ/2)|1>:
    <Z> = cos(θ), <X> = sin(θ)
    
    So E(θ) = -coupling * sum cos(θ_i)cos(θ_{i+1}) - field * sum sin(θ_i)
    
    Minimize this with respect to θ_i.
    """
    theta = np.zeros(layers, dtype=np.float64)

    # Simple gradient descent on mean-field energy
    for _ in range(200):
        # Compute gradient
        grad = np.zeros(n, dtype=np.float64)
        for i in range(n):
            dE_dtheta_i = coupling * np.sin(theta[i]) * np.cos(theta[i])
            if i > 0:
                dE_dtheta_i += coupling * np.cos(theta[i]) * np.sin(theta[i - 1])
            if i < n - 1:
                dE_dtheta_i += coupling * np.cos(theta[i]) * np.sin(theta[i + 1])
            dE_dtheta_i += field * np.cos(theta[i])
            grad[i] = -dE_dtheta_i  # negative because we're minimizing

        # Gradient step
        lr = 0.05
        theta -= lr * grad

    # Distribute the mean-field angles across layers
    # Heuristic: spread the rotation across layers
    # For 3 layers, we can try: layer 1 gets small, layer 2 gets medium, layer 3 gets the rest
    # Or simply divide equally
    result = np.zeros((layers, n), dtype=np.float64)

    # Try a few different distributions and pick the one with best mean-field energy
    best_energy = np.inf
    best_result = result

    # Distribution 1: equal split
    for l in range(layers):
        result[l] = theta / layers
    energy = _mean_field_energy(n, coupling, field, result)
    if energy < best_energy:
        best_energy = energy
        best_result = result.copy()

    # Distribution 2: last layer does most of the work
    result2 = np.zeros((layers, n), dtype=np.float64)
    result2[-1] = theta * 0.7
    result2[-2] = theta * 0.2
    result2[0] = theta * 0.1
    energy2 = _mean_field_energy(n, coupling, field, result2)
    if energy2 < best_energy:
        best_energy = energy2
        best_result = result2.copy()

    # Distribution 3: first and last layers
    result3 = np.zeros((layers, n), dtype=np.float64)
    result3[0] = theta * 0.3
    result3[2] = theta * 0.7
    energy3 = _mean_field_energy(n, coupling, field, result3)
    if energy3 < best_energy:
        best_energy = energy3
        best_result = result3.copy()

    return best_result


def _mean_field_energy(n, coupling, field, thetas):
    """Compute approximate energy for mean-field analysis."""
    # For a product state, <Z_i> = cos(total_theta_i), <X_i> = sin(total_theta_i)
    total_theta = np.sum(thetas, axis=0)
    energy = 0.0
    for i in range(n - 1):
        energy += -coupling * np.cos(total_theta[i]) * np.cos(total_theta[i + 1])
    for i in range(n):
        energy += -field * np.sin(total_theta[i])
    return energy


def _objective(params, n, target_state):
    """
    Objective function for least_squares: returns real and imaginary parts
    of (circuit_state - target_state).
    """
    thetas = params.reshape(3, n)
    circuit_state = _simulate_circuit_fast(n, thetas)
    diff = circuit_state - target_state
    # Return as real vector
    return np.concatenate([diff.real, diff.imag])


def _objective_jac(params, n, target_state):
    """
    Numerical Jacobian for the objective.
    """
    # Use finite differences for the Jacobian
    eps = 1e-7
    m = params.size
    jacobian = np.zeros((2 * (1 << n), m), dtype=np.float64)

    for j in range(m):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[j] += eps
        params_minus[j] -= eps

        f_plus = _objective(params_plus, n, target_state)
        f_minus = _objective(params_minus, n, target_state)

        jacobian[:, j] = (f_plus - f_minus) / (2 * eps)

    return jacobian


def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to approximate the ground state.
    
    Args:
        n: number of qubits (4 or 6)
        coupling: Ising coupling constant
        field: transverse field strength
        layers: number of layers (always 3)
    
    Returns:
        (layers, n) array of angles in radians
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    # Compute exact ground state
    target_state, ground_energy = _exact_ground_state(n, coupling, field)

    # Initialize using mean-field approximation
    init_thetas = _mean_field_init(n, coupling, field, layers)
    init_params = init_thetas.flatten()

    best_params = init_params.copy()
    best_obj_val = None

    # Compute initial objective
    try:
        init_obj = _objective(init_params, n, target_state)
        best_obj_val = np.sum(init_obj ** 2)
    except Exception:
        pass

    # Use least_squares with numerical Jacobian
    try:
        result = least_squares(
            _objective,
            init_params,
            args=(n, target_state),
            method='trf',
            max_nfev=5000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        if result.success or result.cost < (best_obj_val if best_obj_val is not None else np.inf):
            best_params = result.x
            best_obj_val = result.cost

        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            # Try a polish pass with tighter tolerances
            result2 = least_squares(
                _objective,
                best_params,
                args=(n, target_state),
                method='trf',
                max_nfev=3000,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14,
            )

            if result2.cost < best_obj_val:
                best_params = result2.x
                best_obj_val = result2.cost

    except Exception as e:
        pass

    # Try a few random restarts if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 15:
                break

            random_params = np.random.uniform(-np.pi, np.pi, size=(3, n)).flatten()
            try:
                result_r = least_squares(
                    _objective,
                    random_params,
                    args=(n, target_state),
                    method='trf',
                    max_nfev=2000,
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                )

                if result_r.cost < best_obj_val:
                    best_params = result_r.x
                    best_obj_val = result_r.cost

            except Exception:
                pass

    # Reshape to (layers, n)
    result_thetas = best_params.reshape(layers, n)

    # Ensure all values are finite
    result_thetas = np.nan_to_num(result_thetas, nan=0.0, posinf=np.pi, neginf=-np.pi)

    return result_thetas

# EVOLVE-BLOCK-END
