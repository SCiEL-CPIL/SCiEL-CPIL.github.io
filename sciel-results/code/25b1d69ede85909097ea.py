"""Rewritten program: layer-wise variational + multi-start optimization for Ry/CZ circuit."""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _ry_matrix(theta):
    """Ry rotation matrix for angle theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def _cz_matrix():
    """CZ gate matrix (4x4)."""
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.float64)

def _simulate_circuit(theta, n):
    """
    Simulate the circuit: start |0>^n, for each layer apply Ry(theta[layer]) to all qubits,
    then apply CZ on adjacent pairs (i, i+1) in ascending order (except after last layer).
    Returns state vector as float64 array of size 2^n.
    """
    dim = 1 << n
    # Start in |0>^n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0

    layers = theta.shape[0]
    for layer in range(layers):
        # Apply Ry(theta[layer, q]) to each qubit q
        for q in range(n):
            R = _ry_matrix(theta[layer, q])
            # Apply R to qubit q: reshape state, apply, reshape back
            # Qubit q is bit q (LSB). We need to apply R to the q-th qubit.
            # Reshape: state as tensor of shape (2,)*n, where index [i0, i1, ..., i_{n-1}]
            # corresponds to qubit 0 = i0 (LSB), qubit 1 = i1, etc.
            shape = [2] * n
            state_t = state.reshape(shape)
            # Apply R on axis q
            # R acts on the qubit: new[i_q] = sum_j R[i_q, j] old[j]
            # np.einsum or manual:
            # We'll do: new_state = np.tensordot(R, state_t, axes=[1, q])
            # This gives shape: (2, 2, ..., 2) with the q-axis replaced by R's first axis
            # Actually tensordot(R, state_t, axes=[1, q]) contracts R's col with state_t's axis q
            # Result shape: R's row axis (2) + remaining axes of state_t
            # Need to move the new axis to position q
            new_t = np.tensordot(R, state_t, axes=[1, q])
            # new_t has shape: (2,) + shape with axis q removed
            # We need to insert the (2,) axis back at position q
            new_shape = [2] * n
            # Move axis 0 to position q
            new_t = np.moveaxis(new_t, 0, q)
            state_t = new_t
            state = state_t.reshape(dim)

        # Apply CZ on adjacent pairs (i, i+1) for i in 0..n-2, ascending order
        if layer < layers - 1:
            for i in range(n - 1):
                CZ = _cz_matrix()
                # Apply CZ to qubits i and i+1
                # We need to apply a 4x4 gate to two adjacent qubits
                # Reshape state to tensor, apply CZ on axes i and i+1
                shape = [2] * n
                state_t = state.reshape(shape)
                # Move axes i and i+1 to the end for easier manipulation
                # Actually, let's use a different approach:
                # For each basis state, if bits i and i+1 are both 1, flip sign
                # This is simpler and faster for CZ
                # CZ: |11> -> -|11>, others unchanged
                # We can do this by masking
                mask = np.zeros(dim, dtype=np.float64)
                for idx in range(dim):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        mask[idx] = -1.0
                    else:
                        mask[idx] = 1.0
                state = state * mask

    return state

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open transverse-field Ising chain."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * z_i * z_ip1

    # X_i terms
    for i in range(n):
        for idx in range(dim):
            # X_i flips bit i
            idx_flip = idx ^ (1 << i)
            H[idx, idx_flip] += -field

    return H

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    # Use numpy eigvalsh for symmetric matrix
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]

def _energy(state, n, coupling, field):
    """Compute <state|H|state>."""
    dim = state.shape[0]
    energy = 0.0
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            energy += -coupling * z_i * z_ip1 * state[idx] * state[idx]

    # X_i terms
    for i in range(n):
        for idx in range(dim):
            idx_flip = idx ^ (1 << i)
            energy += -field * state[idx] * state[idx_flip]

    return energy

def _objective(params, n, coupling, field, layers):
    """Objective: mean energy gap per spin."""
    theta = params.reshape((layers, n))
    state = _simulate_circuit(theta, n)
    e_circuit = _energy(state, n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    return (e_circuit - e_ground) / n

def _mean_field_init(n, coupling, field, layers):
    """Mean-field inspired initialization."""
    theta = np.zeros((layers, n), dtype=np.float64)
    # For a single spin in field h=field, the ground state angle is arccos(h/sqrt(h^2+J^2)) approx
    # Simple: start with small angles
    theta[0] = field / 4.0
    theta[1] = field / 3.0
    theta[-1] = field / 2.0
    return theta

def _product_state_init(n, coupling, field, layers):
    """Try to approximate product ground state."""
    theta = np.zeros((layers, n), dtype=np.float64)
    # The +x eigenstate is pi/2
    theta[-1] = np.pi / 2
    theta[1] = np.pi / 4
    theta[0] = np.pi / 8
    return theta

def _random_init(n, coupling, field, layers, rng):
    """Random initialization."""
    return rng.uniform(-np.pi, np.pi, size=(layers, n))

def _transfer_matrix_init(n, coupling, field, layers, rng):
    """
    MPS-inspired: fit each layer's angles by matching the effective transfer matrix
    to the ground state's correlation structure.
    """
    # Get exact ground state
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_state = eigvecs[:, 0].real  # Should be real for this Hamiltonian
    ground_state = ground_state / np.linalg.norm(ground_state)

    # Try to extract single-qubit reduced density matrices to inform angles
    theta = np.zeros((layers, n), dtype=np.float64)
    dim = 1 << n

    for q in range(n):
        # Compute <Z_q> and <X_q> in ground state
        z_q = 0.0
        x_q = 0.0
        for idx in range(dim):
            bit_q = (idx >> q) & 1
            z_val = 1.0 if bit_q == 0 else -1.0
            z_q += z_val * ground_state[idx] * ground_state[idx]
            idx_flip = idx ^ (1 << q)
            x_q += ground_state[idx] * ground_state[idx_flip]

        # For a single qubit with <Z> = cos(theta_eff), <X> = sin(theta_eff)
        # The Ry angle that produces this: Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        # <Z> = cos(theta), <X> = sin(theta)
        # So theta_eff = atan2(x_q, z_q)
        theta_eff = np.arctan2(x_q, z_q)

        # Distribute this effective angle across layers
        # Layer 0: small correction, Layer 1: medium, Layer 2: large
        theta[0, q] = theta_eff * 0.2
        theta[1, q] = theta_eff * 0.3
        theta[2, q] = theta_eff * 0.5

    return theta

def angles(n, coupling, field, layers):
    """Main entry point: optimize Ry angles for the given parameters."""
    global _START_TIME
    _START_TIME = time.time()

    # Pre-compute exact ground energy
    e_ground = _exact_ground_energy(n, coupling, field)

    # Generate diverse initial guesses
    rng = np.random.RandomState(42)
    inits = []
    inits.append(_mean_field_init(n, coupling, field, layers))
    inits.append(_product_state_init(n, coupling, field, layers))
    inits.append(_transfer_matrix_init(n, coupling, field, layers, rng))
    for _ in range(4):
        inits.append(_random_init(n, coupling, field, layers, rng))

    best_theta = None
    best_energy = np.inf

    # Phase 1: L-BFGS-B from each initial guess
    for init in inits:
        if _remaining() < 5.0:
            break

        params0 = init.flatten()
        try:
            res = minimize(
                _objective,
                params0,
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < best_energy:
                e = res.fun
                if e < best_energy:
                    best_energy = e
                    best_theta = res.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 2: Polish with Nelder-Mead from best
    if best_theta is not None and _remaining() > 10.0:
        try:
            res2 = minimize(
                _objective,
                best_theta.flatten(),
                args=(n, coupling, field, layers),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res2.fun < best_energy:
                best_energy = res2.fun
                best_theta = res2.x.reshape((layers, n))
        except Exception:
            pass

    # Phase 3: Additional L-BFGS-B polish
    if best_theta is not None and _remaining() > 5.0:
        try:
            res3 = minimize(
                _objective,
                best_theta.flatten(),
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res3.fun < best_energy:
                best_energy = res3.fun
                best_theta = res3.x.reshape((layers, n))
        except Exception:
            pass

    # Fallback: if optimization failed, return a reasonable default
    if best_theta is None:
        best_theta = _product_state_init(n, coupling, field, layers)

    return best_theta.astype(np.float64)
