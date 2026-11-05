"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _circuit_state_vectorized(thetas, n, layers):
    """
    Apply the circuit to |0...0> using vectorized operations.
    thetas: (layers, n) array of angles.
    Returns state vector of dimension 2**n.
    """
    dim = 2 ** n
    # State as array of shape (2, 2, ..., 2) with n axes, qubit i is axis i
    state = np.zeros([2] * n, dtype=np.complex128)
    state[(0,) * n] = 1.0

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Move qubit i to front, apply 2x2 matrix, move back
            state = np.moveaxis(state, i, 0)
            new_state = np.empty_like(state)
            new_state[0] = c * state[0] - s * state[1]
            new_state[1] = s * state[0] + c * state[1]
            state = np.moveaxis(new_state, 0, i)

        # Apply CZ on every adjacent pair in ascending order, except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ: multiply by -1 when both qubits i and i+1 are 1
                # Create mask for states where bit i = 1 and bit i+1 = 1
                shape = [1] * n
                shape[i] = 2
                shape[i + 1] = 2
                mask = np.zeros([2] * n, dtype=np.complex128)
                mask[(0,) * i + (1, 1) + (0,) * (n - i - 2)] = -1.0
                # Actually, let's do this more simply with broadcasting
                # Extract the 2x2 block for qubits i and i+1
                state = np.moveaxis(state, i, 0)
                state = np.moveaxis(state, i, 0)  # now qubit i is axis 0, qubit i+1 is axis 1 (originally i+1, now shifted)
                # Wait, after moving i to 0, the original i+1 is now at position 1
                # Apply CZ phase: state[1,1,...] *= -1
                idx = (1, 1) + (0,) * (n - 2)
                state[idx] *= -1.0
                # Move back
                state = np.moveaxis(state, 0, i)
                state = np.moveaxis(state, 0, i)

    return state.reshape(dim)


def _circuit_state_fast(thetas, n, layers):
    """
    Fast circuit simulation using reshape-based qubit operations.
    Qubit i corresponds to bit i (little-endian).
    State indexed as state[b0, b1, ..., b_{n-1}] where bi is bit i.
    """
    dim = 2 ** n
    # State as n-dimensional array, axis i = qubit i
    state = np.zeros([2] * n, dtype=np.complex128)
    state[(0,) * n] = 1.0

    for layer in range(layers):
        # Apply Ry(theta) to each qubit
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Move qubit i to axis 0
            state = np.moveaxis(state, i, 0)
            # Apply Ry: new[0] = c*old[0] - s*old[1], new[1] = s*old[0] + c*old[1]
            old0 = state[0].copy()
            old1 = state[1].copy()
            state[0] = c * old0 - s * old1
            state[1] = s * old0 + c * old1
            # Move back
            state = np.moveaxis(state, 0, i)

        # Apply CZ between adjacent qubits (ascending order), except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ on qubits i and i+1: phase -1 when both are 1
                # Move qubit i to axis 0, qubit i+1 to axis 1
                state = np.moveaxis(state, i, 0)
                # Now qubit i+1 was at axis i+1, after moving i to 0, it's at axis i
                state = np.moveaxis(state, i, 1)
                # Apply phase: state[1, 1, ...] *= -1
                state[1, 1] *= -1.0
                # Move back
                state = np.moveaxis(state, 1, i)
                state = np.moveaxis(state, 0, i)

    return state.reshape(dim)


def _energy_from_state(state, n, coupling, field):
    """
    Compute <H> for the state.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    Qubit i is bit i (little-endian).
    """
    dim = 2 ** n
    # Reshape state to n-dimensional array for easier indexing
    state_n = state.reshape([2] * n)

    e_zz = 0.0
    for i in range(n - 1):
        # Z_i Z_{i+1}: +1 if both 0 or both 1, -1 otherwise
        # Z eigenvalue: +1 for |0>, -1 for |1>
        # So Z_i Z_{i+1} = +1 if b_i == b_{i+1}, -1 otherwise
        # = 1 - 2*(b_i XOR b_{i+1})
        # Expectation: sum |psi|^2 * (1 - 2*(b_i XOR b_{i+1}))
        # = 1 - 2 * P(b_i != b_{i+1})
        # P(b_i != b_{i+1}) = sum_{b_i != b_{i+1}} |psi|^2
        # Move qubits i, i+1 to front
        s = np.moveaxis(state_n, i, 0)
        s = np.moveaxis(s, i, 1)
        # s[0,0,...] = both 0, s[0,1,...] = i=0,i+1=1, s[1,0,...] = i=1,i+1=0, s[1,1,...] = both 1
        p00 = np.sum(np.abs(s[0, 0]) ** 2)
        p01 = np.sum(np.abs(s[0, 1]) ** 2)
        p10 = np.sum(np.abs(s[1, 0]) ** 2)
        p11 = np.sum(np.abs(s[1, 1]) ** 2)
        # Z_i Z_{i+1} expectation: p00 - p01 - p10 + p11
        e_zz += p00 - p01 - p10 + p11

    e_x = 0.0
    for i in range(n):
        # X_i expectation: <psi|X_i|psi> = sum_{b_i=0} psi^*[...0...] psi[...1...] + c.c.
        # = 2 Re(sum_{b_i=0} conj(psi_0) * psi_1)
        s = np.moveaxis(state_n, i, 0)
        # s[0, ...] = states with b_i = 0, s[1, ...] = states with b_i = 1
        e_x += 2.0 * np.real(np.sum(np.conj(s[0]) * s[1]))

    return -coupling * e_zz - field * e_x


def _exact_ground_energy(n, coupling, field):
    """
    Compute exact ground state energy by diagonalizing H.
    """
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.complex128)

    # ZZ terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * z_i * z_ip1

    # X terms
    for i in range(n):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                idx1 = idx | (1 << i)
                H[idx, idx1] += -field
                H[idx1, idx] += -field

    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])


def _circuit_energy_and_grad(thetas, n, coupling, field, layers):
    """
    Compute circuit energy and its gradient with respect to thetas using
    the parameter-shift rule for Ry gates.
    """
    dim = 2 ** n
    e0 = _circuit_state_fast(thetas, n, layers)
    energy = _energy_from_state(e0, n, coupling, field)

    # Gradient via parameter-shift rule: dE/dtheta_i = (E(theta_i + pi/2) - E(theta_i - pi/2)) / 2
    grad = np.zeros_like(thetas)
    for layer in range(layers):
        for i in range(n):
            theta_plus = thetas.copy()
            theta_plus[layer, i] += np.pi / 2.0
            theta_minus = thetas.copy()
            theta_minus[layer, i] -= np.pi / 2.0

            e_plus = _circuit_state_fast(theta_plus, n, layers)
            e_minus = _circuit_state_fast(theta_minus, n, layers)

            ep = _energy_from_state(e_plus, n, coupling, field)
            em = _energy_from_state(e_minus, n, coupling, field)

            grad[layer, i] = (ep - em) / 2.0

    return energy, grad


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    # Precompute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)

    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        thetas = params.reshape(layers, n)
        state = _circuit_state_fast(thetas, n, layers)
        return _energy_from_state(state, n, coupling, field)

    def obj_grad(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10, np.zeros_like(params)
        thetas = params.reshape(layers, n)
        e, g = _circuit_energy_and_grad(thetas, n, coupling, field, layers)
        return e, g.reshape(-1)

    # Generate diverse initializations
    n_params = layers * n
    inits = []

    # Mean-field like initialization: pi/2 for all
    inits.append(np.full(n_params, np.pi / 2.0))

    # All zeros
    inits.append(np.zeros(n_params))

    # All -pi/2
    inits.append(np.full(n_params, -np.pi / 2.0))

    # Symmetric: same angle per layer
    for _ in range(6):
        layer_angles = np.random.uniform(-np.pi, np.pi, size=layers)
        inits.append(np.repeat(layer_angles, n))

    # Random inits
    np.random.seed(42)
    for _ in range(12):
        inits.append(np.random.uniform(-np.pi, np.pi, size=n_params))

    # Near-symmetric with small perturbations
    for _ in range(6):
        base = np.random.uniform(-np.pi/4, np.pi/4, size=n_params)
        inits.append(base + np.random.randn(n_params) * 0.3)

    # Sort by objective value
    init_objs = []
    for x in inits:
        init_objs.append((obj(x), x))
    init_objs.sort(key=lambda t: t[0])

    best_params = init_objs[0][1].copy()
    best_val = init_objs[0][0]

    # Optimize from best initializations using L-BFGS-B with analytic gradients
    n_restarts = 8
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 20.0:
            break

        if r < len(init_objs):
            x_start = init_objs[r][1].copy()
        else:
            x_start = best_params + np.random.randn(n_params) * 0.5

        try:
            res = minimize(
                obj_grad,
                x_start,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            res = minimize(
                obj_grad,
                best_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # One more polish if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            res = minimize(
                obj_grad,
                best_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
