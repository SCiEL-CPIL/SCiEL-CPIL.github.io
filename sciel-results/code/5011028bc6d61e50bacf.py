"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_circuit(state, thetas, n, layers):
    """
    Apply the circuit to an initial |0...0> state.
    thetas: (layers, n) array of angles.
    Returns the final state vector of dimension 2**n.
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0...0>

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            ry = _ry_matrix(thetas[layer, i])
            # Apply Ry on qubit i
            new_state = np.zeros(dim, dtype=np.complex128)
            for idx in range(dim):
                # Determine if qubit i is 0 or 1 in this basis state
                bit = (idx >> i) & 1
                if bit == 0:
                    # qubit i is 0
                    idx1 = idx | (1 << i)  # flip bit i to 1
                    new_state[idx] += ry[0, 0] * state[idx] + ry[0, 1] * state[idx1]
                    new_state[idx1] += ry[1, 0] * state[idx] + ry[1, 1] * state[idx1]
                # We handle each pair (idx, idx1) exactly once by only processing when bit==0
            state = new_state

        # Apply CZ on every adjacent pair in ascending order, except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                # Apply CZ on qubits i and i+1
                # CZ is diagonal: |11> gets phase -1
                for idx in range(dim):
                    bit_i = (idx >> i) & 1
                    bit_ip1 = (idx >> (i + 1)) & 1
                    if bit_i == 1 and bit_ip1 == 1:
                        state[idx] *= -1.0

    return state


def _energy(state, n, coupling, field):
    """
    Compute <H> for the state.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    dim = 2 ** n
    e_zz = 0.0
    e_x = 0.0

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            # Z eigenvalue is +1 if bit=0, -1 if bit=1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz = z_i * z_ip1
            e_zz += zz * (np.abs(state[idx]) ** 2)

    # X_i terms: <X_i> = <0|_i X |1> + <1|_i X |0> = sum_{pairs} (c_0^* c_1 + c_1^* c_0)
    for i in range(n):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            if bit_i == 0:
                idx1 = idx | (1 << i)
                e_x += 2.0 * np.real(np.conj(state[idx]) * state[idx1])

    return -coupling * e_zz - field * e_x


def _objective(params, n, coupling, field, layers):
    thetas = params.reshape(layers, n)
    state = _apply_circuit(np.zeros(2**n, dtype=np.complex128), thetas, n, layers)
    # Reset state[0] = 1 is done inside _apply_circuit, but we pass a zero array.
    # Wait, _apply_circuit sets state[0]=1.0 internally.
    e = _energy(state, n, coupling, field)
    return e


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


def _circuit_energy_fast(thetas, n, coupling, field, layers):
    """Compute circuit energy using a faster method."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        # Apply Ry to all qubits
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # For each pair of basis states differing only in bit i
            for idx in range(dim):
                if (idx >> i) & 1 == 0:
                    idx1 = idx | (1 << i)
                    a, b = state[idx], state[idx1]
                    state[idx] = c * a - s * b
                    state[idx1] = s * a + c * b
        
        # Apply CZ between layers
        if layer < layers - 1:
            for i in range(n - 1):
                for idx in range(dim):
                    if ((idx >> i) & 1) and ((idx >> (i+1)) & 1):
                        state[idx] *= -1.0
    
    # Compute energy
    e_zz = 0.0
    e_x = 0.0
    for i in range(n - 1):
        for idx in range(dim):
            z_i = 1.0 if ((idx >> i) & 1) == 0 else -1.0
            z_ip1 = 1.0 if ((idx >> (i+1)) & 1) == 0 else -1.0
            e_zz += z_i * z_ip1 * (np.abs(state[idx]) ** 2)
    for i in range(n):
        for idx in range(dim):
            if ((idx >> i) & 1) == 0:
                idx1 = idx | (1 << i)
                e_x += 2.0 * np.real(np.conj(state[idx]) * state[idx1])
    
    return -coupling * e_zz - field * e_x


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
        e = _circuit_energy_fast(thetas, n, coupling, field, layers)
        return e

    # Generate diverse initializations
    n_params = layers * n
    inits = []
    
    # Mean-field like initialization: pi/2 for all
    inits.append(np.full(n_params, np.pi / 2.0))
    
    # All zeros
    inits.append(np.zeros(n_params))
    
    # Random inits
    np.random.seed(42)
    for _ in range(8):
        inits.append(np.random.uniform(-np.pi, np.pi, size=n_params))
    
    # Symmetric inits: same angle for all qubits in a layer
    for _ in range(4):
        layer_angles = np.random.uniform(-np.pi, np.pi, size=layers)
        inits.append(np.repeat(layer_angles, n))
    
    # Sort by objective value
    init_objs = []
    for x in inits:
        init_objs.append((obj(x), x))
    init_objs.sort(key=lambda t: t[0])
    
    best_params = init_objs[0][1]
    best_val = init_objs[0][0]
    
    # Optimize from best initializations
    n_restarts = 6
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
        
        x_start = init_objs[r][1] if r < len(init_objs) else init_objs[0][1] + np.random.randn(n_params) * 0.3
        
        try:
            res = minimize(
                obj,
                x_start,
                method='BFGS',
                options={'maxiter': 300, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        try:
            res = minimize(
                obj,
                best_params,
                method='BFGS',
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return best_params.reshape(layers, n)


# EVOLVE-BLOCK-END
