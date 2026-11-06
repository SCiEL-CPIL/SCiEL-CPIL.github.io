"""Variational ansatz: fit circuit angles to exact ground-state amplitudes via least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares


def _exact_ground_state(n, coupling, field):
    """Compute the exact ground state of the open transverse-field Ising chain."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    for basis in range(dim):
        zz_energy = 0.0
        for i in range(n - 1):
            bit_i = (basis >> i) & 1
            bit_ip1 = (basis >> (i + 1)) & 1
            z_i = 1.0 if bit_i == 0 else -1.0
            z_ip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz_energy += z_i * z_ip1
        H[basis, basis] += -coupling * zz_energy
        
        for i in range(n):
            flipped = basis ^ (1 << i)
            H[flipped, basis] += -field
            
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_idx = np.argmin(eigvals)
    return eigvecs[:, ground_idx].real, eigvals[ground_idx]


def _apply_circuit(state, angles, n):
    """Apply the 3-layer Ry/CZ circuit to the initial |0>^n state."""
    layers = angles.shape[0]
    state = state.copy()
    
    # Precompute Ry matrices for unique angles? No, angles vary.
    
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = angles[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            R = np.array([[c, -s], [s, c]])
            
            # Reshape state to tensor
            # Bit 0 is LSB, so it varies fastest.
            # In C-order array, last axis varies fastest.
            # So axis (n-1-i) corresponds to bit i?
            # Let's check:
            # n=2. Indices 0,1,2,3.
            # 0: 00. 1: 01. 2: 10. 3: 11.
            # Reshape (2,2).
            # [[0, 1], [2, 3]].
            # Axis 0: bit 1. Axis 1: bit 0.
            # So axis for bit i is n - 1 - i.
            
            axis = n - 1 - i
            shape = [2] * n
            state_tensor = state.reshape(shape)
            
            # Apply R to axis
            # tensordot(R, state_tensor, axes=(1, axis))
            # This contracts the second index of R (which is the "input" index) with the axis of the state.
            # Wait, R_{out, in}. State_{in, ...}.
            # Result_{out, ...}.
            # We want to replace the axis in the state.
            # new_state = np.tensordot(R, state_tensor, axes=(1, axis))
            # This puts the new axis at position 0.
            # We need to move it back to position 'axis'.
            
            new_state_tensor = np.tensordot(R, state_tensor, axes=(1, axis))
            new_state_tensor = np.moveaxis(new_state_tensor, 0, axis)
            
            state = new_state_tensor.reshape(-1)
        
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                # CZ on i, i+1
                # Flip sign where bit i == 1 and bit i+1 == 1
                mask = np.zeros(dim, dtype=bool)
                for basis in range(dim):
                    if ((basis >> i) & 1) and ((basis >> (i+1)) & 1):
                        mask[basis] = True
                state[mask] *= -1.0
    
    return state


def _circuit_state(angles, n):
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0
    return _apply_circuit(state, angles, n)


def _residuals(params, n, coupling, field, target_state):
    angles = params.reshape((3, n))
    circuit_state = _circuit_state(angles, n)
    # Target is real
    return circuit_state - target_state


def _marginal_init(n, target_state):
    """Initialize angles based on marginal probabilities of the target state."""
    dim = 2 ** n
    # Probabilities
    probs = np.abs(target_state) ** 2
    
    inits = []
    # Zero init
    inits.append(np.zeros(3 * n))
    
    # Marginal init
    init = np.zeros((3, n))
    for i in range(n):
        # Sum probabilities where bit i is 1
        p1 = 0.0
        for basis in range(dim):
            if (basis >> i) & 1:
                p1 += probs[basis]
        # p1 = sin^2(theta/2) => theta = 2 arcsin(sqrt(p1))
        theta = 2 * np.arcsin(np.sqrt(np.clip(p1, 0, 1)))
        # Distribute across 3 layers
        init[:, i] = theta / 3.0
    inits.append(init.flatten())
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        rand_init = rng.uniform(-np.pi, np.pi, size=(3, n))
        inits.append(rand_init.flatten())
        
    return inits


def angles(n, coupling, field, layers):
    """Return circuit angles (layers, n) that approximate the ground state."""
    target_state, ground_energy = _exact_ground_state(n, coupling, field)
    
    inits = _marginal_init(n, target_state)
    
    best_params = None
    best_cost = np.inf
    
    for init in inits:
        try:
            result = least_squares(
                _residuals,
                init,
                args=(n, coupling, field, target_state),
                method='trf',
                max_nfev=2000,
                xtol=1e-14,
                ftol=1e-14,
                gtol=1e-14
            )
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_params = result.x
        except Exception:
            continue
    
    if best_params is None:
        return np.zeros((layers, n))
    
    return best_params.reshape((layers, n))
# EVOLVE-BLOCK-END
