import numpy as np
from scipy.optimize import minimize
import time

def _build_circuit_state(theta, n, layers):
    """Build the state vector from the circuit parameters using vectorized ops."""
    dim = 2 ** n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0  # |0>^n
    
    # Precompute bit masks
    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            t = theta[layer, q]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            # Vectorized: split state by bit q
            mask = np.zeros(dim, dtype=bool)
            for idx in range(dim):
                if (idx >> q) & 1:
                    mask[idx] = True
            idx0 = np.where(~mask)[0]
            idx1 = np.where(mask)[0]
            v0 = state[idx0].copy()
            v1 = state[idx1].copy()
            state[idx0] = c * v0 - s * v1
            state[idx1] = s * v0 + c * v1
        
        # Apply CZ between this layer and next (if not last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ: phase -1 when both qubits are 1
                mask = np.zeros(dim, dtype=bool)
                for idx in range(dim):
                    if ((idx >> q) & 1) and ((idx >> (q + 1)) & 1):
                        mask[idx] = True
                state[mask] *= -1.0
    
    return state

def _compute_energy(state, n, coupling, field):
    """Compute expectation value of H = -coupling*sum Z_i Z_(i+1) - field*sum X_i."""
    dim = 2 ** n
    probs = np.abs(state) ** 2
    energy = 0.0
    
    # Z_i Z_(i+1) terms
    for i in range(n - 1):
        zz = 0.0
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            zz += probs[idx] * zi * zip1
        energy += -coupling * zz
    
    # X_i terms
    for i in range(n):
        x_val = 0.0 + 0.0j
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            x_val += np.conj(state[idx]) * state[flipped]
        energy += -field * np.real(x_val)
    
    return np.real(energy)

def _get_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    
    # Z_i Z_(i+1) terms (diagonal)
    for i in range(n - 1):
        for idx in range(dim):
            bit_i = (idx >> i) & 1
            bit_ip1 = (idx >> (i + 1)) & 1
            zi = 1.0 if bit_i == 0 else -1.0
            zip1 = 1.0 if bit_ip1 == 0 else -1.0
            H[idx, idx] += -coupling * zi * zip1
    
    # X_i terms (off-diagonal)
    for i in range(n):
        for idx in range(dim):
            flipped = idx ^ (1 << i)
            H[idx, flipped] += -field
    
    eigvals = np.linalg.eigvalsh(H)
    return np.min(eigvals)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    # Get ground energy
    ground_energy = _get_ground_energy(n, coupling, field)
    
    def objective(params):
        theta = params.reshape((layers, n))
        state = _build_circuit_state(theta, n, layers)
        energy = _compute_energy(state, n, coupling, field)
        return (energy - ground_energy) / n
    
    best_result = None
    best_val = np.inf
    
    # Generate multiple starting points
    starts = []
    # All zeros
    starts.append(np.zeros(layers * n))
    # pi/2 in last layer
    x0 = np.zeros(layers * n)
    x0[(layers-1)*n:layers*n] = np.pi / 2
    starts.append(x0)
    # Several random starts
    np.random.seed(42)
    for _ in range(6):
        starts.append(np.random.uniform(-np.pi, np.pi, size=layers * n))
    
    for x0 in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                          bounds=[(-np.pi, np.pi)] * (layers * n),
                          options={'maxiter': 3000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x.copy()
        except Exception:
            pass
    
    if best_result is None:
        theta = np.zeros((layers, n))
        theta[-1] = np.pi / 2
        return theta
    
    return best_result.reshape((layers, n))
