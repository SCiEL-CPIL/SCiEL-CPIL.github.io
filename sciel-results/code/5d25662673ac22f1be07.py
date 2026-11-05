# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian H = -J sum Z_i Z_{i+1} - h sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=float)
    
    I = np.eye(2, dtype=float)
    X = np.array([[0, 1], [1, 0]], dtype=float)
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    
    def kron_list(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i+1] = Z
        H -= coupling * kron_list(ops)
        
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
        
    return H

def _precompute_circuit_indices(n, layers):
    """Precompute all index arrays needed for circuit state evolution."""
    dim = 1 << n
    all_indices = np.arange(dim)
    
    # Precompute Ry index pairs for each qubit
    ry_pairs = []
    for i in range(n):
        mask = 1 << i
        idx0 = all_indices[(all_indices & mask) == 0]
        idx1 = idx0 | mask
        ry_pairs.append((idx0, idx1))
    
    # Precompute CZ masks (boolean arrays)
    cz_masks = []
    for i in range(n - 1):
        mask_i = 1 << i
        mask_ip1 = 1 << (i + 1)
        both_set = (all_indices & mask_i) != 0
        both_set &= (all_indices & mask_ip1) != 0
        cz_masks.append(both_set)
    
    return ry_pairs, cz_masks

def _circuit_state_vectorized(theta, n, layers, precomp):
    """Compute the statevector of the circuit using precomputed indices."""
    dim = 1 << n
    state = np.zeros(dim, dtype=float)
    state[0] = 1.0
    ry_pairs, cz_masks = precomp
    
    for l in range(layers):
        for i in range(n):
            t = theta[l, i]
            c = np.cos(t / 2.0)
            s = np.sin(t / 2.0)
            idx0, idx1 = ry_pairs[i]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - s * old1
            state[idx1] = s * old0 + c * old1
        
        if l < layers - 1:
            for mask in cz_masks:
                state[mask] *= -1.0
            
    return state

def _energy_from_state(state, H):
    """Compute <H> for a real state vector using matrix multiplication."""
    return float(state @ (H @ state))

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])

def _energy_vectorized(thetas, n, H, layers, precomp):
    """Compute circuit energy using precomputed indices and matrix multiplication."""
    state = _circuit_state_vectorized(thetas, n, layers, precomp)
    return _energy_from_state(state, H)

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Build Hamiltonian once
    H = _build_hamiltonian(n, coupling, field)
    
    # Precompute exact ground energy
    e0 = _exact_ground_energy(n, coupling, field)
    
    # Precompute circuit indices
    precomp = _precompute_circuit_indices(n, layers)

    def obj(params):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            return 1e10
        thetas = params.reshape(layers, n)
        e = _energy_vectorized(thetas, n, H, layers, precomp)
        return (e - e0) / n

    # Generate diverse initializations
    n_params = layers * n
    inits = []
    
    # Mean-field like initialization: pi/2 for all
    inits.append(np.full(n_params, np.pi / 2.0))
    
    # All zeros
    inits.append(np.zeros(n_params))
    
    # Small angles (transverse field dominated)
    inits.append(np.full(n_params, np.pi / 4.0))
    
    # Large angles
    inits.append(np.full(n_params, 3.0 * np.pi / 4.0))
    
    # Random inits with fixed seed for reproducibility
    np.random.seed(42)
    for _ in range(12):
        inits.append(np.random.uniform(-np.pi, np.pi, size=n_params))
    
    # Symmetric inits: same angle for all qubits in a layer
    for _ in range(6):
        layer_angles = np.random.uniform(-np.pi, np.pi, size=layers)
        inits.append(np.repeat(layer_angles, n))
    
    # Alternating inits
    for _ in range(4):
        a = np.random.uniform(-np.pi, np.pi)
        b = np.random.uniform(-np.pi, np.pi)
        init = np.zeros(n_params)
        for l in range(layers):
            for i in range(n):
                init[l * n + i] = a if i % 2 == 0 else b
        inits.append(init)
    
    # Physical intuition: for Ising model, try angles that favor Z alignment
    # Small angles keep qubits near |0>, large angles rotate toward |1>
    for a in [0.3, 0.7, 1.1, 1.5]:
        init = np.zeros(n_params)
        for l in range(layers):
            for i in range(n):
                init[l * n + i] = a
        inits.append(init)
    
    # Sort by objective value
    init_objs = []
    for x in inits:
        init_objs.append((obj(x), x))
    init_objs.sort(key=lambda t: t[0])
    
    best_params = init_objs[0][1]
    best_val = init_objs[0][0]
    
    # Optimize from best initializations
    n_restarts = 10
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 25.0:
            break
        
        if r < len(init_objs):
            x_start = init_objs[r][1]
        else:
            x_start = init_objs[0][1] + np.random.randn(n_params) * 0.2
        
        try:
            res = minimize(
                obj,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        try:
            res = minimize(
                obj,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Second polish with Nelder-Mead if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 8.0:
        try:
            res = minimize(
                obj,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final ultra-precise polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 3.0:
        try:
            res = minimize(
                obj,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Ensure output is finite
    best_params = np.nan_to_num(best_params, nan=0.0, posinf=np.pi, neginf=-np.pi)
    
    return best_params.reshape(layers, n)
# EVOLVE-BLOCK-END
