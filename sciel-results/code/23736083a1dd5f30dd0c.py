# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian H = -coupling*sum Z_i Z_{i+1} - field*sum X_i."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Pauli X
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)
    
    def kron_list(ops):
        result = np.array([[1.0]])
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # ZZ terms: -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1}
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_list(ops)
    
    # X terms: -field * sum_{i=0}^{n-1} X_i
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)
    
    return H


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit of an n-qubit state."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    
    dim = 1 << n
    new_state = np.zeros_like(state)
    
    for idx in range(dim):
        bit = (idx >> qubit) & 1
        if bit == 0:
            idx1 = idx | (1 << qubit)
            new_state[idx] = c * state[idx] - s * state[idx1]
            new_state[idx1] = s * state[idx] + c * state[idx1]
    
    return new_state


def _apply_cz(state, q1, q2, n):
    """Apply CZ between qubits q1 and q2."""
    dim = 1 << n
    new_state = np.array(state, dtype=np.complex128)
    
    for idx in range(dim):
        b1 = (idx >> q1) & 1
        b2 = (idx >> q2) & 1
        if b1 == 1 and b2 == 1:
            new_state[idx] = -state[idx]
    
    return new_state


def _circuit_state(params, n, layers):
    """Prepare the circuit state from parameters."""
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    for layer in range(layers):
        for q in range(n):
            theta = params[layer * n + q]
            state = _apply_ry(state, theta, q, n)
        
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1, n)
    
    return state


def _energy(state, H):
    """Compute <state|H|state>."""
    return np.real(np.vdot(state, H @ state))


def _objective(params, n, coupling, field, layers, H, E0):
    """Objective: mean energy gap per spin."""
    state = _circuit_state(params, n, layers)
    E = _energy(state, H)
    return (E - E0) / n


def _gradient(params, n, coupling, field, layers, H, E0):
    """Compute gradient using parameter-shift rule for Ry gates."""
    state = _circuit_state(params, n, layers)
    E = _energy(state, H)
    grad = np.zeros_like(params)
    
    # For each Ry gate, compute gradient via parameter-shift rule
    # d/dtheta <psi|H|psi> = 0.5 * (<psi(theta+pi/2)|H|psi(theta+pi/2)> - <psi(theta-pi/2)|H|psi(theta-pi/2)>)
    for layer in range(layers):
        for q in range(n):
            idx = layer * n + q
            
            # Compute state with theta + pi/2
            params_plus = params.copy()
            params_plus[idx] += np.pi / 2
            state_plus = _circuit_state(params_plus, n, layers)
            E_plus = _energy(state_plus, H)
            
            # Compute state with theta - pi/2
            params_minus = params.copy()
            params_minus[idx] -= np.pi / 2
            state_minus = _circuit_state(params_minus, n, layers)
            E_minus = _energy(state_minus, H)
            
            grad[idx] = (E_plus - E_minus) / (2 * n)
    
    return grad


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build Hamiltonian and find exact ground state
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    E0 = eigvals[0]
    
    # Initial guess
    best_params = np.zeros(layers * n)
    best_val = _objective(best_params, n, coupling, field, layers, H, E0)
    
    # Try multiple initializations with diverse starting points
    np.random.seed(42)
    
    n_restarts = 12 if n == 4 else 6
    
    # Physics-motivated initializations
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros(layers * n))
    
    # 2. Mean-field inspired: small angles for transverse field
    mean_field_angle = np.arcsin(field / (field + coupling)) / 2 if (field + coupling) > 0 else 0
    initializations.append(mean_field_angle * np.ones(layers * n))
    
    # 3. Layer-dependent scaling
    for layer_scale in [0.5, 1.0, 1.5]:
        params = np.zeros(layers * n)
        for layer in range(layers):
            params[layer * n:(layer + 1) * n] = mean_field_angle * layer_scale
        initializations.append(params)
    
    # 4. Random initializations
    for _ in range(max(0, n_restarts - len(initializations))):
        initializations.append(np.random.uniform(-np.pi / 2, np.pi / 2, size=(layers * n)))
    
    for restart, params0 in enumerate(initializations[:n_restarts]):
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Use L-BFGS-B with analytical gradient
            result = minimize(
                _objective,
                params0,
                jac=_gradient,
                args=(n, coupling, field, layers, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective,
                best_params,
                jac=_gradient,
                args=(n, coupling, field, layers, H, E0),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 200}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(layers, n)

# EVOLVE-BLOCK-END
