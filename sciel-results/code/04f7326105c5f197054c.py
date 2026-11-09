"""Rewritten program: optimize 3-layer Ry/CZ circuit for Ising ground states."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 150.0  # wall-clock limit
_MARGIN = 5.0      # safety margin


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _apply_ry(state, theta, qubit, n):
    """Apply Ry(theta) to qubit. Returns new state (in-place style via copy)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    # Ry = [[c, -s], [s, c]] in computational basis
    # For qubit q, we need to reshape and apply
    shape = [2] * n
    idx = list(range(n))
    # Move target qubit to first axis
    perm = [qubit] + [i for i in range(n) if i != qubit]
    state_t = np.transpose(state, perm)
    state_t = state_t.reshape(2, -1)
    new = np.empty_like(state_t)
    new[0] = c * state_t[0] - s * state_t[1]
    new[1] = s * state_t[0] + c * state_t[1]
    new = new.reshape(shape)
    inv_perm = [0] * n
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return np.transpose(new, inv_perm)


def _apply_cz(state, a, b, n):
    """Apply CZ between qubits a and b."""
    # CZ flips phase of |11> component
    # Build mask for states where both qubits are 1
    mask = np.zeros(2 ** n, dtype=np.float64)
    for k in range(2 ** n):
        if (k >> a) & 1 and (k >> b) & 1:
            mask[k] = -1.0
        else:
            mask[k] = 1.0
    return state * mask


def _circuit_state(theta, n, layers, coupling, field):
    """Build state vector for given circuit parameters."""
    # Start |0>^n
    state = np.zeros(2 ** n, dtype=np.float64)
    state[0] = 1.0

    # Precompute CZ masks for each adjacent pair
    cz_masks = []
    for i in range(n - 1):
        mask = np.ones(2 ** n, dtype=np.float64)
        for k in range(2 ** n):
            if (k >> i) & 1 and (k >> (i + 1)) & 1:
                mask[k] = -1.0
        cz_masks.append(mask)

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, theta[layer, q], q, n)
        # Apply CZ between adjacent pairs (ascending order), except after final layer
        if layer < layers - 1:
            for i in range(n - 1):
                state = state * cz_masks[i]
    return state


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    dim = 2 ** n
    # Build Hamiltonian H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # Z = diag(1, -1), X = [[0,1],[1,0]]
    Z = np.diag([1.0, -1.0])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])

    H = np.zeros((dim, dim), dtype=np.float64)

    # Identity
    I = np.eye(2)

    def kron_list(ops):
        """Kronecker product of list of 2x2 matrices."""
        result = np.array([[1.0]])
        for op in ops:
            result = np.kron(result, op)
        return result

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        ops = [I] * n
        ops[i] = Z
        ops[i + 1] = Z
        H -= coupling * kron_list(ops)

    # X_i terms
    for i in range(n):
        ops = [I] * n
        ops[i] = X
        H -= field * kron_list(ops)

    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _energy(state, n, coupling, field):
    """Compute <state|H|state> for the Ising Hamiltonian."""
    dim = 2 ** n
    # Build Hamiltonian (cached per (n, coupling, field))
    key = (n, coupling, field)
    if not hasattr(_energy, '_cache'):
        _energy._cache = {}
    if key not in _energy._cache:
        I = np.eye(2)
        Z = np.diag([1.0, -1.0])
        X = np.array([[0.0, 1.0], [1.0, 0.0]])

        def kron_list(ops):
            result = np.array([[1.0]])
            for op in ops:
                result = np.kron(result, op)
            return result

        H = np.zeros((dim, dim), dtype=np.float64)
        for i in range(n - 1):
            ops = [I] * n
            ops[i] = Z
            ops[i + 1] = Z
            H -= coupling * kron_list(ops)
        for i in range(n):
            ops = [I] * n
            ops[i] = X
            H -= field * kron_list(ops)
        _energy._cache[key] = H

    H = _energy._cache[key]
    return np.real(np.vdot(state, H @ state))


def _energy_and_grad(theta, n, layers, coupling, field):
    """Compute energy and gradient via finite differences or analytical."""
    state = _circuit_state(theta, n, layers, coupling, field)
    e = _energy(state, n, coupling, field)

    # Analytical gradient: dE/dtheta_{l,q} = 2 * Re[ <psi| d|psi>/dtheta * H ]
    # We'll use forward differences for simplicity but with small step
    # Actually, let's do a proper analytical gradient by tracking the state evolution
    # For efficiency, use central differences with small step
    grad = np.zeros_like(theta)
    eps = 1e-7
    for l in range(layers):
        for q in range(n):
            theta_plus = theta.copy()
            theta_plus[l, q] += eps
            theta_minus = theta.copy()
            theta_minus[l, q] -= eps
            sp = _circuit_state(theta_plus, n, layers, coupling, field)
            sm = _circuit_state(theta_minus, n, layers, coupling, field)
            ep = _energy(sp, n, coupling, field)
            em = _energy(sm, n, coupling, field)
            grad[l, q] = (ep - em) / (2 * eps)
    return e, grad


def _objective(theta, n, layers, coupling, field, ground_e):
    """Objective: (circuit energy - ground energy) / n"""
    e = _energy(_circuit_state(theta, n, layers, coupling, field), n, coupling, field)
    return (e - ground_e) / n


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to approximate ground state."""
    dim = 2 ** n
    ground_e = _exact_ground_energy(n, coupling, field)

    # Initial guess: try a few starting points
    best_theta = None
    best_obj = np.inf

    # Starting points:
    # 1. All zeros
    # 2. All pi/4
    # 3. Alternating
    # 4. Random
    starts = [
        np.zeros((layers, n)),
        np.full((layers, n), np.pi / 4),
        np.full((layers, n), np.pi / 8),
        np.random.uniform(-np.pi/4, np.pi/4, (layers, n)),
    ]

    for start in starts:
        if _time_left() < 10.0:
            break

        # Use L-BFGS-B with analytical-like gradient (finite diff)
        def obj_func(x):
            return _objective(x.reshape(layers, n), n, layers, coupling, field, ground_e)

        def grad_func(x):
            e, g = _energy_and_grad(x.reshape(layers, n), n, layers, coupling, field)
            return g.ravel()

        x0 = start.ravel()
        try:
            res = minimize(obj_func, x0, jac=grad_func, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < best_obj:
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_theta = res.x.reshape(layers, n)
        except Exception:
            pass

    # If no optimization succeeded, fall back to a reasonable default
    if best_theta is None:
        best_theta = np.zeros((layers, n))
        best_theta[-1] = np.pi / 2

    # Final polish: try a few more random restarts if time permits
    if _time_left() > 15.0:
        for _ in range(5):
            if _time_left() < 10.0:
                break
            x0 = np.random.uniform(-np.pi/2, np.pi/2, (layers, n)).ravel()

            def obj_func(x):
                return _objective(x.reshape(layers, n), n, layers, coupling, field, ground_e)

            def grad_func(x):
                e, g = _energy_and_grad(x.reshape(layers, n), n, layers, coupling, field)
                return g.ravel()

            try:
                res = minimize(obj_func, x0, jac=grad_func, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_theta = res.x.reshape(layers, n)
            except Exception:
                pass

    return best_theta
# EVOLVE-BLOCK-END
