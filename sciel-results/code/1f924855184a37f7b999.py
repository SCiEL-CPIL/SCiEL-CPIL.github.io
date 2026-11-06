"""Optimized program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_idx_cache = {}


def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = float(eigvals[0])
    _e0_cache[key] = e0
    return e0


def _get_indices(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    qubit_indices = []
    for q in range(n):
        mask = 1 << q
        idx_0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx_1 = idx_0 + mask
        qubit_indices.append((idx_0, idx_1))
    cz_masks = []
    all_idx = np.arange(dim)
    for i in range(n - 1):
        m1 = 1 << i
        m2 = 1 << (i + 1)
        both_set = ((all_idx & m1) != 0) & ((all_idx & m2) != 0)
        cz_masks.append(both_set)
    _idx_cache[n] = (qubit_indices, cz_masks)
    return _idx_cache[n]


def _circuit_state(n, thetas, layers):
    dim = 1 << n
    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0
    qubit_indices, cz_masks = _get_indices(n)
    for layer in range(layers):
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = qubit_indices[q]
            vals_0 = state[idx_0].copy()
            vals_1 = state[idx_1].copy()
            state[idx_0] = c * vals_0 - s * vals_1
            state[idx_1] = s * vals_0 + c * vals_1
        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0
    return state


def _circuit_energy_and_grad(n, coupling, field, thetas, layers):
    """
    Compute energy and gradient of <psi|H|psi> w.r.t. all angles.
    Uses the chain rule: for each parameter, compute dpsi/dtheta and
    grad = 2*Re(<psi|H|dpsi/dtheta>)
    """
    H = _build_H(n, coupling, field)
    dim = 1 << n
    qubit_indices, cz_masks = _get_indices(n)

    # Forward pass: store states after each layer's Ry (before CZ)
    # states_after_ry[layer] = state after applying all Ry in that layer
    # states_after_cz[layer] = state after CZ (if applicable)
    states_after_ry = []
    states_before_cz = []

    state = np.zeros(dim, dtype=np.float64)
    state[0] = 1.0

    for layer in range(layers):
        # Apply Ry to each qubit
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            idx_0, idx_1 = qubit_indices[q]
            vals_0 = state[idx_0].copy()
            vals_1 = state[idx_1].copy()
            state[idx_0] = c * vals_0 - s * vals_1
            state[idx_1] = s * vals_0 + c * vals_1
        states_after_ry.append(state.copy())

        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0
            states_before_cz.append(state.copy())
        else:
            states_before_cz.append(state.copy())

    psi = state
    energy = float(np.dot(psi, H @ psi))

    # Compute H @ psi once
    Hpsi = H @ psi

    # Gradient computation
    # For each (layer, qubit), we need dpsi/dtheta_{layer,q}
    # The derivative propagates forward through subsequent gates.
    #
    # Key insight: Since all gates are unitary and we start from |0>,
    # dpsi/dtheta = U_final ... U_{layer+1} * (dRy_q/dtheta * U_layer_before_q) ... |0>
    #
    # More efficiently: compute the "adjoint" by working backwards.
    # Let chi = H @ psi. Then dE/dtheta = 2*Re(<chi|dpsi/dtheta>)
    #
    # For a single Ry gate on qubit q at layer l:
    # dRy/dtheta = 0.5 * [[-sin(theta/2), -cos(theta/2)],
    #                      [ cos(theta/2), -sin(theta/2)]]
    #
    # We can compute dpsi/dtheta by forward-propagating the derivative.
    # But a more efficient approach: use the fact that the circuit is
    # a sequence of gates, and compute the gradient via the "parameter
    # shift" or direct differentiation.

    # Direct approach: for each parameter, compute the derivative state
    # by re-running the circuit with the derivative at that gate.
    # This is O(layers * n * dim * n) which is acceptable for small dim.

    grad = np.zeros((layers, n), dtype=np.float64)

    for l in range(layers):
        for q in range(n):
            theta = thetas[l, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            # Derivative of Ry(theta) w.r.t. theta:
            # dRy/dtheta = 0.5 * [[-s, -c], [c, -s]]
            # So if we have a state |psi> = a|0> + b|1> on qubit q,
            # d|psi>/dtheta = 0.5 * (-s*a - c*b)|0> + 0.5*(c*a - s*b)|1>

            # Forward propagate the derivative from layer l, qubit q
            # We need the state just before applying Ry(l,q)
            # and then propagate the derivative forward through all remaining gates.

            # Reconstruct state before layer l
            state_before_l = np.zeros(dim, dtype=np.float64)
            state_before_l[0] = 1.0

            for ll in range(l):
                for qq in range(n):
                    th = thetas[ll, qq]
                    cc = np.cos(th / 2.0)
                    ss = np.sin(th / 2.0)
                    idx_0, idx_1 = qubit_indices[qq]
                    v0 = state_before_l[idx_0].copy()
                    v1 = state_before_l[idx_1].copy()
                    state_before_l[idx_0] = cc * v0 - ss * v1
                    state_before_l[idx_1] = ss * v0 + cc * v1
                if ll < layers - 1:
                    for i in range(n - 1):
                        state_before_l[cz_masks[i]] *= -1.0

            # Now apply Ry for qubits 0..q-1 in layer l
            for qq in range(q):
                th = thetas[l, qq]
                cc = np.cos(th / 2.0)
                ss = np.sin(th / 2.0)
                idx_0, idx_1 = qubit_indices[qq]
                v0 = state_before_l[idx_0].copy()
                v1 = state_before_l[idx_1].copy()
                state_before_l[idx_0] = cc * v0 - ss * v1
                state_before_l[idx_1] = ss * v0 + cc * v1

            # Now state_before_l is the state just before Ry(l, q)
            # Compute derivative state at this point
            idx_0, idx_1 = qubit_indices[q]
            v0 = state_before_l[idx_0].copy()
            v1 = state_before_l[idx_1].copy()

            # d|psi>/dtheta = 0.5 * [(-s*v0 - c*v1), (c*v0 - s*v1)]
            dv0 = 0.5 * (-s * v0 - c * v1)
            dv1 = 0.5 * (c * v0 - s * v1)

            # The derivative state: replace the qubit-q components
            deriv_state = np.zeros(dim, dtype=np.float64)
            # Copy all other components from state_before_l
            # For qubit q, use the derivative values
            # We need to be careful: for each basis state, the amplitude
            # depends on the value of qubit q.

            # For indices where qubit q is 0: amplitude = dv0 * (rest)
            # For indices where qubit q is 1: amplitude = dv1 * (rest)
            # But we need to extract the "rest" amplitudes.

            # Simpler: build the full derivative state
            # For each basis state |x>, the amplitude is:
            # if x_q = 0: dv0 * <x'|psi_before> where x' has qubit q set to 0
            # if x_q = 1: dv1 * <x'|psi_before> where x' has qubit q set to 1
            # But this isn't quite right because the state is a superposition.

            # Actually, the derivative state is:
            # |dpsi> = sum_x c_x |x> where c_x = d< x | psi > / dtheta
            # For a Ry gate on qubit q:
            # <x|dpsi/dtheta> = 0.5 * (-s*<x|psi> - c*<x_flip_q|psi>) if x_q=0
            #                   = 0.5 * (c*<x_flip_q|psi> - s*<x|psi>) if x_q=1
            # where x_flip_q is x with qubit q flipped.

            # Let me just construct it directly:
            deriv_state[idx_0] = dv0  # This is the amplitude for states with qubit q = 0
            # Wait, this isn't right either. The state is a vector over all 2^n basis states.
            # The Ry gate acts on the qubit-q subspace. For a fixed value of all other qubits,
            # the 2-dim subspace (q=0, q=1) is rotated.

            # Let me think again. The state vector has components psi[x] for x in 0..2^n-1.
            # After Ry on qubit q:
            # psi_new[x] = c*psi_old[x] - s*psi_old[x^mask] if x_q=0
            # psi_new[x] = s*psi_old[x^mask] + c*psi_old[x] if x_q=1
            # where mask = 1<<q

            # So dpsi_new[x]/dtheta = 0.5*(-s*psi_old[x] - c*psi_old[x^mask]) if x_q=0
            #                       = 0.5*(c*psi_old[x^mask] - s*psi_old[x]) if x_q=1

            # So I need psi_old (state before Ry(l,q)) and can compute the derivative.
            mask = 1 << q
            for x in range(dim):
                if (x & mask) == 0:
                    deriv_state[x] = 0.5 * (-s * state_before_l[x] - c * state_before_l[x | mask])
                else:
                    deriv_state[x] = 0.5 * (c * state_before_l[x ^ mask] - s * state_before_l[x])

            # Now propagate deriv_state forward through remaining gates
            # Apply Ry for qubits q+1..n-1 in layer l
            for qq in range(q + 1, n):
                th = thetas[l, qq]
                cc = np.cos(th / 2.0)
                ss = np.sin(th / 2.0)
                idx_0, idx_1 = qubit_indices[qq]
                v0 = deriv_state[idx_0].copy()
                v1 = deriv_state[idx_1].copy()
                deriv_state[idx_0] = cc * v0 - ss * v1
                deriv_state[idx_1] = ss * v0 + cc * v1

            # Apply CZ if not last layer
            if l < layers - 1:
                for i in range(n - 1):
                    deriv_state[cz_masks[i]] *= -1.0

            # Apply all gates in layers l+1..layers-1
            for ll in range(l + 1, layers):
                for qq in range(n):
                    th = thetas[ll, qq]
                    cc = np.cos(th / 2.0)
                    ss = np.sin(th / 2.0)
                    idx_0, idx_1 = qubit_indices[qq]
                    v0 = deriv_state[idx_0].copy()
                    v1 = deriv_state[idx_1].copy()
                    deriv_state[idx_0] = cc * v0 - ss * v1
                    deriv_state[idx_1] = ss * v0 + cc * v1
                if ll < layers - 1:
                    for i in range(n - 1):
                        deriv_state[cz_masks[i]] *= -1.0

            # grad = 2 * Re(<Hpsi | deriv_state>)
            grad[l, q] = 2.0 * np.dot(Hpsi, deriv_state)

    return energy, grad


def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n


def _objective_and_grad(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    energy, grad = _circuit_energy_and_grad(n, coupling, field, thetas, layers)
    val = (energy - e0) / n
    g = grad / n
    return val, g.flatten()


def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state(n, thetas, layers)
    return float(np.dot(psi, H @ psi))


def _mean_field_init(n, coupling, field, layers):
    """
    Mean-field inspired initialization for TFIM.
    For a single spin in field h with coupling J, the optimal angle
    satisfies tan(theta) ~ h/J. We use this as a starting point.
    """
    theta_mf = np.arctan2(field, coupling)
    # The circuit starts at |0> (Z+). Ry(theta) rotates towards X.
    # For the ground state, we want to be aligned with the effective field.
    # A reasonable initialization: all angles = pi/2 - theta_mf
    init = np.zeros(layers * n)
    for l in range(layers):
        for q in range(n):
            init[l * n + q] = np.pi / 2 - theta_mf
    return init


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    e0 = _exact_ground_energy(n, coupling, field)

    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)

    _get_indices(n)

    if n <= 4:
        num_restarts = 30
    else:
        num_restarts = 20

    def obj(p):
        return _objective(p, n, coupling, field, layers, e0)

    def obj_grad(p):
        return _objective_and_grad(p, n, coupling, field, layers, e0)

    # Generate diverse initializations
    inits = []
    # 0: zeros
    inits.append(np.zeros(layers * n))
    # 1: all pi/2
    inits.append(np.full(layers * n, np.pi / 2))
    # 2: all -pi/2
    inits.append(np.full(layers * n, -np.pi / 2))
    # 3: alternating
    inits.append(np.array([np.pi / 2 if (j % 2 == 0) else -np.pi / 2 for j in range(layers * n)]))
    # 4: first layer pi/2
    x4 = np.zeros(layers * n)
    x4[:n] = np.pi / 2
    inits.append(x4)
    # 5: last layer pi/2
    x5 = np.zeros(layers * n)
    x5[-n:] = np.pi / 2
    inits.append(x5)
    # 6: small angles
    inits.append(np.full(layers * n, 0.1))
    # 7: pi/3
    inits.append(np.full(layers * n, np.pi / 3))
    # 8: increasing across layers
    x8 = np.zeros(layers * n)
    for l in range(layers):
        x8[l * n:(l + 1) * n] = (l + 1) * np.pi / (layers + 1)
    inits.append(x8)
    # 9: alternating layers
    x9 = np.zeros(layers * n)
    for l in range(layers):
        if l % 2 == 0:
            x9[l * n:(l + 1) * n] = np.pi / 2
    inits.append(x9)
    # 10: mean field
    inits.append(_mean_field_init(n, coupling, field, layers))
    # 11-19: random
    for i in range(10):
        np.random.seed(2000 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))

    # Optimize from each initialization
    for i, x_init in enumerate(inits):
        if time.time() > deadline:
            break
        if i == 0:
            continue  # already evaluated

        try:
            res = minimize(obj, x_init, method='L-BFGS-B', jac=obj_grad,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            try:
                res = minimize(obj, x_init, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

    # Final polish with gradient
    if time.time() < deadline - 20.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B', jac=obj_grad,
                          options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 100})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second polish: Nelder-Mead for robustness
    if time.time() < deadline - 10.0:
        try:
            res = minimize(obj, best_x, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Third polish: L-BFGS-B with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res = minimize(obj, best_x, method='L-BFGS-B', jac=obj_grad,
                          options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 200})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
