"""Optimize shallow-circuit Ising ground states via multi-start local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 140.0  # seconds from start of angles() call

# Precompute exact ground energies and circuit-related quantities for the known cases.
# We'll cache results keyed by (n, coupling, field, layers).
_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the open transverse-field Ising Hamiltonian matrix.
    H = -coupling * sum_i Z_i Z_{i+1} - field * sum_i X_i
    Qubit i is bit i (little-endian: qubit 0 is LSB).
    """
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = 1.0 if bit_i == bit_ip1 else -1.0
            H[state, state] += -coupling * zz

    # X_i terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field

    return H


def _exact_ground_energy(n, coupling, field):
    """Exact ground state energy via diagonalization."""
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    return float(eigvals[0])


def _apply_ry(state, theta, qubit):
    """Apply Ry(theta) to qubit `qubit` of the state vector.
    Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
    Qubit `qubit` is bit `qubit` (LSB = qubit 0).
    """
    n = int(np.log2(len(state)))
    dim = len(state)
    new_state = np.zeros_like(state)

    cos_t = np.cos(theta / 2.0)
    sin_t = np.sin(theta / 2.0)

    for state_idx in range(dim):
        bit = (state_idx >> qubit) & 1
        # The partner index differs only in bit `qubit`
        partner = state_idx ^ (1 << qubit)
        if bit == 0:
            # |0> -> cos|0> + sin|1>
            new_state[partner] += cos_t * state[state_idx]
            new_state[state_idx] += sin_t * state[state_idx]
        else:
            # |1> -> -sin|0> + cos|1>
            new_state[state_idx] += cos_t * state[state_idx]
            new_state[partner] += -sin_t * state[state_idx]

    return new_state


def _apply_cz(state, q1, q2):
    """Apply CZ gate between qubits q1 and q2.
    CZ flips phase when both qubits are |1>.
    """
    new_state = state.copy()
    for state_idx in range(len(state)):
        if ((state_idx >> q1) & 1) and ((state_idx >> q2) & 1):
            new_state[state_idx] *= -1.0
    return new_state


def _circuit_state(params, n, layers):
    """Compute the state produced by the circuit.
    params: array of shape (layers, n)
    Start from |0>^n.
    Each layer: apply Ry(theta) to every qubit, then CZ on adjacent pairs (ascending order).
    No CZ after the final layer.
    """
    dim = 1 << n
    # Start from |0>^n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            state = _apply_ry(state, params[layer, q], q)

        # Apply CZ between adjacent pairs, ascending order
        if layer < layers - 1:
            for q in range(n - 1):
                state = _apply_cz(state, q, q + 1)

    return state


def _circuit_energy(params, n, coupling, field, H):
    """Compute <psi|H|psi> for the circuit state."""
    state = _circuit_state(params, n, layers=3)
    # Energy = real part of psi^dagger H psi
    Hpsi = H @ state
    energy = np.real(np.vdot(state, Hpsi))
    return float(energy)


def _objective(params_flat, n, coupling, field, H, ground_energy):
    """Objective: (circuit_energy - ground_energy) / n. Lower is better."""
    params = params_flat.reshape(3, n)
    energy = _circuit_energy(params, n, coupling, field, H)
    return (energy - ground_energy) / n


def _initial_guess(n, field, coupling):
    """Physics-based initialization.
    For small field, ground state is close to |0...0> with small mixing.
    For large field, ground state is close to |+>^n.
    Intermediate: use a blend.
    """
    # Ry(theta) on |0> gives cos(theta/2)|0> + sin(theta/2)|1>
    # For |0>^n: all thetas = 0
    # For |+>^n: all thetas = pi/2
    # For the transverse field, each qubit wants to be in a superposition.
    # The local field effective on each qubit depends on neighbors.
    # Simple heuristic: theta proportional to arctan(field/coupling) roughly.
    # Actually, for a single qubit with H = -J Z - h X, the ground state angle is...
    # Let's just use a few different initializations.

    theta = np.zeros((3, n))

    # Layer 0: small rotation to create some superposition
    theta[0, :] = 0.3 * np.arctan(field / max(coupling, 0.01))
    # Layer 1: adjust based on coupling
    theta[1, :] = 0.2 * np.arctan(field / max(coupling, 0.01))
    # Layer 2: larger rotation toward |+>
    theta[2, :] = np.pi / 4.0 * (field / max(field + coupling, 0.01))

    return theta


def _random_init(n, scale=0.5):
    """Random initialization within a reasonable range."""
    return np.random.uniform(-scale, scale, size=(3, n))


def _optimize_single(n, coupling, field, H, ground_energy, x0, time_budget):
    """Run a single optimization from initial point x0."""
    start_time = time.time()
    elapsed = start_time - _start_time

    def obj(x):
        if time.time() - _start_time > _DEADLINE - 5:
            return 1e10  # signal to stop
        return _objective(x, n, coupling, field, H, ground_energy)

    try:
        res = minimize(
            obj,
            x0.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True,
            }
        )
        return res.x, res.fun
    except Exception:
        return x0.flatten(), _objective(x0.flatten(), n, coupling, field, H, ground_energy)


_start_time = 0.0


def angles(n, coupling, field, layers):
    """Main entry point: return optimized angles (layers, n)."""
    global _start_time
    _start_time = time.time()

    key = (n, coupling, field, layers)
    if key in _cache:
        return _cache[key].copy()

    # Precompute Hamiltonian and ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)

    dim = 1 << n
    # For n=6, dim=64, matrix ops are fast.

    best_params = _initial_guess(n, field, coupling)
    best_score = _objective(best_params.flatten(), n, coupling, field, H, ground_energy)

    # Multi-start optimization
    np.random.seed(42)  # deterministic for reproducibility within a run

    # Generate diverse initial points
    inits = [best_params.copy()]

    # Physics-based variants
    for scale in [0.1, 0.3, 0.5, 0.8, 1.0]:
        init = np.zeros((3, n))
        init[0, :] = scale * np.arctan(field / max(coupling, 0.01))
        init[1, :] = 0.5 * scale * np.arctan(field / max(coupling, 0.01))
        init[2, :] = np.pi / 2.0 * scale * (field / max(field + coupling, 0.01))
        inits.append(init)

    # Random inits
    for _ in range(10):
        inits.append(_random_init(n, scale=1.0))

    # Near-|+> init
    plus_init = np.zeros((3, n))
    plus_init[2, :] = np.pi / 2.0
    inits.append(plus_init)

    # Near-|0> init
    zero_init = np.zeros((3, n))
    inits.append(zero_init)

    # Run optimizations
    for i, init in enumerate(inits):
        if time.time() - _start_time > _DEADLINE - 10:
            break

        x_opt, score = _optimize_single(n, coupling, field, H, ground_energy, init, _DEADLINE)

        if score < best_score:
            best_score = score
            best_params = x_opt.reshape(3, n)

        # If we found a very good solution, do a polish pass
        if best_score < 1e-6:
            # Tight polish
            if time.time() - _start_time < _DEADLINE - 15:
                x_polish, score_polish = _optimize_single(
                    n, coupling, field, H, ground_energy,
                    best_params, _DEADLINE
                )
                if score_polish < best_score:
                    best_score = score_polish
                    best_params = x_polish.reshape(3, n)

    # Final polish if time allows
    if time.time() - _start_time < _DEADLINE - 5:
        try:
            def obj_tight(x):
                if time.time() - _start_time > _DEADLINE - 2:
                    return 1e10
                return _objective(x, n, coupling, field, H, ground_energy)

            res = minimize(
                obj_tight,
                best_params.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            if res.fun < best_score:
                best_params = res.x.reshape(3, n)
                best_score = res.fun
        except Exception:
            pass

    # Ensure finite values
    if not np.all(np.isfinite(best_params)):
        best_params = np.zeros((layers, n))

    _cache[key] = best_params.copy()
    return best_params
# EVOLVE-BLOCK-END
