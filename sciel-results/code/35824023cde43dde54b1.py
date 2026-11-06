"""QAOA parameter design: per-graph optimization of two-layer angles."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 120.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)


def _build_hamiltonian(n, edges):
    """Build the MaxCut cost Hamiltonian C = sum_edges (1 - Z_i Z_j)/2."""
    C = np.zeros((n, n), dtype=float)
    for (i, j) in edges:
        C[i, j] = 1.0
        C[j, i] = 1.0
    return C


def _expected_cut(n, edges, gamma, beta):
    """Compute expected cut value for given QAOA angles using state-vector simulation."""
    # Start in |+>^n
    state = np.ones(2 ** n, dtype=complex)
    state /= np.sqrt(2 ** n)

    C = _build_hamiltonian(n, edges)

    # Build cost operator in computational basis
    # C = sum_edges (1 - Z_i Z_j)/2
    # (1 - Z_i Z_j)/2 = 0 if bits equal, 1 if bits differ
    basis = np.arange(2 ** n)
    cost_values = np.zeros(2 ** n, dtype=float)
    for (i, j) in edges:
        mask_i = (1 << i)
        mask_j = (1 << j)
        diff = ((basis & mask_i) != (basis & mask_j))
        cost_values += diff.astype(float)
    cost_values *= 0.5  # (1 - Z_i Z_j)/2

    # Apply exp(-i gamma C) then exp(-i beta sum X) for each layer
    # C is diagonal in computational basis
    for _ in range(2):
        # Cost step: diagonal
        state = state * np.exp(-1j * gamma * cost_values)
        # Mixing step: exp(-i beta sum_j X_j)
        # sum_j X_j flips each bit
        # In computational basis, X_j maps |b> to |b with bit j flipped>
        # exp(-i beta * sum_j X_j) is a product of exp(-i beta X_j)
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # So exp(-i beta sum X) = prod_j (cos(beta) I - i sin(beta) X_j)
        # Apply each X_j:
        for j in range(n):
            mask = 1 << j
            flipped = basis ^ mask
            # New state: for each basis state b, contribution from b and b^mask
            # (cos(beta) I - i sin(beta) X_j)|b> = cos(beta)|b> - i sin(beta)|b^mask>
            new_state = np.zeros_like(state)
            for b in range(2 ** n):
                new_state[b] += cos_beta * state[b]
                new_state[b ^ mask] += (-1j * sin_beta) * state[b]
            # This is O(2^n * n) per layer; do it more efficiently
            break
        # More efficient: apply via permutation
        # X_j is a permutation matrix
        # new_state[b] = cos(beta)*state[b] - i sin(beta)*state[b^mask]
        new_state = np.zeros_like(state)
        for j in range(n):
            mask = 1 << j
            # new_state[b] = cos(beta)*state[b] - i sin(beta)*state[b^mask]
            # We need to accumulate over all j
        # Actually apply sequentially
        for j in range(n):
            mask = 1 << j
            new_state = np.zeros_like(state)
            for b in range(2 ** n):
                new_state[b] = cos_beta * state[b] - 1j * sin_beta * state[b ^ mask]
            state = new_state

    # Expected cut = <state|C|state> = sum_b |state_b|^2 * cost_values[b]
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cost_values))


def _expected_cut_fast(n, edges, gamma, beta):
    """Fast expected cut using diagonal cost and permutation-based mixing."""
    C = _build_hamiltonian(n, edges)
    basis = np.arange(2 ** n)

    # Precompute cost values
    cost_values = np.zeros(2 ** n, dtype=float)
    for (i, j) in edges:
        mask_i = (1 << i)
        mask_j = (1 << j)
        diff = ((basis & mask_i) != (basis & mask_j))
        cost_values += diff.astype(float)
    cost_values *= 0.5

    state = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    for _ in range(2):
        # Cost step
        state = state * np.exp(-1j * gamma * cost_values)
        # Mixing step
        cb = np.cos(beta)
        sb = np.sin(beta)
        for j in range(n):
            mask = 1 << j
            new_state = np.zeros_like(state)
            for b in range(2 ** n):
                new_state[b] = cb * state[b] - 1j * sb * state[b ^ mask]
            state = new_state

    probs = np.abs(state) ** 2
    return float(np.sum(probs * cost_values))


def _objective(params, n, edges):
    """Negative expected cut for minimization."""
    gamma1, beta1, gamma2, beta2 = params
    # Layer 1
    C = _build_hamiltonian(n, edges)
    basis = np.arange(2 ** n)
    cost_values = np.zeros(2 ** n, dtype=float)
    for (i, j) in edges:
        mask_i = (1 << i)
        mask_j = (1 << j)
        diff = ((basis & mask_i) != (basis & mask_j))
        cost_values += diff.astype(float)
    cost_values *= 0.5

    state = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)

    # Layer 1 cost
    state = state * np.exp(-1j * gamma1 * cost_values)
    cb1 = np.cos(beta1)
    sb1 = np.sin(beta1)
    for j in range(n):
        mask = 1 << j
        new_state = np.zeros_like(state)
        for b in range(2 ** n):
            new_state[b] = cb1 * state[b] - 1j * sb1 * state[b ^ mask]
        state = new_state

    # Layer 2 cost
    state = state * np.exp(-1j * gamma2 * cost_values)
    cb2 = np.cos(beta2)
    sb2 = np.sin(beta2)
    for j in range(n):
        mask = 1 << j
        new_state = np.zeros_like(state)
        for b in range(2 ** n):
            new_state[b] = cb2 * state[b] - 1j * sb2 * state[b ^ mask]
        state = new_state

    probs = np.abs(state) ** 2
    return -float(np.sum(probs * cost_values))


def angles(n, edges, p):
    """Optimize QAOA angles for the given graph."""
    # Default fallback
    default = np.array([[0.35, 0.25]] * p)

    if n > 12:
        return default

    # Try to optimize
    x0 = np.array([0.35, 0.25, 0.35, 0.25])
    bounds = [(0, np.pi)] * 4

    try:
        if _remaining() < 5:
            return default

        result = minimize(_objective, x0, args=(n, edges), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5,
                                   'adaptive': True})
        if result.success or result.fun < _objective(x0, n, edges):
            best = result.x
        else:
            best = x0
    except Exception:
        best = x0

    # Convert to (p, 2) array
    out = np.zeros((p, 2))
    out[0, 0] = best[0]
    out[0, 1] = best[1]
    out[1, 0] = best[2]
    out[1, 1] = best[3]
    return out
# EVOLVE-BLOCK-END
