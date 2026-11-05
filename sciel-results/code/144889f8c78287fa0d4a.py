"""
Quantum information: shallow-circuit Ising ground states
MPS-based optimization with bond dimension chi=4 for exact circuit representation.
"""
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
_DEADLINE = 140.0  # seconds, leave 10s margin for the 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


_START_TIME = time.time()


def _apply_ry(theta):
    """Return 2x2 Ry rotation matrix for angle theta."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])


def _apply_cz_to_mps(mps, n, bond):
    """
    Apply CZ gates on all adjacent pairs in ascending order to an MPS.
    mps: list of tensors, mps[i] has shape (chi_left, 2, chi_right)
         For i=0: chi_left=1, for i=n-1: chi_right=1
    bond: bond dimension (chi)
    Returns new mps list.
    """
    # CZ matrix: diag(1,1,1,-1) in computational basis
    # CZ|00>=|00>, CZ|01>=|01>, CZ|10>=|10>, CZ|11>=-|11>
    cz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype=complex)

    new_mps = [t.copy() for t in mps]

    for i in range(n - 1):
        # Apply CZ between qubit i and i+1
        # Merge i and i+1, apply CZ, then split back
        # mps[i]: (chi_l, 2, chi_m)  shape (bond, 2, bond)
        # mps[i+1]: (chi_m, 2, chi_r) shape (bond, 2, bond)
        # Combined: (chi_l, 2, 2, chi_r) with bond index chi_m contracted
        combined = np.einsum('abc,cd e->abde', new_mps[i], new_mps[i+1])
        # combined shape: (bond, 2, 2, bond)
        # Reshape to (bond, 4, bond)
        combined = combined.reshape(bond, 4, bond)
        # Apply CZ: sum_k cz[a,k] * combined[:, k, :]
        # CZ acts on the 4-dimensional space
        new_combined = np.einsum('ak,ckd->acd', cz, combined)
        # new_combined shape: (bond, 4, bond)
        # Now we need to split back into two tensors.
        # Use SVD to maintain bond dimension
        # Reshape to (bond*4, bond) -> actually we need to split the 4 into (2,2)
        # Reshape new_combined to (bond, 2, 2, bond)
        new_combined = new_combined.reshape(bond, 2, 2, bond)
        # Now we need to split the middle 2x2 into two tensors
        # Reshape to (bond*2, 2*bond) for SVD
        mat = new_combined.transpose(0, 2, 1, 3).reshape(bond * 2, 2 * bond)
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        # Truncate to bond dimension
        chi_cut = min(bond, len(S))
        U = U[:, :chi_cut]
        S = S[:chi_cut]
        Vt = Vt[:chi_cut, :]
        # Reconstruct
        left = (U.reshape(bond, 2, chi_cut) * S).transpose(0, 2, 1)  # (bond, chi_cut, 2)
        right = Vt.reshape(chi_cut, 2, bond)  # (chi_cut, 2, bond)
        # Pad if chi_cut < bond
        if chi_cut < bond:
            left = np.zeros((bond, bond, 2), dtype=complex)
            left[:, :chi_cut, :] = (U.reshape(bond, 2, chi_cut) * S).transpose(0, 2, 1)
            right = np.zeros((bond, 2, bond), dtype=complex)
            right[:chi_cut, :, :] = Vt.reshape(chi_cut, 2, bond)
        new_mps[i] = left
        new_mps[i+1] = right

    return new_mps


def _circuit_mps(n, coupling, field, layers, thetas):
    """
    Build the MPS for the circuit with given angles.
    thetas: (layers, n) array
    Returns list of MPS tensors.
    """
    bond = 4
    # Start with |0>^n
    # Each tensor: (1, 2, 1) initially, representing |0>
    mps = []
    for i in range(n):
        t = np.zeros((1, 2, 1), dtype=complex)
        t[0, 0, 0] = 1.0
        mps.append(t)

    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            ry = _apply_ry(thetas[layer, i])
            mps[i] = np.einsum('ab,bcd->acd', ry, mps[i])
        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            mps = _apply_cz_to_mps(mps, n, bond)

    return mps


def _compute_energy_mps(mps, n, coupling, field):
    """
    Compute <H> for the MPS state.
    H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    """
    # X = [[0,1],[1,0]], Z = [[1,0],[0,-1]]
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    energy = 0.0

    # Compute <X_i> for each i
    for i in range(n):
        # <psi| X_i |psi>
        # Need to contract MPS with X on site i
        # Left contraction: all sites < i
        # Right contraction: all sites > i

        # For small n, we can just build the full state
        pass

    # For n=4,6 it's feasible to build the full state from MPS
    # Let's do that for simplicity and correctness
    state = _mps_to_state(mps, n)
    H = _build_hamiltonian(n, coupling, field)
    energy = np.vdot(state, H @ state).real
    return energy


def _mps_to_state(mps, n):
    """Convert MPS to full state vector."""
    state = mps[0]  # (1, 2, chi)
    for i in range(1, n):
        state = np.einsum('...ab,bcd->...ad', state, mps[i])
    return state.flatten()


def _build_hamiltonian(n, coupling, field):
    """Build the full Hamiltonian matrix for small n."""
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    # -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        for state in range(dim):
            # Compute Z_i Z_{i+1} eigenvalue
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            zz = (1 if bit_i == 0 else -1) * (1 if bit_ip1 == 0 else -1)
            H[state, state] += -coupling * zz

    # -field * sum X_i
    for i in range(n):
        for state in range(dim):
            new_state = state ^ (1 << i)
            H[new_state, state] += -field

    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy."""
    H = _build_hamiltonian(n, coupling, field)
    evals = np.linalg.eigvalsh(H)
    return evals[0]


def _energy_gap(n, coupling, field, thetas):
    """Compute mean energy gap per spin."""
    e_circuit = _compute_energy_mps(_circuit_mps(n, coupling, field, 3, thetas), n, coupling, field)
    e_ground = _exact_ground_energy(n, coupling, field)
    return (e_circuit - e_ground) / n


def _objective(params, n, coupling, field):
    """Objective function for optimizer."""
    thetas = params.reshape(3, n)
    return _energy_gap(n, coupling, field, thetas)


def angles(n, coupling, field, layers):
    global _START_TIME
    _START_TIME = time.time()

    # Parameters: 3 layers x n angles
    n_params = layers * n

    # Initial guess: product |+> state
    x0 = np.zeros(n_params)
    x0[-n:] = np.pi / 2  # Last layer all pi/2 for |+>

    # Multi-start optimization
    best_x = x0.copy()
    best_val = _objective(x0, n, coupling, field)

    # Try a few random starts
    rng = np.random.default_rng(42)
    n_starts = 5
    for start in range(n_starts):
        if _time_remaining() < 20:
            break
        if start == 0:
            x_init = x0.copy()
        else:
            x_init = rng.uniform(-np.pi, np.pi, n_params)

        try:
            res = minimize(_objective, x_init, args=(n, coupling, field),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            val = res.fun
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass

    # Polish with tighter tolerance
    if _time_remaining() > 15:
        try:
            res = minimize(_objective, best_x, args=(n, coupling, field),
                           method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Second polish
    if _time_remaining() > 10:
        try:
            res = minimize(_objective, best_x, args=(n, coupling, field),
                           method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    thetas = best_x.reshape(layers, n)
    return thetas
