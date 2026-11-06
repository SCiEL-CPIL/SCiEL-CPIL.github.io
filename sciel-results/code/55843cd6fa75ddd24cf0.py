"""Quantum information: shallow-circuit Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIMEOUT = 145.0  # safety margin before 150s kill


def _remaining():
    return max(0.0, _TIMEOUT - (time.time() - _START_TIME))


def _ry_matrix(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _cz_matrix():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=np.complex128)


def _simulate_circuit(n, thetas):
    """thetas: (layers, n) array. Returns state vector."""
    layers = thetas.shape[0]
    # Start |0>^n
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            R = _ry_matrix(thetas[layer, q])
            # Build full operator: I ⊗ ... ⊗ R ⊗ ... ⊗ I
            # Use Kronecker product
            ops = []
            for j in range(n):
                if j == q:
                    ops.append(R)
                else:
                    ops.append(np.eye(2, dtype=np.complex128))
            U = ops[0]
            for j in range(1, n):
                U = np.kron(U, ops[j])
            psi = U @ psi

        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ between qubit q and q+1
                CZ = _cz_matrix()
                ops = []
                for j in range(n):
                    if j == q or j == q + 1:
                        if j == q:
                            ops.append(CZ)
                        # q+1 handled implicitly by CZ structure
                        continue
                    else:
                        ops.append(np.eye(2, dtype=np.complex128))
                # Need to handle CZ properly: it acts on qubits q and q+1
                # Rebuild: for qubits q and q+1, use CZ; others identity
                ops = []
                for j in range(n):
                    if j == q:
                        ops.append(CZ)
                    elif j == q + 1:
                        ops.append(np.eye(2, dtype=np.complex128))  # CZ already covers both
                    else:
                        ops.append(np.eye(2, dtype=np.complex128))
                # Actually, CZ is a 4x4 operator on two qubits. Let me redo this.
                # For CZ on qubits q, q+1:
                # The operator is: for all other qubits, identity; for q,q+1, CZ
                # We need to construct it correctly.
                pass

    return psi


def _simulate_circuit_v2(n, thetas):
    """Better simulation using tensor operations."""
    layers = thetas.shape[0]
    # State as tensor of shape (2,)*n
    psi = np.ones((2,) * n, dtype=np.complex128)
    psi[(0,) * n] = 1.0
    # Normalize: |0...0> means all indices are 0
    psi = np.zeros((2,) * n, dtype=np.complex128)
    psi[(0,) * n] = 1.0

    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            R = _ry_matrix(thetas[layer, q])
            # Apply R to qubit q: contract with R
            # psi has shape (2,)*n. We apply R on axis q.
            # New psi[i0,...,in] = sum_k R[iq, k] * psi[..., k, ...]
            new_psi = np.zeros_like(psi)
            # Build index arrays
            idx = np.indices((2,) * n)
            for i_val in range(2):
                for k_val in range(2):
                    # Set axis q to i_val, sum over k_val
                    sl_i = [slice(None)] * n
                    sl_k = [slice(None)] * n
                    sl_i[q] = i_val
                    sl_k[q] = k_val
                    new_psi[tuple(sl_i)] += R[i_val, k_val] * psi[tuple(sl_k)]
            psi = new_psi

        # Apply CZ on adjacent pairs (except after final layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ: |11> -> -|11>, others unchanged
                # Apply to qubits q and q+1
                new_psi = np.zeros_like(psi)
                for a in range(2):
                    for b in range(2):
                        # CZ[a,b] is nonzero only when a==b==1 (value -1) or a!=1 or b!=1 (value 1)
                        if a == 1 and b == 1:
                            cz_val = -1.0
                        else:
                            cz_val = 1.0
                        if cz_val == 0:
                            continue
                        sl = [slice(None)] * n
                        sl[q] = a
                        sl[q + 1] = b
                        new_psi[tuple(sl)] += cz_val * psi[tuple(sl)]
                psi = new_psi

    return psi


def _compute_energy(n, coupling, field, psi):
    """Compute <psi|H|psi> where H = -coupling*sum Z_i Z_{i+1} - field*sum X_i"""
    dim = 2**n
    psi_flat = psi.reshape(-1)

    energy = 0.0 + 0j

    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        # Z_i Z_{i+1} is diagonal: eigenvalue = (+1) if bits i and i+1 are same, (-1) if different
        # Actually Z|0>=|0>, Z|1>=-|1>, so Z_i Z_{i+1} eigenvalue = z_i * z_{i+1} where z_i = +1 for |0>, -1 for |1>
        # In binary: if bit i and bit i+1 are same -> +1, else -1
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            zz_val = z_i * z_ip1
            energy += zz_val * (np.abs(psi_flat[state])**2)

    # X_i terms
    for i in range(n):
        # X_i flips bit i
        for state in range(dim):
            flipped = state ^ (1 << i)
            energy += psi_flat[flipped].conj() * psi_flat[state]

    energy = -coupling * energy_real_zz - field * energy_x if False else 0
    # Let me redo this more carefully
    
    # Recompute properly
    energy = 0.0 + 0j
    # ZZ terms
    zz_energy = 0.0
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            zz_energy += (z_i * z_ip1) * (np.abs(psi_flat[state])**2)
    
    # X terms
    x_energy = 0.0 + 0j
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            x_energy += psi_flat[flipped].conj() * psi_flat[state]
    
    energy = -coupling * zz_energy - field * x_energy
    return energy.real


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy via diagonalization."""
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # ZZ terms
    for i in range(n - 1):
        for state in range(dim):
            bit_i = (state >> i) & 1
            bit_ip1 = (state >> (i + 1)) & 1
            z_i = 1 if bit_i == 0 else -1
            z_ip1 = 1 if bit_ip1 == 0 else -1
            H[state, state] += -coupling * (z_i * z_ip1)
    
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    
    # Hermitian, use eigh
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0].real


def angles(n, coupling, field, layers):
    t_start = time.time()
    
    # Get exact ground energy
    e_exact = _exact_ground_energy(n, coupling, field)
    
    # Objective function: mean energy gap per spin
    def objective(theta_flat):
        thetas = theta_flat.reshape(layers, n)
        psi = _simulate_circuit_v2(n, thetas)
        e_circuit = _compute_energy(n, coupling, field, psi)
        return (e_circuit - e_exact) / n
    
    # Initial guess: correlation-informed initialization
    # Layer weights (0.35, 0.30, 0.35) to account for CZ phase shifts
    # For transverse field Ising, ground state is roughly product state with angles depending on field/coupling
    # Simple heuristic: theta ~ arctan(field/coupling) / 2 or similar
    base_angle = np.arctan2(field, coupling) / 2.0
    weights = np.array([0.35, 0.30, 0.35])
    theta0 = np.zeros((layers, n))
    for l in range(layers):
        theta0[l, :] = base_angle * weights[l]
    
    theta0_flat = theta0.flatten()
    
    # Try multiple starting points
    best_theta = theta0_flat.copy()
    best_val = objective(theta0_flat)
    
    # Random restarts
    n_restarts = 8
    for r in range(n_restarts):
        if _remaining() < 5.0:
            break
        # Perturb initial guess
        rng = np.random.default_rng(seed=42 + r)
        theta_try = theta0_flat + rng.normal(0, 0.3, size=theta0_flat.shape)
        # Also try different base angles
        base_alt = np.arctan2(field, coupling) * (0.5 + 0.5 * rng.random())
        theta_alt = np.zeros((layers, n))
        for l in range(layers):
            theta_alt[l, :] = base_alt * weights[l] * (1 + 0.2 * rng.standard_normal())
        theta_try2 = theta_alt.flatten()
        
        for start in [theta_try, theta_try2]:
            if _remaining() < 5.0:
                break
            try:
                res = minimize(objective, start, method='L-BFGS-B', 
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x.copy()
            except Exception:
                pass
    
    # Nelder-Mead polish on best result
    if _remaining() > 10.0:
        try:
            res_nm = minimize(objective, best_theta, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_theta = res_nm.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if _remaining() > 5.0:
        try:
            res_final = minimize(objective, best_theta, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_theta = res_final.x.copy()
        except Exception:
            pass
    
    return best_theta.reshape(layers, n)
# EVOLVE-BLOCK-END
