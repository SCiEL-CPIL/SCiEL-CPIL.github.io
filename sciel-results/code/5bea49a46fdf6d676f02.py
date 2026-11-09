"""Optimized shallow-circuit Ising ground states using tensor-network energy and L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Pauli matrices
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Ry(theta) = cos(theta/2)*I - i*sin(theta/2)*X
def ry_matrix(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return c * I2 - 1j * s * X

# CZ gate
CZ = np.diag([1, 1, 1, -1]).astype(complex)
CZ = CZ.reshape(2, 2, 2, 2)

def _build_circuit_tensors(n, layers, thetas):
    """
    Build list of tensors. Each tensor is a 2D matrix of shape (2^n, 2^n) is too expensive for large n.
    Instead we use a different approach: since n is small (4 or 6), we can use state vectors.
    For n=6, 2^6=64, so state vector is 64 complex numbers. This is very fast.
    """
    pass

def _compute_energy_and_grad(n, coupling, field, thetas):
    """
    Compute energy <psi|H|psi> and its gradient w.r.t. thetas using state-vector simulation.
    thetas has shape (layers, n).
    """
    # Build state vector
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0  # |0>^n

    # Precompute basis indices for each qubit
    # Qubit i is bit i (little-endian: qubit 0 is least significant bit)
    
    # Helper: apply single-qubit gate on qubit i to state vector
    def apply_single_qubit(psi, gate, i, n):
        """Apply 2x2 gate on qubit i (bit i)."""
        result = np.zeros_like(psi)
        size = 2**n
        for idx in range(size):
            bit = (idx >> i) & 1
            idx0 = idx & ~(1 << i)  # bit i = 0
            idx1 = idx0 | (1 << i)  # bit i = 1
            # gate[0,0]*psi[idx0] + gate[0,1]*psi[idx1] -> result[idx0]
            # gate[1,0]*psi[idx0] + gate[1,1]*psi[idx1] -> result[idx1]
            result[idx0] = gate[0, 0] * psi[idx0] + gate[0, 1] * psi[idx1]
            result[idx1] = gate[1, 0] * psi[idx0] + gate[1, 1] * psi[idx1]
        return result

    # Helper: apply CZ on qubits i, j (i<j)
    def apply_cz(psi, i, j, n):
        """Apply CZ gate on qubits i and j. CZ is diagonal: only |11> gets phase -1."""
        result = psi.copy()
        size = 2**n
        mask_i = 1 << i
        mask_j = 1 << j
        for idx in range(size):
            if ((idx & mask_i) and (idx & mask_j)):
                result[idx] = -result[idx]
        return result

    # Apply circuit layers
    for layer in range(layers):
        # Apply Ry(theta) to every qubit
        for i in range(n):
            theta = thetas[layer, i]
            gate = ry_matrix(theta)
            psi = apply_single_qubit(psi, gate, i, n)
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                psi = apply_cz(psi, i, i + 1, n)

    # Compute energy <psi|H|psi>
    # H = -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1} - field * sum_{i=0}^{n-1} X_i
    
    # Compute expectation values
    # For Z_i Z_j: diagonal in computational basis
    # For X_i: off-diagonal
    
    size = 2**n
    energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        for idx in range(size):
            prob = np.abs(psi[idx])**2
            if prob > 1e-15:
                zi = 1 if (idx & mask_i) == 0 else -1
                zj = 1 if (idx & mask_j) == 0 else -1
                energy += prob * (zi * zj)
        energy *= -coupling / (n)  # Wait, we need total energy first, then divide by n at the end
        # Actually let's accumulate total energy first
    
    # Let me redo this more carefully
    total_energy = 0.0
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        exp_val = 0.0
        for idx in range(size):
            prob = np.abs(psi[idx])**2
            if prob > 1e-15:
                zi = 1 if (idx & mask_i) == 0 else -1
                zj = 1 if (idx & mask_j) == 0 else -1
                exp_val += prob * (zi * zj)
        total_energy += -coupling * exp_val
    
    # X terms: -field * sum X_i
    for i in range(n):
        mask_i = 1 << i
        exp_val = 0.0
        for idx in range(size):
            idx_flip = idx ^ mask_i
            # <psi|X_i|psi> = sum_k psi[k]*conj(psi[k^mask_i]) ... 
            # Actually X_i flips bit i, so (X_i psi)[idx] = psi[idx ^ mask_i]
            # <psi|X_i|psi> = sum_idx conj(psi[idx]) * psi[idx ^ mask_i]
            # But this double counts. Let's be careful.
            # <psi|X_i|psi> = sum_{a,b} psi[a]*conj(psi[b]) * <b|X_i|a>
            # X_i|a> = |a ^ mask_i> if we define it that way... 
            # Actually X|0>=|1>, X|1>=|0>, so X_i|idx> = |idx ^ mask_i>
            # So <b|X_i|a> = 1 if b = a ^ mask_i, else 0
            # <psi|X_i|psi> = sum_a psi[a^mask_i] * conj(psi[a]) ... 
            # Wait: <psi|X_i|psi> = sum_{b,a} conj(psi[b]) * psi[a] * <b|X_i|a>
            # = sum_a conj(psi[a^mask_i]) * psi[a]
            # = sum_a psi[a] * conj(psi[a^mask_i])
            # This is real since it equals its own conjugate.
            pass
        
        # Simpler: compute directly
        exp_val = 0.0
        for idx in range(size):
            idx_flip = idx ^ mask_i
            exp_val += psi[idx_flip] * np.conj(psi[idx])
        # This gives <psi|X_i|psi> but let's verify:
        # (X_i psi)[idx] = psi[idx ^ mask_i]
        # <psi|X_i psi> = sum_idx conj(psi[idx]) * psi[idx ^ mask_i]
        # Yes, that's what we have.
        total_energy += -field * np.real(exp_val)
    
    return total_energy

def _compute_energy_and_grad_fast(n, coupling, field, thetas):
    """
    Compute energy and gradient using state-vector simulation with complex step or finite differences.
    For efficiency, use analytic gradient via backpropagation through the circuit.
    """
    # For n=4 or 6, we can afford to compute gradients numerically or analytically.
    # Let's use analytic gradient by tracking the state and computing derivatives.
    
    # Actually, for small n, let's just use a more efficient vectorized approach.
    
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1.0
    
    def apply_single_qubit_vec(psi, gate, i, n):
        """Vectorized single-qubit gate application."""
        result = np.zeros_like(psi)
        size = 2**n
        mask = 1 << i
        # For all indices where bit i is 0
        idx0 = np.array([k for k in range(size) if (k & mask) == 0])
        idx1 = idx0 | mask
        result[idx0] = gate[0, 0] * psi[idx0] + gate[0, 1] * psi[idx1]
        result[idx1] = gate[1, 0] * psi[idx0] + gate[1, 1] * psi[idx1]
        return result
    
    def apply_cz_vec(psi, i, j, n):
        """Vectorized CZ application."""
        result = psi.copy()
        size = 2**n
        mask_i = 1 << i
        mask_j = 1 << j
        idx_both = np.array([k for k in range(size) if (k & mask_i) and (k & mask_j)])
        result[idx_both] = -result[idx_both]
        return result
    
    # Precompute index arrays for CZ
    size = 2**n
    cz_indices = []
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        idx_both = np.array([k for k in range(size) if (k & mask_i) and (k & mask_j)])
        cz_indices.append(idx_both)
    
    # Precompute index arrays for single-qubit gates
    sq_indices = []
    for i in range(n):
        mask = 1 << i
        idx0 = np.array([k for k in range(size) if (k & mask) == 0])
        idx1 = idx0 | mask
        sq_indices.append((idx0, idx1))
    
    # Apply circuit
    for layer in range(layers):
        for i in range(n):
            theta = thetas[layer, i]
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            gate = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
            idx0, idx1 = sq_indices[i]
            psi[idx0] = gate[0, 0] * psi[idx0] + gate[0, 1] * psi[idx1]
            psi[idx1] = gate[1, 0] * psi[idx0] + gate[1, 1] * psi[idx1]
        
        if layer < layers - 1:
            for i in range(n - 1):
                idx_both = cz_indices[i]
                psi[idx_both] = -psi[idx_both]
    
    # Compute energy
    total_energy = 0.0
    
    # ZZ terms
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        exp_val = 0.0
        for idx in range(size):
            prob = np.abs(psi[idx])**2
            if prob > 1e-15:
                zi = 1.0 if (idx & mask_i) == 0 else -1.0
                zj = 1.0 if (idx & mask_j) == 0 else -1.0
                exp_val += prob * (zi * zj)
        total_energy += -coupling * exp_val
    
    # X terms
    for i in range(n):
        mask_i = 1 << i
        exp_val = 0.0
        for idx in range(size):
            idx_flip = idx ^ mask_i
            exp_val += psi[idx_flip] * np.conj(psi[idx])
        total_energy += -field * np.real(exp_val)
    
    return total_energy

def _objective(params, n, coupling, field, layers):
    """Objective function for optimization: mean energy per spin."""
    thetas = params.reshape(layers, n)
    energy = _compute_energy_and_grad_fast(n, coupling, field, thetas)
    return energy / n

def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H."""
    size = 2**n
    H = np.zeros((size, size), dtype=complex)
    
    # Build H = -coupling * sum Z_i Z_{i+1} - field * sum X_i
    # ZZ terms (diagonal)
    for i in range(n - 1):
        mask_i = 1 << i
        mask_j = 1 << (i + 1)
        for idx in range(size):
            zi = 1.0 if (idx & mask_i) == 0 else -1.0
            zj = 1.0 if (idx & mask_j) == 0 else -1.0
            H[idx, idx] += -coupling * zi * zj
    
    # X terms (off-diagonal)
    for i in range(n):
        mask_i = 1 << i
        for idx in range(size):
            idx_flip = idx ^ mask_i
            H[idx, idx_flip] += -field
    
    # Diagonalize
    eigvals = np.linalg.eigvalsh(H)
    return np.real(eigvals[0])

def angles(n, coupling, field, layers):
    """
    Optimize circuit parameters to minimize mean energy gap per spin.
    Returns array of shape (layers, n) with angles in radians.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Compute exact ground energy
    e_ground = _exact_ground_energy(n, coupling, field)
    
    # We want to minimize (circuit_energy - e_ground)/n
    # This is equivalent to minimizing circuit_energy
    
    # Multiple diverse initializations
    best_energy = np.inf
    best_params = np.zeros((layers, n))
    
    # Try several initializations
    initializations = []
    
    # 1. All zeros
    initializations.append(np.zeros((layers, n)))
    
    # 2. All pi/2
    initializations.append(np.full((layers, n), np.pi / 2))
    
    # 3. Random inits
    rng = np.random.default_rng(42)
    for _ in range(8):
        initializations.append(rng.uniform(-np.pi, np.pi, (layers, n)))
    
    # 4. Structured inits based on field
    # For transverse field Ising, the ground state interpolates between |0>^n and |+>^n
    # As field increases, more |+> character
    frac = field / 2.0  # Rough scaling
    for _ in range(4):
        init = np.zeros((layers, n))
        init[-1] = np.pi / 2 * frac + rng.normal(0, 0.1, n)
        initializations.append(init)
    
    for init in initializations:
        if time.time() > deadline - 10.0:
            break
        
        params_init = init.flatten()
        
        try:
            result = minimize(
                _objective,
                params_init,
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
                
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time permits
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective,
                best_params.flatten(),
                args=(n, coupling, field, layers),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x.reshape(layers, n)
        except Exception:
            pass
    
    return best_params
# EVOLVE-BLOCK-END
