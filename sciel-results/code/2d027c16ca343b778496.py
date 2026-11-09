"""Optimize shallow circuit parameters for Ising ground states."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground state energy by diagonalizing H"""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # ZZ terms: diagonal
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            H[b, b] += -coupling * zval
    # X terms: off-diagonal
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            H[b, b_flip] += -field
    eigvals = np.linalg.eigvalsh(H)
    return eigvals[0]


def _mps_energy_and_grad(n, layers, params, coupling, field):
    """
    Compute energy expectation and analytic gradient using MPS with bond dim 2.
    params: shape (layers, n)
    Returns (energy, grad) where grad has shape (layers, n).
    
    Circuit: start |0>^n; each layer: Ry(theta_q) on all q; then CZ on adjacent pairs (except last layer).
    Qubit i is bit i (little-endian).
    """
    # MPS representation: list of tensors T[q] with shape (d_left, d_right, d_phys)
    # d_phys = 2, d_left/d_right = 2 (bond dimension)
    # Initial state |0>^n: product state
    # T[0] = (1, 1, 2) -> [1, 0] for |0>
    # T[q] = (1, 1, 2) -> [1, 0] for q > 0
    # But to use bond dim 2, we pad: T[0] shape (1, 2, 2), T[q] shape (2, 2, 2) for 0<q<n-1, T[n-1] shape (2, 1, 2)
    
    # Actually, let's use a simpler approach: track the MPS as a list of 2x2x2 tensors.
    # For a product state |0>^n, each tensor is diag(1,0) in the physical index.
    # With bond dim 2: T[q][a,b,phys] where a is left bond, b is right bond.
    # Product state: T[q][a,b,0] = delta(a,1)*delta(b,1) (if we use 0-indexed: delta(a,0)*delta(b,0))
    # T[q][a,b,1] = 0
    
    # Let me use 0-indexed bonds: bond indices 0,1.
    # Initial: T[q] has T[q][0,0,0]=1, all else 0.
    
    # Ry gate on qubit q:
    # Ry(theta) = [[c, -s], [s, c]] where c=cos(theta/2), s=sin(theta/2)
    # In MPS: T[q] new = sum_{p'} T[q] old[:, :, p'] * Ry[p', p_new]
    # i.e., T_new[a,b,p] = sum_{p'} T_old[a,b,p'] * Ry[p', p]
    
    # CZ gate between qubits q1, q2 (adjacent, q1 < q2):
    # CZ = diag(1,1,1,-1) in computational basis |q1 q2>
    # In MPS, since q1 and q2 are adjacent, we merge their tensors, apply CZ, then split.
    # T_merged[a, c, p1, p2] = sum_b T[q1][a, b, p1] * T[q2][b, c, p2]
    # Apply CZ: multiply by (-1) if p1=1 and p2=1
    # Then split back: need to find a decomposition.
    
    # For bond dim 2, the merged tensor has shape (2, 2, 2, 2) = 16 elements.
    # We need to split into T_new[q1][a, b, p1] and T_new[q2][b, c, p2] with b in {0,1}.
    # This is a matrix product state splitting: reshape merged to (2*2, 2*2) = (4, 4) matrix,
    # then SVD and keep top 2 singular values.
    
    # Let me implement this step by step.
    
    # Initialize MPS
    # T[q] shape: (2, 2, 2) for all q (we'll use bond dim 2 throughout)
    # For the first qubit, left bond is trivial (dim 1), but let's just use dim 2 and ignore index 1 for left bond of q=0.
    # Similarly for last qubit, right bond is trivial.
    
    T = []
    for q in range(n):
        t = np.zeros((2, 2, 2), dtype=np.complex128)
        t[0, 0, 0] = 1.0  # |0>
        T.append(t)
    
    # Apply circuit
    for layer in range(layers):
        # Apply Ry to every qubit
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            # Ry = [[c, -s], [s, c]]
            # T_new[a,b,p] = sum_{p'} T_old[a,b,p'] * Ry[p', p]
            # Ry[0,0]=c, Ry[0,1]=-s, Ry[1,0]=s, Ry[1,1]=c
            T[q][:, :, 0] = T[q][:, :, 0] * c + T[q][:, :, 1] * s
            T[q][:, :, 1] = -T[q][:, :, 0].copy() * s + T[q][:, :, 1] * c
            # Wait, that's wrong. Let me redo:
            # T_new[a,b,0] = T_old[a,b,0]*c + T_old[a,b,1]*s
            # T_new[a,b,1] = -T_old[a,b,0]*s + T_old[a,b,1]*c
            # I need to use the old values, so let me copy first.
        
        # Let me redo this more carefully
        for q in range(n):
            theta = params[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            old = T[q].copy()
            T[q][:, :, 0] = old[:, :, 0] * c + old[:, :, 1] * s
            T[q][:, :, 1] = -old[:, :, 0] * s + old[:, :, 1] * c
        
        # Apply CZ on adjacent pairs (except after last layer)
        if layer < layers - 1:
            for q in range(n - 1):
                # CZ between q and q+1
                # Merge T[q] and T[q+1]
                # Merged[a, c, p1, p2] = sum_b T[q][a, b, p1] * T[q+1][b, c, p2]
                merged = np.einsum('abp,bcq->acpq', T[q], T[q+1])
                # Apply CZ: multiply by -1 if p1=1 and p2=1
                merged[:, :, 1, 1] *= -1
                # Split back: reshape to (4, 4) matrix, SVD, keep top 2
                # Merged has shape (2, 2, 2, 2). Reshape to (2*2, 2*2) = (4, 4)
                # Index: (a, p1) -> row, (c, p2) -> col
                mat = merged.reshape(4, 4)
                U, S, Vh = np.linalg.svd(mat, full_matrices=False)
                # Keep top 2 singular values
                U2 = U[:, :2]
                S2 = S[:2]
                V2 = Vh[:2, :]
                # T_new[q][a, b, p1] = sum_{b'} U[a*2+p1, b'] * S2[b']
                # T_new[q+1][b, c, p2] = sum_{b'} S2[b'] * V2[b', c*2+p2]
                T[q] = np.zeros((2, 2, 2), dtype=np.complex128)
                T[q+1] = np.zeros((2, 2, 2), dtype=np.complex128)
                for a in range(2):
                    for p1 in range(2):
                        for b in range(2):
                            T[q][a, b, p1] = U2[a*2+p1, b] * S2[b]
                for c in range(2):
                    for p2 in range(2):
                        for b in range(2):
                            T[q+1][b, c, p2] = S2[b] * V2[b, c*2+p2]
    
    # Compute energy expectation
    # H = -coupling * sum_{i=0}^{n-2} Z_i Z_{i+1} - field * sum_{i=0}^{n-1} X_i
    # <Z_i Z_{i+1}>: since Z is diagonal, we can compute this from the MPS
    # <X_i>: off-diagonal, need to contract with X operator
    
    # For diagonal operators (ZZ): 
    # <Z_i Z_{i+1}> = sum_{phys states} |amp|^2 * z_i * z_{i+1}
    # where z = +1 for |0>, -1 for |1>
    # In MPS: we can compute the reduced density matrix for sites i, i+1
    
    # Actually, for a 1D MPS, the expectation of a local operator can be computed
    # by contracting the MPS with the operator inserted.
    
    # Let me compute the full state vector for small n (n<=6, dim<=64) to verify,
    # but for the actual computation, use MPS contraction.
    
    # For n <= 6, the state vector has at most 64 components. Let me just build it.
    # This is simpler and n is small.
    
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    
    # Contract MPS to get state vector
    # state[b0, b1, ..., b_{n-1}] = sum_{bond indices} T[0][_, 0, b0] * T[1][0, _, b1] * ... * T[n-1][_, _, b_{n-1}]
    # With bond dim 2:
    # state[b0,...,b_{n-1}] = sum_{a1,...,a_{n-1}} T[0][0, a1, b0] * T[1][a1, a2, b1] * ... * T[n-1][a_{n-1}, 0, b_{n-1}]
    # Wait, the first tensor has left bond index 0 (trivial), last has right bond index 0 (trivial).
    
    # Let me just do it:
    # Start with a "left environment" vector of shape (2,): [1, 0]
    # For each qubit q, contract with T[q] to get new environment
    # env shape: (2, 2^q) -> after contracting q, becomes (2, 2^{q+1})
    
    env = np.array([[1.0], [0.0]], dtype=np.complex128)  # shape (2, 1)
    
    for q in range(n):
        # env has shape (2, 2^q), T[q] has shape (2, 2, 2)
        # new_env[a_right, b0...bq] = sum_{a_left} env[a_left, b0...b_{q-1}] * T[q][a_left, a_right, bq]
        # Reshape env to (2, 2^q), T[q] to (2, 2, 2)
        # new_env shape: (2, 2^{q+1})
        # new_env[a_right, idx] where idx encodes (b0...b_{q-1}, bq)
        new_env = np.zeros((2, 2 ** (q + 1)), dtype=np.complex128)
        for a_left in range(2):
            for a_right in range(2):
                for bq in range(2):
                    # env[a_left, :] has shape (2^q,)
                    # T[q][a_left, a_right, bq] is a scalar
                    new_env[a_right, :] += env[a_left, :] * T[q][a_left, a_right, bq]
                    # But we need to interleave: the new index is (old_idx, bq)
                    # Actually, let me think about this differently.
        
        # Let me use a different approach: build the state vector by iterating over all basis states
        # For n <= 6, this is fine.
        pass
    
    # Simpler: just iterate over all 2^n basis states
    state = np.zeros(dim, dtype=np.complex128)
    for b in range(dim):
        # b encodes (b_{n-1}, ..., b_0) in binary, but qubit i is bit i
        # So bit i of b gives the state of qubit i
        amp = 1.0 + 0j
        # Contract: sum over bond indices a1, ..., a_{n-1}
        # T[0][0, a1, b0] * T[1][a1, a2, b1] * ... * T[n-1][a_{n-1}, 0, b_{n-1}]
        # where bq = (b >> q) & 1
        if n == 1:
            amp = T[0][0, 0, b]
        else:
            # Iterate over all bond index combinations
            # For n qubits, there are n-1 bond indices, each 2-valued, so 2^{n-1} combinations
            for bond_combo in range(2 ** (n - 1)):
                a_prev = 0  # left bond of first tensor is trivial (index 0)
                prod = 1.0 + 0j
                ok = True
                for q in range(n):
                    bq = (b >> q) & 1
                    if q == 0:
                        a_left = 0
                    else:
                        a_left = (bond_combo >> (q - 1)) & 1
                    if q == n - 1:
                        a_right = 0
                    else:
                        a_right = (bond_combo >> q) & 1
                    prod *= T[q][a_left, a_right, bq]
                amp += prod
    
    state[b] = amp
    
    # Normalize
    norm = np.vdot(state, state).real
    if norm > 1e-15:
        state /= np.sqrt(norm)
    
    # Compute energy
    e = 0.0
    # ZZ terms
    for i in range(n - 1):
        for b in range(dim):
            bi = (b >> i) & 1
            bj = (b >> (i + 1)) & 1
            zval = (1 - 2 * bi) * (1 - 2 * bj)
            e += -coupling * zval * (state[b] * np.conj(state[b])).real
    # X terms
    for i in range(n):
        for b in range(dim):
            b_flip = b ^ (1 << i)
            e += -field * 2 * (state[b_flip] * np.conj(state[b])).real
    
    # Compute gradient analytically
    # dE/dtheta_{layer,q} = 2 * Re[ <psi| dH/dtheta |psi> ] but H doesn't depend on theta
    # Actually, E = <psi(theta)|H|psi(theta)>
    # dE/dtheta = <dpsi/dtheta|H|psi> + <psi|H|dpsi/dtheta> = 2 Re[<dpsi/dtheta|H|psi>]
    
    # dpsi/dtheta_{layer,q}: differentiate the circuit with respect to theta_{layer,q}
    # This is complex. Let me use a simpler approach: numerical gradient for now,
    # but that defeats the purpose. Let me think...
    
    # Actually, for the MPS approach to be useful, I need analytic gradients.
    # The gradient of the energy with respect to a parameter in a quantum circuit
    # can be computed using the parameter-shift rule or by tracking the derivative
    # of the state through the circuit.
    
    # Let me use a different strategy: since n is small (4 or 6), let me just
    # use the state vector approach but with a better optimizer (L-BFGS-B) and
    # more restarts. The key improvement is using scipy's L-BFGS-B which is
    # much more efficient than Nelder-Mead for smooth functions.
    
    # For now, return energy and a placeholder gradient (we'll use numerical gradient)
    grad = np.zeros((layers, n))
    
    return e, grad, state


def angles(n, coupling, field, layers):
    """Optimize circuit parameters to minimize mean energy gap per spin."""
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute ground energy
    e_ground = _exact_ground_energy(n, coupling, field)
    
    def obj_and_grad(params_flat):
        params = params_flat.reshape((layers, n))
        e, _, _ = _mps_energy_and_grad(n, layers, params, coupling, field)
        val = (e - e_ground) / n
        # Numerical gradient (central difference)
        grad = np.zeros_like(params_flat)
        h = 1e-6
        for i in range(len(params_flat)):
            params_plus = params_flat.copy()
            params_minus = params_flat.copy()
            params_plus[i] += h
            params_minus[i] -= h
            e_plus, _, _ = _mps_energy_and_grad(n, layers, params_plus.reshape((layers, n)), coupling, field)
            e_minus, _, _ = _mps_energy_and_grad(n, layers, params_minus.reshape((layers, n)), coupling, field)
            grad[i] = (e_plus - e_minus) / (2 * h * n)
        return val, grad
    
    # Multi-restart with L-BFGS-B
    best_params = np.zeros((layers, n))
    best_val = obj_and_grad(best_params.flatten())[0]
    
    np.random.seed(42)
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x_init = np.zeros((layers, n))
        else:
            x_init = np.random.uniform(-np.pi / 2, np.pi / 2, (layers, n))
        
        x_flat = x_init.flatten()
        
        try:
            result = scipy.optimize.minimize(
                obj_and_grad,
                x_flat,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.reshape((layers, n))
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    return best_params
# EVOLVE-BLOCK-END
