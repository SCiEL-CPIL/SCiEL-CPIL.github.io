"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 115.0


def _time_left():
    return _DEADLINE - time.time()


def _build_cost_diag(n, edges):
    """
    Efficiently compute the diagonal of the MaxCut cost operator C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    """
    num_states = 1 << n
    c_diag = np.zeros(num_states, dtype=np.float64)

    # Precompute edge masks: for edge (u,v), mask = (1<<u) | (1<<v)
    # We check if bit u != bit v.
    # Actually, we can just iterate. For n<=10, 2^n <= 1024.
    # Let's use a vectorized approach if possible, or simple loops.
    
    # Simple loop is fine for n<=10
    for state in range(num_states):
        cut = 0
        for u, v in edges:
            # Check if bits u and v are different
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut += 1
        c_diag[state] = cut
        
    return c_diag


def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X_i) to state.
    This is a tensor product of Rx(beta) on each qubit.
    Rx(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    """
    c = np.cos(beta)
    s = np.sin(beta)
    
    # State shape: (2, 2, ..., 2)
    # We can apply the gate on each axis sequentially.
    # However, applying a 2x2 gate on an axis of a dense tensor is O(2^n).
    # Doing it n times is O(n 2^n).
    
    # Alternative: Since the mixer is the same on all qubits,
    # we can use the fact that U_B = \otimes R_x(beta).
    # The matrix elements are products of cos/sin terms.
    # Specifically, U_B_{i,j} = \prod_k (R_x)_{i_k, j_k}
    # where i_k, j_k are the bits of i, j at position k.
    
    # This allows us to compute the full matrix U_B in O(4^n) which is 2^(2n).
    # For n=10, 2^20 = 1M elements. That's a 1024x1024 matrix.
    # Building it once per beta evaluation might be expensive if we do it many times.
    # But wait, we can precompute the structure? No, beta changes.
    
    # Let's stick to the tensor method which is O(n 2^n) per application.
    # For n=10, 10 * 1024 = 10k operations. Very fast.
    
    shape = [2] * n
    state_tensor = state.reshape(shape)
    
    # We need to apply Rx on each axis.
    # numpy doesn't have a built-in "apply 2x2 matrix on axis".
    # We can use einsum or tensordot with moveaxis.
    
    # Let's try a different approach:
    # The mixer is a global operation.
    # We can compute the full unitary matrix U_B and multiply.
    # U_B is 2^n x 2^n.
    # For n=10, 1024x1024. Matrix-vector product is 1M multiplications.
    # Building U_B:
    # U_B = R_x \otimes R_x \otimes ... \otimes R_x
    # We can build this recursively or using kron.
    # np.kron is efficient for building the matrix.
    # Then state_new = U_B @ state.
    
    # Let's benchmark mentally:
    # Method 1 (Tensor): 10 iterations of moveaxis + tensordot.
    # Method 2 (Matrix): Build 1024x1024 matrix (1M entries) + mat-vec (1M mults).
    # Method 1 is likely faster in Python/NumPy because it avoids building the huge sparse-looking dense matrix.
    
    # Let's implement Method 1 carefully.
    
    for axis in range(n):
        # Move axis to front
        state_tensor = np.moveaxis(state_tensor, axis, 0)
        # Apply Rx: new[i, ...] = sum_k Rx[i, k] old[k, ...]
        # state_tensor shape: (2, 2, ..., 2)
        # Rx shape: (2, 2)
        # We want to contract the last dim of Rx with the first dim of state_tensor.
        state_tensor = np.tensordot(Rx := np.array([[c, -1j*s], [-1j*s, c]], dtype=complex), state_tensor, axes=([1], [0]))
        # Move axis back
        state_tensor = np.moveaxis(state_tensor, 0, axis)
        
    return state_tensor.reshape(-1)


def _apply_cost(state, gamma, c_diag):
    """
    Apply exp(-i gamma C) to state.
    C is diagonal with entries c_diag.
    """
    phases = np.exp(-1j * gamma * c_diag)
    return state * phases


def _expected_cut(params, n, c_diag):
    """
    Compute expected cut for given QAOA parameters.
    params: [g1, b1, g2, b2]
    """
    dim = 1 << n
    # Initial state |+>^n
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)
    
    g1, b1, g2, b2 = params
    
    # Layer 1
    state = _apply_cost(state, g1, c_diag)
    state = _apply_mixer(state, b1, n)
    
    # Layer 2
    state = _apply_cost(state, g2, c_diag)
    state = _apply_mixer(state, b2, n)
    
    # Expected cut = <state| C |state> = sum |state_i|^2 c_diag_i
    probs = np.abs(state) ** 2
    return float(np.dot(probs, c_diag))


def _optimize_angles(n, edges):
    """
    Optimize QAOA angles for a given graph.
    Returns array of shape (2, 2) with [gamma, beta] pairs.
    """
    c_diag = _build_cost_diag(n, edges)
    
    # Initial guess
    # Common heuristics: gamma ~ pi / (2 * max_degree), beta ~ pi / (4 * max_degree)
    # Or just start from a grid.
    
    # Let's do a coarse grid search first to find a good starting point.
    # Parameters are in [0, pi].
    # Grid size: 10x10x10x10 = 10,000 evaluations.
    # Each eval is O(n 2^n). For n=10, ~10k ops. Total ~100M ops.
    # This might be slow in Python if not optimized.
    # Let's reduce grid to 5x5x5x5 = 625 evals.
    
    grid_size = 5
    grid = np.linspace(0.1, np.pi - 0.1, grid_size)
    
    best_val = -np.inf
    best_params = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Vectorized grid search?
    # We can't easily vectorize the mixer application across different betas.
    # So we loop.
    
    count = 0
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if _time_left() < 5.0:
                        return best_params.reshape(2, 2)
                    params = [g1, b1, g2, b2]
                    val = _expected_cut(params, n, c_diag)
                    if val > best_val:
                        best_val = val
                        best_params = params
                    count += 1
    
    # Local refinement using Nelder-Mead
    if _time_left() > 10.0:
        def neg_cut(params):
            return -_expected_cut(params, n, c_diag)
        
        try:
            res = minimize(neg_cut, best_params.copy(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < neg_cut(best_params):
                best_params = res.x
        except Exception:
            pass
    
    # Ensure parameters are within [0, pi]
    best_params = np.clip(best_params, 0.0, np.pi)
    
    return best_params.reshape(2, 2)


def angles(n, edges, p):
    """
    Return QAOA angles for the given graph.
    n: number of qubits
    edges: list of edges (u, v)
    p: number of layers (fixed to 2 in this task)
    """
    if _time_left() < 1.0:
        # Fallback to some reasonable default
        return np.tile([0.5, 0.5], (p, 1))
    
    try:
        # Optimize for p=2
        angles_2 = _optimize_angles(n, edges)
        
        if p == 2:
            return angles_2
        elif p < 2:
            return angles_2[:p]
        else:
            # Tile for p > 2 (though task says p=2)
            reps = (p + 1) // 2
            return np.tile(angles_2, (reps, 1))[:p]
            
    except Exception:
        # Fallback
        return np.tile([0.5, 0.5], (p, 1))
