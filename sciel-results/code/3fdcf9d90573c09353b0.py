"""QAOA parameter design: gradient-based optimization with efficient simulation."""
import numpy as np
import time
from scipy.optimize import minimize

# Global deadline tracking
_DEADLINE = 150.0
_start_time = None


def _check_time():
    """Return True if we should stop (deadline approaching)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (_DEADLINE - 5.0)


def _build_hamiltonian(n, edges):
    """
    Build the MaxCut cost Hamiltonian C and the mixer Hamiltonian B = sum X_i.
    C = sum_{(i,j) in edges} (1 - Z_i Z_j) / 2
    B = sum_i X_i
    We work with the unitary exp(-i*gamma*C) and exp(-i*beta*B).
    """
    # C as a matrix: (1 - Z_i Z_j)/2 for each edge
    # Z_i Z_j is diagonal with entries +1 if same bit, -1 if different
    # (1 - Z_i Z_j)/2 = 1 if different bits (cut), 0 if same
    # So C is diagonal with 1 on cut configurations, 0 otherwise
    # Actually C = sum (1-Z_iZ_j)/2, which is diagonal
    # C|psi> = sum_{(i,j) in edges} (1 - Z_i Z_j)|psi> / 2
    
    # For efficiency, we represent C as a diagonal vector
    # C_diag[k] = number of edges cut in configuration k
    C_diag = np.zeros(2**n, dtype=np.float64)
    for (i, j) in edges:
        # For each configuration, check if bits i and j differ
        for k in range(2**n):
            bi = (k >> i) & 1
            bj = (k >> j) & 1
            if bi != bj:
                C_diag[k] += 1.0
    # C = sum (1 - Z_iZ_j)/2, so each edge contributes 0 or 1
    # C_diag is already the eigenvalue (number of cut edges)
    
    # B = sum X_i, which is a sparse matrix
    # X flips one bit
    # B|k> = sum_i |k with bit i flipped>
    # B is a 2^n x 2^n matrix, but very sparse (n non-zeros per row)
    # We'll apply it as a function
    
    return C_diag


def _apply_cost_unitary(state, C_diag, gamma):
    """Apply exp(-i*gamma*C) to state. C is diagonal, so this is element-wise."""
    # exp(-i*gamma*C) applied to state
    # C is diagonal with values C_diag
    phase = np.exp(-1j * gamma * C_diag)
    return state * phase


def _apply_mixer_unitary(state, beta, n):
    """Apply exp(-i*beta*sum_i X_i) to state."""
    # sum_i X_i: each X_i flips bit i
    # exp(-i*beta*sum X_i) = product_i exp(-i*beta*X_i) since X_i commute
    # exp(-i*beta*X) = cos(beta)*I - i*sin(beta)*X
    # So for each qubit i: state_new = cos(beta)*state + (-i*sin(beta))*X_i*state
    # We can apply this sequentially for each qubit
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    
    result = state.copy()
    for i in range(n):
        # X_i flips bit i: for each basis state k, X_i|k> = |k XOR (1<<i)>
        flipped = result.copy()
        for k in range(2**n):
            flipped[k] = result[k ^ (1 << i)]
        result = c * result + s * flipped
    
    return result


def _simulate(state0, gammas, betas, C_diag, n):
    """
    Full state-vector simulation of the QAOA circuit.
    Returns the final state vector.
    """
    state = state0.copy()
    for gamma, beta in zip(gammas, betas):
        state = _apply_cost_unitary(state, C_diag, gamma)
        state = _apply_mixer_unitary(state, beta, n)
    return state


def _expected_cut(state, edges, n):
    """
    Compute the expected cut value from the final state.
    E[cut] = sum_{(i,j) in edges} (1 - <Z_i Z_j>) / 2
    <Z_i Z_j> = sum_k Z_i(k) Z_j(k) |psi_k|^2
    """
    probs = np.abs(state)**2
    expected_cut = 0.0
    for (i, j) in edges:
        # Z_i Z_j for each basis state
        zz = np.zeros(2**n, dtype=np.float64)
        for k in range(2**n):
            bi = (k >> i) & 1
            bj = (k >> j) & 1
            zi = 1.0 if bi == 0 else -1.0
            zj = 1.0 if bj == 0 else -1.0
            zz[k] = zi * zj
        exp_zz = np.sum(zz * probs)
        expected_cut += (1.0 - exp_zz) / 2.0
    return expected_cut


def _exact_maxcut(n, edges):
    """Exhaustive search for the exact MaxCut value."""
    best = 0
    for k in range(2**n):
        cut = 0
        for (i, j) in edges:
            bi = (k >> i) & 1
            bj = (k >> j) & 1
            if bi != bj:
                cut += 1
        best = max(best, cut)
    return best


def _objective(params, n, edges, C_diag, exact_cut):
    """
    Objective function to maximize: expected_cut / exact_cut.
    We minimize the negative.
    """
    if _check_time():
        return 1.0  # Large penalty to stop optimization
    
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones(2^n)
    state0 = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    state = _simulate(state0, gammas, betas, C_diag, n)
    exp_cut = _expected_cut(state, edges, n)
    
    # Return negative ratio (we minimize)
    return -exp_cut / exact_cut


def _gradient(params, n, edges, C_diag, exact_cut, eps=1e-5):
    """Numerical gradient of the objective."""
    grad = np.zeros_like(params)
    f0 = _objective(params, n, edges, C_diag, exact_cut)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, n, edges, C_diag, exact_cut)
        grad[i] = (f_plus - f0) / eps
    return grad


def _optimize_angles(n, edges, p, C_diag, exact_cut, max_iter=50):
    """
    Optimize QAOA angles using L-BFGS-B.
    """
    if _check_time():
        return np.tile([0.35, 0.25], (p, 1))
    
    # Try multiple initial points
    best_params = None
    best_obj = np.inf
    
    n_starts = 5
    for start_idx in range(n_starts):
        if _check_time():
            break
        
        if start_idx == 0:
            # Start near the seed values
            x0 = np.tile([0.35, 0.25], p)
        else:
            # Random initialization in [0, pi]
            x0 = np.random.uniform(0.1, np.pi, size=2*p)
        
        try:
            result = minimize(
                _objective, x0,
                args=(n, edges, C_diag, exact_cut),
                method='L-BFGS-B',
                bounds=[(0.0, np.pi)] * 2*p,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Reshape to (p, 2)
    p_actual = len(best_params) // 2
    angles_arr = best_params.reshape(p_actual, 2)
    return angles_arr


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits
        edges: list of edges (i, j)
        p: number of QAOA layers (p=2)
    
    Returns:
        numpy array of shape (p, 2), rows are (gamma, beta) in radians
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure edges is a list of tuples
    edges = list(edges)
    
    # Build the cost Hamiltonian diagonal
    C_diag = _build_hamiltonian(n, edges)
    
    # Compute exact MaxCut
    exact_cut = _exact_maxcut(n, edges)
    if exact_cut == 0:
        return np.zeros((p, 2), dtype=np.float64)
    
    # Optimize
    result = _optimize_angles(n, edges, p, C_diag, exact_cut)
    
    # Ensure correct shape
    if result.shape != (p, 2):
        result = np.tile([0.35, 0.25], (p, 1))
    
    return result
