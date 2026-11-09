"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 135.0  # safety margin before 150s kill
_START_TIME = None


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut size for a given QAOA circuit.
    State: |+>^n, then for each layer: exp(-i*gamma*C) then exp(-i*beta*sum X)
    C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2  (MaxCut cost)
    Expected cut = sum_{(u,v) in edges} (1 - <Z_u Z_v>)/2
    """
    dim = 1 << n
    # Start in |+>^n state
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute bit masks for each qubit
    masks = [1 << i for i in range(n)]
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i * gamma * C)
        # C = sum_{(u,v)} (1 - Z_u Z_v)/2
        # So exp(-i*gamma*C) = exp(-i*gamma/2 * sum (1-Z_u Z_v))
        # = exp(-i*gamma/2 * |E|) * prod_{(u,v)} exp(i*gamma/2 * Z_u Z_v)
        # Since Z_u Z_v is diagonal with entries +1 or -1,
        # exp(i*gamma/2 * Z_u Z_v) is diagonal with entries exp(i*gamma/2) or exp(-i*gamma/2)
        
        # The phase for a basis state |x> is: exp(-i*gamma/2 * |E|) * prod_{(u,v) in E} exp(i*gamma/2 * (-1)^{x_u + x_v})
        # = exp(-i*gamma/2 * |E|) * exp(i*gamma/2 * sum_{(u,v) in E} (-1)^{x_u + x_v})
        # sum_{(u,v) in E} (-1)^{x_u + x_v} = sum_{(u,v) in E} (1 if x_u == x_v else -1)
        # = |E| - 2 * (number of edges cut in state x)
        
        # So phase = exp(-i*gamma/2 * |E|) * exp(i*gamma/2 * (|E| - 2*cut(x)))
        # = exp(i*gamma/2 * (|E| - 2*cut(x)) - i*gamma/2*|E|)
        # = exp(-i*gamma * cut(x))
        
        # So the phase for basis state |x> is exp(-i*gamma*cut(x)) where cut(x) is the number of edges cut by partition x.
        
        # Precompute cut(x) for all basis states
        # For efficiency, compute using the edge list
        phases = np.ones(dim, dtype=np.complex128)
        for x in range(dim):
            cut_count = 0
            for (u, v) in edges:
                xu = (x >> u) & 1
                xv = (x >> v) & 1
                if xu != xv:
                    cut_count += 1
            phases[x] = np.exp(-1j * gamma * cut_count)
        
        state = state * phases
        
        # Apply exp(-i * beta * sum_i X_i)
        # X = |0><1| + |1><0|
        # exp(-i*beta*X) on a single qubit = [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
        # For n qubits: prod_i exp(-i*beta*X_i)
        # This is a tensor product of 2x2 matrices.
        # We can apply it by iterating over each qubit.
        
        # Reshape state to tensor
        state_tensor = state.reshape([2] * n)
        
        for i in range(n):
            # Apply single-qubit gate to qubit i
            # Gate: [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
            c = np.cos(beta)
            s = -1j * np.sin(beta)
            gate = np.array([[c, s], [s, c]], dtype=np.complex128)
            
            # Move axis i to front, apply gate, move back
            state_tensor = np.moveaxis(state_tensor, i, 0)
            state_tensor = np.tensordot(gate, state_tensor, axes=([1], [0]))
            state_tensor = np.moveaxis(state_tensor, 0, i)
        
        state = state_tensor.reshape(dim)
    
    # Compute expected cut
    # E[cut] = sum_{(u,v) in edges} (1 - <Z_u Z_v>)/2
    # <Z_u Z_v> = sum_x (-1)^{x_u + x_v} |psi_x|^2
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for (u, v) in edges:
        # Compute <Z_u Z_v>
        zuv = 0.0
        for x in range(dim):
            xu = (x >> u) & 1
            xv = (x >> v) & 1
            sign = 1.0 if (xu == xv) else -1.0
            zuv += sign * probs[x]
        expected_cut += (1.0 - zuv) / 2.0
    
    return expected_cut


def _expected_cut_fast(n, edges, gammas, betas):
    """
    Faster expected cut computation using vectorized operations.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Compute cut count for all basis states vectorized
        cut_counts = np.zeros(dim, dtype=np.float64)
        for (u, v) in edges:
            # For each basis state, check if bits u and v differ
            mask_u = (np.arange(dim) >> u) & 1
            mask_v = (np.arange(dim) >> v) & 1
            cut_counts += (mask_u != mask_v).astype(np.float64)
        
        phases = np.exp(-1j * gamma * cut_counts)
        state = state * phases
        
        # Apply beta layer: tensor product of single-qubit rotations
        state_tensor = state.reshape([2] * n)
        
        for i in range(n):
            c = np.cos(beta)
            s = -1j * np.sin(beta)
            gate = np.array([[c, s], [s, c]], dtype=np.complex128)
            
            state_tensor = np.moveaxis(state_tensor, i, 0)
            state_tensor = np.tensordot(gate, state_tensor, axes=([1], [0]))
            state_tensor = np.moveaxis(state_tensor, 0, i)
        
        state = state_tensor.reshape(dim)
    
    # Compute expected cut vectorized
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for (u, v) in edges:
        mask_u = (np.arange(dim) >> u) & 1
        mask_v = (np.arange(dim) >> v) & 1
        signs = np.where(mask_u == mask_v, 1.0, -1.0)
        zuv = np.sum(signs * probs)
        expected_cut += (1.0 - zuv) / 2.0
    
    return expected_cut


def _objective(params, n, edges, p):
    """Objective function to minimize (negative expected cut)."""
    gammas = params[:p]
    betas = params[p:]
    cut = _expected_cut_fast(n, edges, gammas, betas)
    return -cut


def _optimize_graph(n, edges, p, time_budget=10.0):
    """
    Optimize QAOA angles for a specific graph using multiple random restarts
    with L-BFGS-B.
    """
    global _START_TIME
    
    best_angles = np.array([0.35, 0.25] * p)  # fallback
    best_cut = _expected_cut_fast(n, edges, best_angles[:p], best_angles[p:])
    
    # Try multiple random restarts
    n_restarts = 8
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        # Check time
        if _START_TIME is not None and (time.time() - _START_TIME) > _DEADLINE - 5:
            break
        
        # Random initialization in [0, pi]
        params0 = rng.uniform(0, np.pi, size=2 * p)
        
        # Use L-BFGS-B with bounds
        bounds = [(0, np.pi)] * (2 * p)
        
        try:
            result = minimize(
                _objective,
                params0,
                args=(n, edges, p),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and result.fun < -best_cut:
                best_cut = -result.fun
                best_angles = result.x
        except Exception:
            continue
    
    # Additional polish: try finer grid around best solution
    if _START_TIME is not None and (time.time() - _START_TIME) > _DEADLINE - 3:
        return best_angles
    
    # Local refinement with tighter tolerance
    try:
        bounds = [(0, np.pi)] * (2 * p)
        result = minimize(
            _objective,
            best_angles,
            args=(n, edges, p),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success and result.fun < -best_cut:
            best_cut = -result.fun
            best_angles = result.x
    except Exception:
        pass
    
    return best_angles


def angles(n, edges, p):
    """
    Compute optimal QAOA angles for the given graph.
    Returns array of shape (p, 2) with rows (gamma, beta).
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Optimize angles for this specific graph
    opt_params = _optimize_graph(n, edges, p)
    
    # Reshape to (p, 2) array: rows are (gamma, beta)
    gammas = opt_params[:p]
    betas = opt_params[p:]
    result = np.column_stack([gammas, betas])
    
    return result
# EVOLVE-BLOCK-END
