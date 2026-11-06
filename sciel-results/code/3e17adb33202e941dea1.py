import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0  # 10s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _build_cost_diag(n, edges):
    """
    Efficiently compute the diagonal of the MaxCut cost operator C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    """
    num_states = 1 << n
    c_diag = np.zeros(num_states, dtype=np.float64)
    
    # Pre-compute bit masks for edges to speed up the loop
    # Actually, the loop is small for n<=10.
    for state in range(num_states):
        cut = 0
        for u, v in edges:
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
    """
    c = np.cos(beta)
    s = np.sin(beta)
    
    shape = [2] * n
    st = state.reshape(shape)
    
    for axis in range(n):
        st = np.moveaxis(st, axis, 0)
        s0 = st[0]
        s1 = st[1]
        new0 = c * s0 - 1j * s * s1
        new1 = -1j * s * s0 + c * s1
        st = np.stack([new0, new1], axis=0)
        st = np.moveaxis(st, 0, axis)
    
    return st.reshape(-1)

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
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    
    g1, b1, g2, b2 = params
    
    # Layer 1
    state = _apply_cost(state, g1, c_diag)
    state = _apply_mixer(state, b1, n)
    
    # Layer 2
    state = _apply_cost(state, g2, c_diag)
    state = _apply_mixer(state, b2, n)
    
    probs = np.abs(state) ** 2
    return float(np.dot(probs, c_diag))

def _optimize_angles(n, edges):
    """
    Optimize QAOA angles for a given graph.
    Returns array of shape (2, 2) with [gamma, beta] pairs.
    """
    c_diag = _build_cost_diag(n, edges)
    
    # Objective function to minimize (negative expected cut)
    def neg_cut(params):
        return -_expected_cut(params, n, c_diag)
    
    best_val = -np.inf
    best_params = np.array([0.5, 0.5, 0.5, 0.5])
    
    # Define a set of initial guesses
    # 1. Asymptotic-inspired guesses
    # 2. Random guesses
    # 3. Small perturbations around known good regions
    
    starts = []
    
    # Known approximate optimal regions for p=2 MaxCut
    # g1 ~ 0.3-0.6, b1 ~ 0.2-0.4, g2 ~ 0.1-0.3, b2 ~ 0.1-0.2
    starts.append([0.4, 0.3, 0.2, 0.1])
    starts.append([0.5, 0.4, 0.3, 0.2])
    starts.append([0.6, 0.5, 0.4, 0.3])
    
    # Add random starts
    np.random.seed(42) # Fixed seed for reproducibility within a run? 
                       # Actually, different graphs might benefit from different random starts.
                       # But we want determinism for the same graph.
                       # The task says "seed 713" for eval graphs, so we should be deterministic.
    
    for _ in range(10):
        starts.append(np.random.uniform(0.05, np.pi - 0.05, 4))
    
    # Run BFGS from each start
    for x0 in starts:
        if _time_left() < 2.0:
            break
            
        try:
            res = minimize(neg_cut, x0, method='BFGS', 
                           options={'maxiter': 200, 'gtol': 1e-8, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if _time_left() > 1.0:
        try:
            res_polish = minimize(neg_cut, best_params, method='BFGS',
                                  options={'maxiter': 500, 'gtol': 1e-10, 'disp': False})
            val_polish = -res_polish.fun
            if val_polish > best_val:
                best_params = res_polish.x
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
        return np.tile([0.5, 0.5], (p, 1))
    
    try:
        angles_2 = _optimize_angles(n, edges)
        
        if p == 2:
            return angles_2
        elif p < 2:
            return angles_2[:p]
        else:
            reps = (p + 1) // 2
            return np.tile(angles_2, (reps, 1))[:p]
            
    except Exception:
        return np.tile([0.5, 0.5], (p, 1))
