import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Strategy:
    1. Exact dense-matrix simulation with precomputed U_X matrices.
    2. Multi-start L-BFGS-B with numerical gradients (finite differences).
    3. Time-bounded search with safety margin.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # 1. Precompute cut values and optimal cut
    max_states = 1 << n
    cut_vals = np.zeros(max_states, dtype=np.float64)
    for s in range(max_states):
        c = 0
        for u, v in edges:
            if ((s >> u) ^ (s >> v)) & 1:
                c += 1
        cut_vals[s] = c
        
    opt_cut = np.max(cut_vals)
    if opt_cut == 0:
        return np.zeros((p, 2))

    # 2. Precompute initial state and U_X matrices for a grid of betas
    psi0 = np.ones(max_states, dtype=np.complex128)
    psi0 /= np.sqrt(max_states)
    
    def get_u_x(beta):
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        rx = np.array([[c, s], [s, c]], dtype=np.complex128)
        u = rx
        for _ in range(n - 1):
            u = np.kron(u, rx)
        return u

    def simulate(params):
        """
        params: array of length 2*p, [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}]
        Returns: expected_cut / opt_cut
        """
        gammas = params[:p]
        betas = params[p:]
        
        state = psi0.copy()
        
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase (diagonal)
            state = state * np.exp(-1j * g * cut_vals)
            
            # Apply X rotations
            u_x = get_u_x(b)
            state = u_x @ state
        
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        return expected_cut / opt_cut

    def objective(params):
        return -simulate(params)

    # 3. Multi-start L-BFGS-B
    best_value = -np.inf
    best_params = None
    
    # Generate multiple random starts
    np.random.seed(42)
    n_starts = 20
    
    for i in range(n_starts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization in [0, pi]
        params = np.random.uniform(0, np.pi, size=2*p)
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            val = -res.fun
            if val > best_value:
                best_value = val
                best_params = res.x
        except Exception:
            pass
    
    # Also try some structured initializations
    structured_starts = [
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.3, 0.3, 0.7, 0.7]),
        np.array([0.7, 0.3, 0.3, 0.7]),
        np.array([0.5, 0.2, 0.5, 0.8]),
        np.array([0.2, 0.5, 0.8, 0.5]),
    ]
    
    for params in structured_starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4})
            val = -res.fun
            if val > best_value:
                best_value = val
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        best_params = np.array([0.5, 0.5, 0.5, 0.5])

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T
