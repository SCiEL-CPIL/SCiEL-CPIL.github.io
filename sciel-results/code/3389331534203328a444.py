import numpy as np
import time
from scipy.optimize import minimize

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Strategy:
    1. Precompute the exact unitary operators for the C phase and X rotations.
    2. Use a dense-matrix formulation for the QAOA evolution.
    3. Perform a coarse grid search to find a good initial region.
    4. Polish the result with L-BFGS-B for high precision.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
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

    # 2. Precompute Unitaries
    # Initial state |+>^n
    psi0 = np.ones(max_states, dtype=np.complex128)
    psi0 /= np.sqrt(max_states)
    
    # C Phase Operator: U_C(gamma) = exp(-i gamma C)
    # This is diagonal. We can just multiply the state by exp(-i gamma cut_vals).
    
    # X Rotation Operator: U_X(beta) = exp(-i beta sum X_i) = tensor product of Rx(beta)
    # Rx(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    # For n<=10, we can precompute the full matrix U_X(beta) for a given beta, 
    # but beta changes. However, we can build the matrix quickly using kron.
    
    def get_u_x(beta):
        # Rx(beta) matrix
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        rx = np.array([[c, s], [s, c]], dtype=np.complex128)
        # Tensor product of n copies
        u = rx
        for _ in range(n - 1):
            u = np.kron(u, rx)
        return u

    def get_u_c(gamma):
        # Diagonal matrix exp(-i gamma cut_vals)
        # We don't need to build the full matrix, just the phase vector
        return np.exp(-1j * gamma * cut_vals)

    def simulate_and_grad(params):
        """
        params: array of length 2*p, [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}]
        Returns: (value, gradient)
        """
        gammas = params[:p]
        betas = params[p:]
        
        # State vector
        state = psi0.copy()
        
        # Forward pass
        for k in range(p):
            g = gammas[k]
            b = betas[k]
            
            # Apply C phase
            phase_c = get_u_c(g)
            state = state * phase_c
            
            # Apply X rotations
            u_x = get_u_x(b)
            state = u_x @ state
        
        # Final state is state
        # Expected cut
        probs = np.abs(state) ** 2
        expected_cut = np.sum(probs * cut_vals)
        value = expected_cut / opt_cut
        
        return value

    def objective(params):
        return -simulate_and_grad(params)

    # Optimization Strategy
    # 1. Coarse grid search
    # 2. Polish with L-BFGS-B
    
    best_value = -np.inf
    best_params = None
    
    # Grid search parameters
    # For p=2, we have 4 parameters: g0, b0, g1, b1
    # Typical ranges: [0, pi]
    # Let's use a coarse grid first
    grid_size = 5  # 5 points per parameter -> 5^4 = 625 evaluations
    # 625 * (matrix mult cost)
    # Matrix mult for n=10: 1024x1024 @ 1024x1 -> ~1M flops. Very fast.
    # 625 * 1M = 625M flops. Should be < 1 second.
    
    g_vals = np.linspace(0, np.pi, grid_size)
    b_vals = np.linspace(0, np.pi, grid_size)
    
    # Precompute U_X for all grid betas to avoid rebuilding
    u_x_grid = {b: get_u_x(b) for b in b_vals}
    
    for g0 in g_vals:
        for b0 in b_vals:
            for g1 in g_vals:
                for b1 in b_vals:
                    if time.time() > deadline - 10:
                        break
                    params = np.array([g0, b0, g1, b1])
                    # Manual simulation for speed in grid
                    state = psi0.copy()
                    state = state * get_u_c(g0)
                    state = u_x_grid[b0] @ state
                    state = state * get_u_c(g1)
                    state = u_x_grid[b1] @ state
                    probs = np.abs(state) ** 2
                    val = np.sum(probs * cut_vals) / opt_cut
                    if val > best_value:
                        best_value = val
                        best_params = params
    
    # Polish with L-BFGS-B
    if best_params is not None:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
            if -res.fun > best_value:
                best_value = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_params = np.array([0.5, 0.5, 0.5, 0.5])

    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.array([gammas, betas]).T
