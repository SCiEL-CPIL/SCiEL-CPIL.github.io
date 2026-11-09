"""Program for spectral graph lifts task."""
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    def is_valid_lift(signs):
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, None
    
    # Phase-based optimization
    # Each vertex v has a phase theta_v in [0, 2*pi)
    # Edge (u,v) sign = sign(cos(theta_u - theta_v))
    # Optimize continuous phases to minimize spectral radius of weighted adjacency
    # where weights are cos(theta_u - theta_v), then discretize.
    
    def objective_and_grad(thetas):
        # Build weighted adjacency with cos weights
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            w = np.cos(thetas[u] - thetas[v])
            A[u, v] = w
            A[v, u] = w
        
        # Spectral radius = max |eigenvalue|
        eigvals, eigvecs = np.linalg.eigh(A)
        idx = np.argmax(np.abs(eigvals))
        r = np.abs(eigvals[idx])
        e = eigvecs[:, idx]
        
        # Gradient computation
        # lambda = e^T A e = 2 * sum_{(u,v) in E} w_uv * e[u] * e[v]
        # d(lambda)/dtheta_k = 2 * sum_{(u,v) in E, k in {u,v}} dw_uv/dtheta_k * e[u]*e[v]
        # dw_uv/dtheta_u = -sin(theta_u - theta_v)
        # dw_uv/dtheta_v = sin(theta_u - theta_v)
        
        grad = np.zeros(n)
        for i, (u, v) in enumerate(edges):
            sin_diff = np.sin(thetas[u] - thetas[v])
            e_uv = e[u] * e[v]
            grad[u] += -2.0 * sin_diff * e_uv
            grad[v] += 2.0 * sin_diff * e_uv
        
        return r, grad
    
    best_signs = None
    best_radius = np.inf
    rng = np.random.default_rng(3)
    
    # Multiple random restarts for phase optimization
    for restart in range(20):
        if restart == 0:
            thetas0 = np.zeros(n)
        elif restart == 1:
            thetas0 = np.pi * rng.random(n)
        else:
            thetas0 = rng.uniform(0, 2 * np.pi, n)
        
        try:
            result = minimize(objective_and_grad, thetas0, method='L-BFGS-B',
                             bounds=[(0, 2 * np.pi)] * n,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            thetas_opt = result.x
            # Discretize to signs
            signs = np.ones(E, dtype=int)
            for i, (u, v) in enumerate(edges):
                cos_val = np.cos(thetas_opt[u] - thetas_opt[v])
                signs[i] = 1 if cos_val >= 0 else -1
            
            valid, radius = is_valid_lift(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        except Exception:
            continue
    
    # Fallback: seed's random samples
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=E)
        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Local search refinement
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        for iteration in range(150):
            improved = False
            for i in range(E):
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                valid, radius = is_valid_lift(current_signs)
                if valid and radius < current_radius:
                    current_radius = radius
                    improved = True
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
            if not improved:
                break
    
    if best_signs is None:
        best_signs = np.ones(E, dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
