import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    # Parameters
    M = 15  # Number of harmonics
    L = 0.5  # Length of the interval [-1/4, 1/4]
    
    # Discretization for checking constraints and computing objective
    n_grid = 2048
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    
    # Time grid for autoconvolution g(t)
    # g(t) is supported on [-L, L] = [-0.5, 0.5]
    n_t_grid = 4096
    t = np.linspace(-L, L, n_t_grid, endpoint=False)
    
    # Precompute basis values for f(x) and g(t)
    # f(x) = sum_{k=0}^M u_k * phi_k(x)
    # phi_0(x) = 1/sqrt(L)
    # phi_k(x) = sqrt(2/L) * cos(2π k x / L) for k >= 1
    
    # Precompute the basis matrix for f
    F_matrix = np.zeros((n_grid, M+1))
    F_matrix[:, 0] = 1.0 / np.sqrt(L)
    for k in range(1, M+1):
        F_matrix[:, k] = np.sqrt(2.0/L) * np.cos(2*np.pi*k*x/L)
    
    # Precompute the basis for g(t)
    # g(t) = (1/L) [ u_0^2 + sum_{k=1}^M u_k^2 cos(2π k t / L) ]
    G_cos = np.zeros((n_t_grid, M))
    for k in range(1, M+1):
        G_cos[:, k-1] = np.cos(2*np.pi*k*t/L)
    
    def compute_f(u):
        """Compute f(x) for given coefficients u."""
        return F_matrix @ u
    
    def compute_g(u):
        """Compute g(t) for given coefficients u."""
        g_vals = np.full(n_t_grid, u[0]**2 / L)
        for k in range(1, M+1):
            g_vals += (u[k]**2 / L) * G_cos[:, k-1]
        return g_vals
    
    def objective_and_grad(u):
        """Compute objective (max g + penalty) and gradient."""
        f_vals = compute_f(u)
        g_vals = compute_g(u)
        
        # Penalty for negative f
        neg_f = np.maximum(0, -f_vals)
        penalty = np.sum(neg_f**2)
        
        obj = np.max(g_vals) + 100.0 * penalty
        
        # Gradient of max(g) at the point of maximum
        idx_max = np.argmax(g_vals)
        t_max = t[idx_max]
        
        grad_g = np.zeros(M+1)
        grad_g[0] = 2 * u[0] / L
        for k in range(1, M+1):
            grad_g[k] = 2 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
        
        # Gradient of penalty
        grad_pen = np.zeros(M+1)
        mask_neg = f_vals < 0
        if np.any(mask_neg):
            # grad_pen[k] = -2 * sum_{j in mask_neg} f_vals[j] * F_matrix[j, k]
            grad_pen = -2.0 * (F_matrix[mask_neg].T @ f_vals[mask_neg])
        
        grad_total = grad_g + 100.0 * grad_pen
        
        return obj, grad_total
    
    # Initial guess: uniform function (u_0 = 1, others 0)
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    # Use scipy.optimize.minimize with L-BFGS-B
    # We need to handle the sphere constraint ||u||=1
    # We can use a penalty method or reparameterize
    
    # Let's use a penalty method with the sphere constraint
    def objective_sphere(u):
        # Penalty for not being on the sphere
        sphere_penalty = 1000.0 * (np.linalg.norm(u) - 1.0)**2
        obj, grad = objective_and_grad(u)
        # Gradient of sphere penalty
        grad_sphere = 2000.0 * (np.linalg.norm(u) - 1.0) * u / max(np.linalg.norm(u), 1e-12)
        return obj + sphere_penalty, grad + grad_sphere
    
    best_score = np.inf
    best_u = u0.copy()
    
    # Run multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if restart == 0:
            u_init = u0.copy()
        else:
            # Random initialization on the sphere
            u_init = np.random.randn(M+1)
            u_init /= np.linalg.norm(u_init)
        
        # Use L-BFGS-B
        result = minimize(objective_sphere, u_init, method='L-BFGS-B', 
                         jac=True, options={'maxiter': 500, 'ftol': 1e-10})
        
        u_opt = result.x
        # Project to sphere
        norm_u = np.linalg.norm(u_opt)
        if norm_u > 1e-12:
            u_opt /= norm_u
        
        # Check if feasible
        f_check = compute_f(u_opt)
        if np.min(f_check) >= -1e-6:
            g_check = compute_g(u_opt)
            score = np.max(g_check)
            if score < best_score:
                best_score = score
                best_u = u_opt.copy()
    
    # Also try a simple gradient descent approach as backup
    lr = 0.01
    n_iter = 200
    for restart in range(n_restarts):
        if restart == 0:
            u = u0.copy()
        else:
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
        
        for i in range(n_iter):
            f_vals = compute_f(u)
            g_vals = compute_g(u)
            
            idx_max = np.argmax(g_vals)
            t_max = t[idx_max]
            
            grad_g = np.zeros(M+1)
            grad_g[0] = 2 * u[0] / L
            for k in range(1, M+1):
                grad_g[k] = 2 * u[k] / L * np.cos(2*np.pi*k*t_max/L)
            
            grad_pen = np.zeros(M+1)
            mask_neg = f_vals < 0
            if np.any(mask_neg):
                grad_pen = -2.0 * (F_matrix[mask_neg].T @ f_vals[mask_neg])
            
            grad_total = grad_g + 50.0 * grad_pen
            
            u = u - lr * grad_total
            norm_u = np.linalg.norm(u)
            if norm_u > 1e-12:
                u /= norm_u
            
            f_check = compute_f(u)
            if np.min(f_check) >= -1e-6:
                g_check = compute_g(u)
                score = np.max(g_check)
                if score < best_score:
                    best_score = score
                    best_u = u.copy()
    
    # Convert best_u to step heights
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    heights = np.zeros(N)
    heights += best_u[0] / np.sqrt(L)
    for k in range(1, M+1):
        heights += best_u[k] * np.sqrt(2.0/L) * np.cos(2*np.pi*k*centers/L)
    
    # Ensure nonnegativity
    heights = np.maximum(0, heights)
    
    return list(heights)
