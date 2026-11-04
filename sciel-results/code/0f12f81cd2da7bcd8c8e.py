import numpy as np
from scipy.optimize import minimize
import sys

def construct():
    N = 600
    L = 0.5  # Length of interval [-0.25, 0.25]
    M = 40   # Increased number of harmonics for better representation
    
    # Grids for optimization
    n_x = 1024  # Higher resolution for non-negativity constraint
    x_grid = np.linspace(-L/2, L/2, n_x, endpoint=False)
    
    n_t = 2048  # Higher resolution for autoconvolution peak
    t_grid = np.linspace(-L/2, L/2, n_t, endpoint=False)
    
    # Precompute basis matrices
    # A: shape (n_x, M+1) - basis for f(x) >= 0 constraint
    A = np.zeros((n_x, M + 1))
    A[:, 0] = np.sqrt(2.0)
    for k in range(1, M + 1):
        A[:, k] = 2.0 * np.cos(2.0 * np.pi * k * x_grid / L)
        
    # B: shape (n_t, M+1) - basis for autoconvolution
    B = np.zeros((n_t, M + 1))
    B[:, 0] = 1.0
    for k in range(1, M + 1):
        B[:, k] = np.cos(2.0 * np.pi * k * t_grid / L)
        
    def compute_g_vals(u):
        u_sq = u**2
        return 2.0 * (B @ u_sq)
    
    def objective_and_gradient(u):
        g_vals = compute_g_vals(u)
        idx_max = np.argmax(g_vals)
        grad = 4.0 * u * B[idx_max, :]
        return g_vals[idx_max], grad

    def constraint_energy(u):
        return np.dot(u, u) - 1.0
        
    def constraint_energy_grad(u):
        return 2.0 * u

    def constraint_nonneg(u):
        return A @ u

    # Initial guess: Uniform function
    u0 = np.zeros(M + 1)
    u0[0] = 1.0
    
    # Multiple restarts with diverse initial guesses
    best_score = np.inf
    best_u = u0.copy()
    
    # Create diverse initial guesses
    initial_guesses = [u0.copy()]
    
    # Add some structured guesses
    # Triangle-like: emphasize low frequencies
    for alpha in [0.1, 0.2, 0.3]:
        u_struct = np.zeros(M + 1)
        u_struct[0] = np.sqrt(1.0 - alpha)
        for k in range(1, min(5, M + 1)):
            u_struct[k] = np.sqrt(alpha / 4.0) * np.cos(np.pi * k / 5.0)
        norm = np.linalg.norm(u_struct)
        if norm > 1e-10:
            u_struct /= norm
            initial_guesses.append(u_struct)
    
    # Add random guesses
    np.random.seed(42)
    for _ in range(8):
        u_rand = np.random.randn(M + 1)
        u_rand /= np.linalg.norm(u_rand)
        initial_guesses.append(u_rand)
    
    n_restarts = len(initial_guesses)
    
    for r in range(n_restarts):
        u_start = initial_guesses[r].copy()
        
        # Define constraints for SLSQP
        cons = [
            {'type': 'eq', 'fun': constraint_energy, 'jac': constraint_energy_grad},
            {'type': 'ineq', 'fun': constraint_nonneg, 'jac': lambda u, jac: A}
        ]
        
        try:
            res = minimize(
                lambda u: objective_and_gradient(u)[0],
                u_start,
                method='SLSQP',
                jac=lambda u: objective_and_gradient(u)[1],
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-8}
            )
            
            # Check result even if not "success" - sometimes it converges well
            u_opt = res.x
            # Normalize energy
            energy = np.dot(u_opt, u_opt)
            if energy > 1e-10:
                u_opt = u_opt / np.sqrt(energy)
                
                # Check non-negativity with tolerance
                f_vals = A @ u_opt
                if np.all(f_vals >= -1e-4):
                    # Calculate score
                    u_sq = u_opt**2
                    g_vals = 2.0 * (B @ u_sq)
                    score = np.max(g_vals)
                    if score < best_score:
                        best_score = score
                        best_u = u_opt.copy()
                        
        except Exception as e:
            pass
    
    # If optimization failed or didn't improve, use best_u (which might be u0)
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Evaluate f at centers
    f_centers = np.zeros(N)
    f_centers += best_u[0] * np.sqrt(2.0)
    for k in range(1, M + 1):
        f_centers += best_u[k] * 2.0 * np.cos(2.0 * np.pi * k * centers / L)
        
    # Ensure non-negative
    f_centers = np.maximum(0, f_centers)
    
    return list(f_centers)

if __name__ == "__main__":
    heights = construct()
    assert len(heights) == 600
    assert all(h >= 0 for h in heights)
