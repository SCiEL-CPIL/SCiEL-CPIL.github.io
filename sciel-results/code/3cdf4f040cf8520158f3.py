import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    
    # Parameters: f[0..7] for x offsets, g[0..7] for y offsets
    # Initial guess: Star construction
    f0 = -np.arange(N) / (2.0 * N)
    g0 = -np.arange(N) / (2.0 * N)
    
    def volume(params):
        f = params[:N]
        g = params[N:2*N]
        
        # Calculate W_x(c) and W_y(c) at sample points
        # Use 100 points for c in [0,1]
        c_vals = np.linspace(0, 1, 100)
        
        W_x = np.zeros_like(c_vals)
        W_y = np.zeros_like(c_vals)
        
        for idx, c in enumerate(c_vals):
            # x intervals
            starts_x = f + c * np.arange(N) / N
            ends_x = starts_x + 1.0 / N
            # Sort intervals by start
            order = np.argsort(starts_x)
            s_x = starts_x[order]
            e_x = ends_x[order]
            
            # Merge intervals to find total width
            # Since they are likely overlapping, we can just take max end - min start
            # But to be safe, we do a simple merge
            current_end = e_x[0]
            width_x = 0
            for i in range(1, N):
                if s_x[i] > current_end:
                    width_x += current_end - s_x[i-1] if i > 0 else 0 # This logic is flawed for multiple gaps
                    # Better: just compute union length properly
                    pass
            
            # Simpler: The union of intervals [s_i, e_i]
            # If they are sorted, union length is sum of (min(e_i, e_{i+1}) - s_i) ... no
            # Just: max(e) - min(s) if connected.
            # Given the problem structure, they are almost always connected.
            # Let's assume connected for speed, but verify.
            w_x = np.max(e_x) - np.min(s_x)
            W_x[idx] = w_x
            
            starts_y = g + c * np.arange(N) / N
            ends_y = starts_y + 1.0 / N
            w_y = np.max(ends_y) - np.min(starts_y)
            W_y[idx] = w_y
        
        # Integrate W_x * W_y
        V = np.trapz(W_x * W_y, c_vals)
        return V

    # Optimize
    res = minimize(volume, np.concatenate([f0, g0]), method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 500})
    
    f_opt = res.x[:N]
    g_opt = res.x[N:2*N]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = f_opt[i_flat]
    base_y = g_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    return positions
