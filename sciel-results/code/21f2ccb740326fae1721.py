import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Fixed centers
        x_c = np.array([-0.2, 0.17])
        v_c = np.array([-0.34, -0.17, 0.0, 0.18, 0.36])
        
        # Create grid of centers
        X_grid, V_grid = np.meshgrid(x_c, v_c, indexing='ij')
        centers_x = X_grid.flatten()
        centers_v = V_grid.flatten()
        
        # Ensure 10 centers
        # 2 * 5 = 10
        
        # Flatten inputs
        x_flat = x.flatten()
        v_flat = v.flatten()
        
        # Compute distances
        # d_i = (x - x_i)^2 + (v - v_i)^2
        # Shape: (N, 10)
        dx = x_flat[:, None] - centers_x[None, :]
        dv = v_flat[:, None] - centers_v[None, :]
        dist_sq = dx**2 + dv**2
        
        sigma = 0.2
        rbfs = np.exp(-dist_sq / (2 * sigma**2))
        
        # Linear term
        # 1 + x + v
        linear_term = 1.0 + x_flat + v_flat
        
        # Basis: rbfs * linear_term
        # Shape: (N, 10)
        basis = rbfs * linear_term[:, None]
        
        # Prediction
        pred = basis @ params
        
        return pred.reshape(x.shape)
