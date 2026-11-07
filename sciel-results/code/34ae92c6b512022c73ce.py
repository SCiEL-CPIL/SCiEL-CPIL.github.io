import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a fixed-center Radial Basis Function (RBF) basis.
    
    Strategy:
    - Uses 10 Gaussian RBFs with fixed centers and widths.
    - The problem reduces to a linear least-squares fit for the 10 parameters.
    - This allows BFGS to converge rapidly and reliably to the global minimum 
      for the linear weights, avoiding local minima associated with non-linear 
      parameter optimization.
    - RBFs provide local flexibility, handling non-linear features better than 
      global polynomials (Chebyshev) or Fourier series for non-periodic data.
    """
    # Fixed centers distributed across the domain [-2, 2]
    # We use 10 centers. 
    # centers = np.linspace(-2, 2, 10) might be too regular if the function is localized,
    # but for general symbolic regression, uniform coverage is a safe default.
    centers = np.array([-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8])
    
    # Fixed widths (sigma). 
    # A moderate width allows overlap between neighbors for smoothness.
    # If sigma is too small, the basis is ill-conditioned. If too large, it's like a polynomial.
    # Let's use a sigma that provides good local adaptation.
    sigma = 0.6
    
    # Compute RBF values: phi(x, c) = exp(-((x - c)^2) / (2 * sigma^2))
    # x shape: (N,)
    # centers shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    
    # rbf_values shape: (N, 10)
    # params shape: (10,)
    # Prediction = rbf_values @ params
    y = rbf_values @ params
    
    return y
