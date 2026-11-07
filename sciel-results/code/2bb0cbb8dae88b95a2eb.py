import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a learnable Radial Basis Function (RBF) network.
    
    Strategy:
    - Uses 5 Gaussian RBFs with learnable centers and widths.
    - The 10 parameters are split into 5 centers (c_i) and 5 widths (sigma_i).
    - This allows the optimizer to adaptively place and shape local features,
      providing local flexibility that can better capture non-linear structures
      than fixed-basis approaches.
    - Each RBF has a fixed amplitude of 1.0 (implicit), so only position and scale
      are learned, keeping the problem well-posed.
    """
    # Split params into 5 centers and 5 widths
    centers = params[:5]  # c_1, ..., c_5
    widths = params[5:]   # sigma_1, ..., sigma_5
    
    # Ensure widths are positive to avoid division by zero or negative variance
    # Use a softplus-like transformation or absolute value + small epsilon
    # Here we use abs + 0.1 to ensure sigma > 0.1
    widths = np.abs(widths) + 0.1
    
    # Compute RBF values: phi_i(x) = exp(-(x - c_i)^2 / sigma_i^2)
    # x shape: (N,)
    # centers shape: (5,)
    # widths shape: (5,)
    
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # rbf_values shape: (N, 5)
    rbf_values = np.exp(-(diff ** 2) / (widths[np.newaxis, :] ** 2))
    
    # Each RBF has amplitude 1.0, so prediction is just the sum of the RBF values
    # But we need to use all 10 params meaningfully. Let's use the centers and widths
    # as the only parameters, and the amplitudes are fixed at 1.0.
    # However, this means the model is y = sum_i exp(-(x-c_i)^2 / sigma_i^2)
    # This is a valid model, but it doesn't use a linear combination.
    
    # Actually, the task says params is an array of length 10. The evaluator fits
    # params to minimize MSE. If we fix amplitudes to 1, then we're only learning
    # centers and widths, which is non-linear. BFGS can handle this.
    
    # But wait: if amplitudes are fixed at 1, then the model is:
    # y(x) = sum_{i=1}^{5} exp(-(x - c_i)^2 / sigma_i^2)
    # This is a valid model. Let's use this.
    
    y = rbf_values.sum(axis=1)
    
    return y
