"""Program for minimizing the normalized maximum autoconvolution of a step function."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Represent the step function as h = |z|^2 where z is a complex vector.
    This enforces nonnegativity and smoothness. We minimize the maximum of the
    autoconvolution using gradient descent in the complex domain.
    """
    
    def autoconvolution_max(h):
        """Compute the maximum of the autoconvolution of step function h."""
        n = len(h)
        conv = np.convolve(h, h, mode='full')
        return np.max(conv) / n
    
    def objective_and_grad(z):
        """
        Compute objective and gradient with respect to z (complex).
        h = |z|^2, normalized so that sum(h) = N.
        """
        h = np.abs(z)**2
        h_sum = np.sum(h)
        if h_sum < 1e-10:
            return 1e10, np.zeros_like(z)
        
        # Normalize: L1 norm = 1, i.e., sum(h) = N
        h_norm = h * (N / h_sum)
        
        # Compute autoconvolution
        conv = np.convolve(h_norm, h_norm, mode='full')
        obj = np.max(conv) / N
        
        # Find where the maximum occurs
        max_idx = np.argmax(conv)
        
        # Gradient with respect to h
        # d(conv[k])/d(h[i]) = h_norm[j] + h_norm[i] where j = k - i (for linear conv)
        # conv[k] = sum_i h_norm[i] * h_norm[k-i] (with appropriate bounds)
        # d(conv[k])/d(h_norm[i]) = h_norm[k-i] + h_norm[i] (if both in range)
        
        # Let's compute gradient numerically for accuracy
        eps = 1e-5
        grad_h = np.zeros(N)
        
        # Use central differences on the normalized h
        for i in range(N):
            h_plus = h_norm.copy()
            h_plus[i] += eps
            h_plus = h_plus * (N / np.sum(h_plus))
            conv_plus = np.convolve(h_plus, h_plus, mode='full')
            f_plus = np.max(conv_plus) / N
            
            h_minus = h_norm.copy()
            h_minus[i] = max(h_minus[i] - eps, 0)
            h_minus = h_minus * (N / np.sum(h_minus))
            conv_minus = np.convolve(h_minus, h_minus, mode='full')
            f_minus = np.max(conv_minus) / N
            
            grad_h[i] = (f_plus - f_minus) / (2 * eps)
        
        # Chain rule: d(obj)/d(z) = 2 * conj(z) * d(obj)/d(h)
        # But h = |z|^2, so d(h[i])/d(z[i]) = 2*conj(z[i]) (real part) 
        # Actually for complex: d(|z|^2)/dz = 2*conj(z), d(|z|^2)/d(conj(z)) = 2*z
        # For gradient descent in complex domain, we use the Wirtinger derivative
        # The gradient with respect to z (treating z and conj(z) as independent)
        # is 2*conj(z) * grad_h
        
        grad_z = 2 * np.conj(z) * grad_h
        
        return obj, grad_z
    
    # Initialize z with random complex values
    np.random.seed(42)
    z = np.ones(N, dtype=complex) + 0j
    z = z + 0.1 * (np.random.randn(N) + 1j * np.random.randn(N))
    
    # Gradient descent with adaptive learning rate
    lr = 0.01
    best_obj = 1e10
    best_z = z.copy()
    
    for iteration in range(300):
        obj, grad_z = objective_and_grad(z)
        
        if obj < best_obj:
            best_obj = obj
            best_z = z.copy()
        
        # Gradient step
        z = z - lr * grad_z
        
        # Adaptive learning rate
        if iteration % 50 == 0 and iteration > 0:
            lr *= 0.9
        
        # Normalize to keep values reasonable
        z_mag = np.abs(z)
        if np.max(z_mag) > 10:
            z = z * (10 / np.max(z_mag))
    
    # Extract h from best_z
    h = np.abs(best_z)**2
    h = h * (N / (np.sum(h) + 1e-10))
    
    return list(h)
# EVOLVE-BLOCK-END
