import numpy as np
import time
import scipy.optimize

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Bivariate Gaussian Radial Basis Function (RBF) Network.
    
    This is a structural change from the current linear-in-parameters basis expansion 
    to a non-linear, localized kernel model. Instead of global basis functions (polynomials, 
    powers), we use 5 Gaussian RBFs centered at specific points in the (strain, temp) space.
    
    The 10 parameters are mapped as follows:
    - p[0:5] : Amplitudes (weights) for the 5 RBFs.
    - p[5:10] : Inverse widths (1/sigma^2) for the 5 RBFs.
    
    The centers are fixed to capture the key features of the stress-strain curve:
    1. (0.0, 0.5) : Elastic region / Yield point area.
    2. (0.2, 0.5) : Early hardening.
    3. (0.5, 0.5) : Mid hardening.
    4. (0.8, 0.5) : Late hardening.
    5. (0.0, 0.1) : Low temperature / High stiffness region (to capture T-dependence).
    
    Why this might work better:
    1. Localized nature: RBFs can independently adjust to different regions of the curve 
       (e.g., sharp knee vs. smooth hardening) without interference.
    2. Flexibility: The widths allow the model to zoom in on sharp features.
    3. Non-linearity: The parameters interact non-linearly with the input, but the 
       landscape for BFGS starting from ones might still be navigable if the centers 
       are well-placed. The amplitudes start at 1, which is a reasonable scale for stress 
       if the data is normalized or if the stress values are O(1). 
       
    Note: The inputs strain and temp are in the ranges [-0.002, 0.96] and [0.07, 1.0] respectively.
    We normalize them internally to [0, 1] or similar to ensure the Gaussian distances are well-scaled.
    Strain is mostly positive, but can be negative. We will map strain to [0, 1] range roughly.
    Actually, let's keep the raw values but scale the distances appropriately.
    Strain range ~1. Temp range ~1.
    A width of 1.0 (inverse width 1.0) means sigma=1.0, which covers the whole domain.
    Starting inverse widths at 1.0 means broad kernels. BFGS can sharpen them if needed.
    """
    
    s = np.asarray(strain, dtype=np.float64)
    t = np.asarray(temp, dtype=np.float64)
    
    p = np.asarray(params, dtype=np.float64).flatten()
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(s)
        
    # Extract parameters
    # First 5 are amplitudes, next 5 are inverse widths (lambda = 1/sigma^2)
    amps = p[0:5]
    inv_widths = p[5:10]
    
    # Ensure inverse widths are positive to avoid negative variance in Gaussian
    # If BFGS drives them negative, the Gaussian becomes exp(+large) -> inf.
    # Clip to a small positive value.
    inv_widths = np.clip(inv_widths, 1e-4, 100.0)
    
    # Define fixed centers in (strain, temp) space
    # These are chosen to cover the domain and key physics regions.
    # Strain range: ~0 to 1. Temp range: ~0.1 to 1.0
    centers = np.array([
        [0.0, 0.5],   # Yield/Elastic
        [0.2, 0.5],   # Early Hardening
        [0.5, 0.5],   # Mid Hardening
        [0.8, 0.5],   # Late Hardening
        [0.0, 0.1]    # Low Temp / High Stiffness (approx)
    ])
    
    # Initialize prediction
    pred = np.zeros_like(s)
    
    for i in range(5):
        # Distance squared between input (s, t) and center (c_s, c_t)
        ds = s - centers[i, 0]
        dt = t - centers[i, 1]
        dist_sq = ds * ds + dt * dt
        
        # Gaussian kernel: exp(-lambda * dist_sq)
        # lambda is the inverse width
        lambda_i = inv_widths[i]
        
        # Clip the exponent to avoid overflow/underflow
        # exp(-large) -> 0, exp(+large) -> inf. Since lambda > 0 and dist_sq >= 0, 
        # the argument is always <= 0. So we only need to worry about underflow to 0, 
        # which is fine. But if lambda is very large and dist_sq is small, it's fine.
        # The main risk is if lambda becomes negative (clipped) or if dist_sq calculation fails.
        # With clipping, lambda is positive.
        
        arg = -lambda_i * dist_sq
        # Clip arg to [-700, 0] to be safe, though it should naturally be <= 0
        arg = np.clip(arg, -700, 0)
        
        kernel = np.exp(arg)
        
        pred += amps[i] * kernel
        
    return np.where(np.isfinite(pred), pred, 0.0)
