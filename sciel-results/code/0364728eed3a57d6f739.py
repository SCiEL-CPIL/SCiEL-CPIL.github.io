import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Interval width
    dx = 1.0 / 1200.0
    
    def compute_score(heights):
        """Compute the normalized max autoconvolution score."""
        # Ensure nonnegativity
        h = np.maximum(heights, 0.0)
        
        # Autoconvolution via FFT
        # The autoconvolution of a step function
        # (f*f)(k) = sum_i h_i * h_{i+k} * dx for appropriate k
        # Using FFT for efficiency
        n_fft = 2 * N + 1
        f_fft = np.fft.rfft(h, n=n_fft)
        conv = np.fft.irfft(f_fft * np.conj(f_fft), n=n_fft)
        # The autoconvolution values
        # conv[j] corresponds to (f*f) at offset j*dx approximately
        # But we need to be careful about the indexing
        
        # Actually, for a discrete step function, the autoconvolution at lag k is:
        # sum_i h[i] * h[i+k] for valid i, times dx
        # The max should be at k=0 typically, but let's check
        
        # Using the FFT result: conv has the circular convolution
        # For autoconvolution, we want the linear one
        # conv[0] = sum h[i]^2 * dx (this is (f*f)(0))
        # conv[k] for k>0 gives the values at positive lags
        
        max_conv = np.max(conv)
        energy = np.sum(h**2) * dx
        
        if energy < 1e-15:
            return 1e10
        return max_conv / energy
    
    # Parameterize using a low-dimensional representation
    # Try: f(x) = a + b*x^2 + c*x^4 on [-1/4, 1/4]
    # Or use a sum of Gaussians
    
    best_score = compute_score(np.ones(N))
    best_heights = np.ones(N)
    
    # Try several initializations and optimize
    rng = np.random.RandomState(42)
    
    def param_to_heights(params):
        """Convert parameters to a height vector."""
        x = np.linspace(-0.25, 0.25, N)
        # Polynomial: a + b*x^2 + c*x^4
        a, b, c = params
        h = a + b * x**2 + c * x**4
        return np.maximum(h, 0.0)
    
    def objective(params):
        h = param_to_heights(params)
        return compute_score(h)
    
    # Try polynomial parameterization
    bounds_poly = [(0.1, 2.0), (-1.0, 1.0), (-0.5, 0.5)]
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x0 = [1.0, 0.0, 0.0]  # start near constant
        else:
            x0 = [rng.uniform(0.1, 2.0), rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5)]
        
        try:
            res = minimize(objective, x0, bounds=bounds_poly, method='L-BFGS-B', 
                          options={'maxiter': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_heights = param_to_heights(res.x)
        except:
            pass
    
    # Try Gaussian mixture parameterization
    # f(x) = sum_j a_j * exp(-b_j * (x - c_j)^2)
    
    def param_to_heights_gauss(params):
        """Convert parameters to a height vector using Gaussian mixture."""
        x = np.linspace(-0.25, 0.25, N)
        # 3 Gaussians: a1,b1,c1, a2,b2,c2, a3,b3,c3
        h = np.zeros(N)
        for j in range(3):
            a, b, c = params[3*j:3*j+3]
            h += a * np.exp(-b * (x - c)**2)
        return np.maximum(h, 0.0)
    
    def objective_gauss(params):
        h = param_to_heights_gauss(params)
        return compute_score(h)
    
    bounds_gauss = [(0.1, 3.0), (1.0, 100.0), (-0.25, 0.25)] * 3
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x0 = [1.0, 20.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # single central Gaussian
        else:
            x0 = []
            for j in range(3):
                x0.append(rng.uniform(0.1, 3.0))
                x0.append(rng.uniform(1.0, 100.0))
                x0.append(rng.uniform(-0.25, 0.25))
        
        try:
            res = minimize(objective_gauss, x0, bounds=bounds_gauss, method='L-BFGS-B',
                          options={'maxiter': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_heights = param_to_heights_gauss(res.x)
        except:
            pass
    
    # Try a flat-top window: f(x) = 1 for |x| < w, 0 otherwise, with w as parameter
    # Or a raised cosine
    
    def param_to_heights_raised_cos(params):
        """Raised cosine window: f(x) = 0.5*(1+cos(pi*x/w)) for |x|<w, 0 otherwise."""
        x = np.linspace(-0.25, 0.25, N)
        w = params[0]
        h = np.zeros(N)
        mask = np.abs(x) < w
        h[mask] = 0.5 * (1 + np.cos(np.pi * x[mask] / w))
        return h
    
    def objective_raised_cos(params):
        h = param_to_heights_raised_cos(params)
        return compute_score(h)
    
    for trial in range(5):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x0 = [0.25]
        else:
            x0 = [rng.uniform(0.1, 0.25)]
        
        try:
            res = minimize(objective_raised_cos, x0, bounds=[(0.05, 0.25)], method='L-BFGS-B',
                          options={'maxiter': 200})
            if res.fun < best_score:
                best_score = res.fun
                best_heights = param_to_heights_raised_cos(res.x)
        except:
            pass
    
    return best_heights.tolist()
