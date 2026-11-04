import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = (1.0/2.0) / N  # interval width = 1/1200

def compute_autoconv(h):
    """Compute the autoconvolution of the step function at interval centers."""
    conv = np.convolve(h, h, mode='full') * DELTA
    return conv

def compute_score(h):
    """Compute normalized max autoconvolution: max(f*f) / (int f)^2"""
    conv = compute_autoconv(h)
    max_conv = np.max(conv)
    integral = np.sum(h) * DELTA
    if integral <= 0:
        return 1e10
    return max_conv / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # 1. Structural Change: Chirp-based Initialization
    # The directive suggests representing the step function to leverage low-autocorrelation properties
    # of linear frequency-modulated (chirp) signals. We use a "phase-matched" boxcar structure.
    # Specifically, we create a function that looks like the magnitude of a chirp, or simply
    # a structured boxcar sequence which tends to have lower peak-to-average power ratio in convolution
    # compared to smooth Gaussians.
    
    # We will generate initial guesses that are "bang-bang" or structured, 
    # inspired by the "Boxcar-initialized" strategy from the inspiration which improved scores.
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Generate structured initial guesses
    inits = []
    
    # 1. Pure Uniform (baseline)
    inits.append(np.ones(N))
    
    # 2. Chirp-like structure: Alternating blocks or specific patterns
    # A simple chirp in time domain for a bandpass signal creates a spread out autocorrelation.
    # For a baseband non-negative signal, we can use a "pulse train" with varying widths or positions.
    
    # Strategy: Random Boxcar Sums (as per inspiration)
    # This creates "rough" functions that often have lower peak autoconvolution than smooth ones.
    np.random.seed(42)
    
    def generate_boxcar_heights(num_boxcars, max_width_frac=0.2):
        heights = np.zeros(N)
        # Bin centers
        bin_centers = np.arange(N)
        for _ in range(num_boxcars):
            c = np.random.uniform(0, N)
            w = np.random.uniform(0.02 * N, max_width_frac * N)
            a = np.random.uniform(0.5, 1.5)
            # Wrap around? No, we are on a finite interval [-1/4, 1/4].
            # Just clip to [0, N-1]
            start = int(np.floor(c - w/2))
            end = int(np.ceil(c + w/2))
            start = max(0, start)
            end = min(N, end)
            heights[start:end] += a
        return heights

    # Add several random boxcar inits
    for _ in range(15):
        h_box = generate_boxcar_heights(np.random.randint(2, 6))
        if np.sum(h_box) > 0:
            inits.append(h_box / np.sum(h_box))
            
    # 3. "Chirp" specific: A function whose Fourier transform is a shifted sinc (box in freq)
    # Or, more directly, we optimize a parameterized family that resembles a chirp envelope.
    # f(x) = A * |cos(pi * x * f_c + 0.5 * pi * k * x^2)|
    # This is non-negative and has the "chirp" property.
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2
    
    for fc in [5, 10, 20, 40]:
        for k in [10, 50, 100]:
            h_chirp = np.abs(np.cos(np.pi * x * fc + 0.5 * np.pi * k * x**2))
            h_chirp = h_chirp + 0.1 # Ensure non-zero
            inits.append(h_chirp / np.sum(h_chirp))

    # 4. Gaussian (standard baseline)
    for sigma in [0.05, 0.1, 0.15]:
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        inits.append(h_gauss / np.sum(h_gauss))

    # Optimization Strategy: 
    # Use L-BFGS-B on the full height vector h (size 600) with bounds [0, inf].
    # The objective is non-smooth (max), so we use a smooth surrogate (log-sum-exp) 
    # for the gradient, but evaluate the true score for tracking the best.
    
    def smooth_surrogate(h, beta=50.0):
        """Smooth surrogate for max(f*f)/(int f)^2."""
        h = np.maximum(h, 0)
        conv = np.convolve(h, h, mode='full') * DELTA
        integral = np.sum(h) * DELTA
        if integral <= 1e-15:
            return 1e10
        
        m = np.max(conv)
        if m < 0:
            m = 0
        # Log-sum-exp
        lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
        return lse / (integral * integral)

    def smooth_surrogate_grad(h, beta=50.0):
        """Gradient of the smooth surrogate."""
        h = np.maximum(h, 0)
        conv = np.convolve(h, h, mode='full') * DELTA
        integral = np.sum(h) * DELTA
        if integral <= 1e-15:
            return np.zeros_like(h), 1e10
        
        m = np.max(conv)
        if m < 0:
            m = 0
            
        # Soft-max weights
        w = np.exp((conv - m) * beta)
        w_sum = np.sum(w)
        if w_sum < 1e-300:
            w = np.zeros_like(conv)
            w[np.argmax(conv)] = 1.0
            w_sum = 1.0
        w = w / w_sum
        
        # Gradient of soft-max conv: 
        # d/dh_i [sum_j w_j conv_j] where conv_j = DELTA * sum_k h_k h_{j-k}
        # = DELTA * sum_j w_j (h_{j-i} + h_{i-j+N-1}) with bounds
        # Using correlation as derived in the current program logic:
        corr_wh = np.correlate(w, h, mode='full')
        S1 = corr_wh[N-1::-1][:N]
        
        h_rev = h[::-1]
        corr_wh_rev = np.correlate(w, h_rev, mode='full')
        S2 = corr_wh_rev[N-1::-1][:N]
        
        d_obj_val = DELTA * (S1 + S2)
        
        obj_val = np.sum(w * conv)
        int_val = integral
        
        # Chain rule for ratio (obj_val / int_val^2)
        grad_obj = (1.0 / (int_val * int_val)) * d_obj_val - (2.0 * obj_val / (int_val ** 3)) * DELTA * np.ones(N)
        
        # Mask gradient where h=0? 
        # Since we use bounds [0, inf], L-BFGS-B handles the boundary.
        # However, if h_i is 0, the gradient might push it negative. 
        # The bound handles this, but we can zero out gradient at 0 to be safe/stable.
        grad_obj = grad_obj * (h > 0)
        
        return grad_obj, obj_val / (int_val * int_val)

    bounds = [(0, None)] * N
    
    # Iterate through initial guesses
    for h0 in inits:
        if time.time() > deadline - 20.0:
            break
            
        h0 = np.maximum(h0, 0)
        if np.sum(h0) < 1e-15:
            continue
        h0 = h0 / np.sum(h0)
        
        # Quick check of initial score
        init_score = compute_score(h0)
        if init_score < best_score:
            best_score = init_score
            best_h = h0.copy()
            
        # Optimize
        try:
            # Use a tight tolerance for speed, as we have many restarts
            result = minimize(
                smooth_surrogate,
                h0.copy(),
                jac=smooth_surrogate_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if time.time() > deadline - 10.0:
                break
                
            if result.fun < 1e9:
                h_candidate = np.maximum(result.x, 0)
                # Evaluate true score
                true_score = compute_score(h_candidate)
                if true_score < best_score:
                    best_score = true_score
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass

    # Final Polish: Run a longer optimization from the best solution found
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                smooth_surrogate,
                best_h.copy(),
                jac=smooth_surrogate_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < 1e9:
                h_candidate = np.maximum(result.x, 0)
                true_score = compute_score(h_candidate)
                if true_score < best_score:
                    best_score = true_score
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # 1. Structural Change: Chirp-based Initialization
    # We use a "phase-matched" boxcar structure and chirp envelopes to initialize.
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Generate structured initial guesses
    inits = []
    
    # 1. Pure Uniform (baseline)
    inits.append(np.ones(N))
    
    # 2. Chirp-like structure: Alternating blocks or specific patterns
    
    # Strategy: Random Boxcar Sums
    np.random.seed(42)
    
    def generate_boxcar_heights(num_boxcars, max_width_frac=0.2):
        heights = np.zeros(N)
        bin_centers = np.arange(N)
        for _ in range(num_boxcars):
            c = np.random.uniform(0, N)
            w = np.random.uniform(0.02 * N, max_width_frac * N)
            a = np.random.uniform(0.5, 1.5)
            start = int(np.floor(c - w/2))
            end = int(np.ceil(c + w/2))
            start = max(0, start)
            end = min(N, end)
            heights[start:end] += a
        return heights

    # Add several random boxcar inits
    for _ in range(15):
        h_box = generate_boxcar_heights(np.random.randint(2, 6))
        if np.sum(h_box) > 0:
            inits.append(h_box / np.sum(h_box))
            
    # 3. "Chirp" specific: f(x) = A * |cos(pi * x * f_c + 0.5 * pi * k * x^2)|
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + DELTA/2
    
    for fc in [5, 10, 20, 40]:
        for k in [10, 50, 100]:
            h_chirp = np.abs(np.cos(np.pi * x * fc + 0.5 * np.pi * k * x**2))
            h_chirp = h_chirp + 0.1 # Ensure non-zero
            inits.append(h_chirp / np.sum(h_chirp))

    # 4. Gaussian (standard baseline)
    for sigma in [0.05, 0.1, 0.15]:
        h_gauss = np.exp(-x**2 / (2 * sigma**2))
        inits.append(h_gauss / np.sum(h_gauss))

    # Optimization Strategy: 
    # Use L-BFGS-B on the full height vector h (size 600) with bounds [0, inf].
    
    def smooth_surrogate(h, beta=50.0):
        """Smooth surrogate for max(f*f)/(int f)^2."""
        h = np.maximum(h, 0)
        conv = np.convolve(h, h, mode='full') * DELTA
        integral = np.sum(h) * DELTA
        if integral <= 1e-15:
            return 1e10
        
        m = np.max(conv)
        if m < 0:
            m = 0
        # Log-sum-exp
        lse = m + np.log(np.sum(np.exp((conv - m) * beta)) / beta)
        return lse / (integral * integral)

    def smooth_surrogate_grad(h, beta=50.0):
        """Gradient of the smooth surrogate."""
        h = np.maximum(h, 0)
        conv = np.convolve(h, h, mode='full') * DELTA
        integral = np.sum(h) * DELTA
        if integral <= 1e-15:
            return np.zeros_like(h), 1e10
        
        m = np.max(conv)
        if m < 0:
            m = 0
            
        # Soft-max weights
        w = np.exp((conv - m) * beta)
        w_sum = np.sum(w)
        if w_sum < 1e-300:
            w = np.zeros_like(conv)
            w[np.argmax(conv)] = 1.0
            w_sum = 1.0
        w = w / w_sum
        
        # Gradient of soft-max conv
        corr_wh = np.correlate(w, h, mode='full')
        S1 = corr_wh[N-1::-1][:N]
        
        h_rev = h[::-1]
        corr_wh_rev = np.correlate(w, h_rev, mode='full')
        S2 = corr_wh_rev[N-1::-1][:N]
        
        d_obj_val = DELTA * (S1 + S2)
        
        obj_val = np.sum(w * conv)
        int_val = integral
        
        # Chain rule for ratio (obj_val / int_val^2)
        grad_obj = (1.0 / (int_val * int_val)) * d_obj_val - (2.0 * obj_val / (int_val ** 3)) * DELTA * np.ones(N)
        
        # Mask gradient where h=0
        grad_obj = grad_obj * (h > 0)
        
        return grad_obj, obj_val / (int_val * int_val)

    bounds = [(0, None)] * N
    
    # Iterate through initial guesses
    for h0 in inits:
        if time.time() > deadline - 20.0:
            break
            
        h0 = np.maximum(h0, 0)
        if np.sum(h0) < 1e-15:
            continue
        h0 = h0 / np.sum(h0)
        
        # Quick check of initial score
        init_score = compute_score(h0)
        if init_score < best_score:
            best_score = init_score
            best_h = h0.copy()
            
        # Optimize
        try:
            result = minimize(
                smooth_surrogate,
                h0.copy(),
                jac=smooth_surrogate_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if time.time() > deadline - 10.0:
                break
                
            if result.fun < 1e9:
                h_candidate = np.maximum(result.x, 0)
                # Evaluate true score
                true_score = compute_score(h_candidate)
                if true_score < best_score:
                    best_score = true_score
                    best_h = h_candidate.copy()
                    
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                smooth_surrogate,
                best_h.copy(),
                jac=smooth_surrogate_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if result.fun < 1e9:
                h_candidate = np.maximum(result.x, 0)
                true_score = compute_score(h_candidate)
                if true_score < best_score:
                    best_score = true_score
                    best_h = h_candidate.copy()
        except Exception:
            pass

    return list(best_h)
# EVOLVE-BLOCK-END
