"""
Strategy: Complex-domain gradient descent with log-sum-exp surrogate.
We represent the step function heights as h = |z|^2 for a complex vector z.
This automatically enforces h >= 0.
We minimize the normalized max autoconvolution using a smooth log-sum-exp
approximation of the max, which allows for efficient gradient computation
and unconstrained optimization in the complex domain.
"""
import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    dx = 0.5 / N
    
    # Precompute FFT setup
    n_fft = 2 * N
    fft_indices = np.arange(n_fft)
    
    def compute_smooth_max_obj(z):
        """
        Compute the smooth objective: log-sum-exp of normalized autoconvolution.
        z: complex array of length N
        Returns: (objective_value, gradient)
        """
        # h = |z|^2
        h = np.abs(z)**2
        
        # Ensure numerical stability: if h is all zeros, return large value
        s = np.sum(h)
        if s < 1e-15:
            return 1e15, np.zeros_like(z)
        
        norm = (s * dx)**2
        if norm < 1e-15:
            return 1e15, np.zeros_like(z)
        
        # Linear autoconvolution
        z_pad = np.zeros(n_fft, dtype=complex)
        z_pad[:N] = z
        Z = np.fft.fft(z_pad)
        # Power spectrum
        P = np.abs(Z)**2
        
        # Autoconvolution c_k = sum h_i h_{i+k}
        # c = ifft(P)
        c = np.fft.ifft(P).real
        
        # The linear autoconvolution is in c[0:N]
        c_lin = c[:N]
        
        # Normalize
        c_norm = c_lin / norm
        
        # Smooth max via log-sum-exp
        # Use a temperature parameter beta to control smoothness
        # As beta -> inf, log-sum-exp(beta * x) / beta -> max(x)
        beta = 100.0
        
        # Shift for numerical stability
        max_c = np.max(c_norm)
        shifted = beta * (c_norm - max_c)
        
        # Clip to prevent overflow in exp
        shifted_clipped = np.clip(shifted, -700, 700)
        
        lse = np.log(np.sum(np.exp(shifted_clipped))) / beta + max_c
        
        # Gradient computation
        # Let f(z) = lse(beta * (c(z)/norm - max_c))
        # This is complex. We use Wirtinger calculus.
        # d/dz |z|^2 = conj(z)
        # c_k = sum_i h_i h_{i+k} = sum_i |z_i|^2 |z_{i+k}|^2
        # This is a 4th order polynomial in z and conj(z).
        
        # Alternative: Use automatic differentiation or finite differences?
        # No, let's derive it.
        # Objective J = log(sum_j exp(beta * c_j / norm))
        # dJ/dc_j = (beta/norm) * exp(beta * c_j / norm) / sum_k exp(beta * c_k / norm)
        # Let w_j = dJ/dc_j
        # dJ/dh_i = sum_j w_j * dc_j/dh_i
        # dc_j/dh_i = h_{i-j} + h_{i+j} (for valid indices)
        # dh_i/dz_i = conj(z_i)
        
        # 1. Compute weights w_j
        exp_terms = np.exp(shifted_clipped)
        sum_exp = np.sum(exp_terms)
        if sum_exp < 1e-300:
            return lse, np.zeros_like(z)
        
        w = (beta / norm) * (exp_terms / sum_exp)
        
        # 2. Compute dc_j/dh_i
        # c_j = sum_k h_k h_{k+j}
        # dc_j/dh_i = h_{i-j} (if i-j in [0,N-1]) + h_{i+j} (if i+j in [0,N-1])
        
        # We need dJ/dh = sum_j w_j * (h_{j} shifted)
        # Let g_i = sum_j w_j * (h_{i-j} + h_{i+j})
        # This is a convolution of w with h, plus a cross-correlation-like term.
        
        # Actually, c is the autoconvolution of h.
        # c_j = (h * h)_j
        # dc_j/dh_i is the derivative of the convolution.
        # Let's use the property that d/dh (h*h) is a linear operator.
        # For a fixed j, dc_j/dh_i = h_{i-j} + h_{i+j}
        
        # Sum over j:
        # g_i = sum_j w_j h_{i-j} + sum_j w_j h_{i+j}
        
        # Term 1: sum_j w_j h_{i-j} is the convolution (w * h)_i
        # Term 2: sum_j w_j h_{i+j} is the cross-correlation (w * h_rev)_i ?
        # Let's check indices.
        # (w * h)_i = sum_j w_j h_{i-j}
        # (w * h)_i with w reversed?
        # Let w_rev[j] = w[-j].
        # sum_j w_j h_{i+j} = sum_k w_{i-k} h_k ? No.
        
        # Let's just compute it directly using FFT for speed, or loops if N is small enough.
        # N=600 is small. But we do many iterations. FFT is better.
        
        # g_1[i] = sum_j w_j h_{i-j}
        # This is linear convolution of w and h.
        # g_2[i] = sum_j w_j h_{i+j}
        # Let k = i+j => j = k-i.
        # sum_k w_{k-i} h_k
        # This is the cross-correlation of w and h, or convolution with reversed w.
        # Let w_rev[k] = w[-k] (circularly).
        # Then sum_k w_{k-i} h_k = (w_rev * h)_i ?
        
        # Let's use FFT to compute these sums efficiently.
        # We need to handle boundaries carefully.
        
        # Simpler approach for gradient:
        # Use the fact that we have c.
        # We can compute the gradient of the log-sum-exp w.r.t. c, which is w.
        # Then we need the Jacobian of c w.r.t. h.
        # J_{j,i} = dc_j/dh_i = h_{i-j} + h_{i+j}
        
        # g = J^T w
        # g_i = sum_j w_j J_{j,i} = sum_j w_j (h_{i-j} + h_{i+j})
        
        # Let's compute g_1 = conv(w, h) and g_2 = cross_corr(w, h)
        
        # Pad w and h for linear convolution
        w_pad = np.zeros(n_fft)
        w_pad[:N] = w
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        
        W = np.fft.fft(w_pad)
        H = np.fft.fft(h_pad)
        
        # g_1: linear convolution w * h
        conv_wh = np.fft.ifft(W * H).real[:N]
        
        # g_2: sum_j w_j h_{i+j}
        # This is equivalent to the cross-correlation of w and h, but with specific indexing.
        # Let's verify:
        # (w * h)_i = sum_j w_j h_{i-j}
        # We want sum_j w_j h_{i+j}.
        # Let k = -j. sum_k w_{-k} h_{i-k}.
        # This is the convolution of w_rev and h, where w_rev[k] = w_{-k}.
        # In circular convolution context:
        # w_circ_rev[k] = w[(-k) % N]
        
        w_rev = np.zeros(n_fft)
        # w_rev[0] = w[0]
        # w_rev[k] = w[N-k] for k=1..N-1
        w_rev[1:N] = w[N-1:0:-1]
        
        W_rev = np.fft.fft(w_rev)
        
        # g_2 = ifft(W_rev * H).real[:N]
        cross_wh = np.fft.ifft(W_rev * H).real[:N]
        
        g = conv_wh + cross_wh
        
        # Now, dJ/dz_i = g_i * dh_i/dz_i = g_i * conj(z_i)
        grad_z = g * np.conj(z)
        
        return lse, grad_z

    # Initialization
    # Start with a Gaussian-like shape in complex domain
    n = np.arange(N)
    center = (N - 1) / 2.0
    sigma = N / 4.0
    
    # h0 = exp(-0.5 * ((n-center)/sigma)**2)
    h0 = np.exp(-0.5 * ((n - center) / sigma)**2)
    
    # z0 = sqrt(h0) * exp(i * theta)
    # Random phases
    theta = np.random.uniform(0, 2 * np.pi, N)
    z0 = np.sqrt(h0) * np.exp(1j * theta)
    
    best_z = z0.copy()
    best_obj, _ = compute_smooth_max_obj(best_z)
    
    # Gradient descent with momentum
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(z0)
    
    # Adaptive learning rate / Line search
    # Since the objective is smooth, we can use standard GD.
    # However, the scale of z matters.
    # We normalize h in the objective, so the scale of z doesn't affect the value,
    # but it affects the gradient magnitude.
    # Let's normalize z to have unit L2 norm at each step?
    # No, h = |z|^2. If we scale z by c, h scales by c^2.
    # The objective is invariant to scaling of h.
    # So we should constrain ||z||_2 or ||h||_1 to avoid drifting to 0 or inf.
    # Let's enforce sum(h) = 1 (i.e., sum(|z|^2) = 1) by projecting z.
    
    def project_z(z):
        h = np.abs(z)**2
        s = np.sum(h)
        if s < 1e-15:
            return z
        # We want sum(h_new) = 1 => sum(|z_new|^2) = 1
        # z_new = z / sqrt(s)
        return z / np.sqrt(s)
    
    z = project_z(z0)
    v = np.zeros_like(z)
    
    # Optimization loop
    max_iters = 2000
    patience = 200
    no_improve_count = 0
    
    for i in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        obj, grad = compute_smooth_max_obj(z)
        
        if np.isnan(obj) or np.isinf(obj):
            break
            
        if obj < best_obj:
            best_obj = obj
            best_z = z.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > patience:
            # Try a different initialization or restart
            # Simple restart with random phases
            z = project_z(np.sqrt(h0) * np.exp(1j * np.random.uniform(0, 2 * np.pi, N)))
            v = np.zeros_like(z)
            lr = lr * 0.5
            no_improve_count = 0
            if lr < 1e-6:
                break
                
        # Update
        # Use Adam-like optimizer for better convergence
        # For simplicity, let's stick to Momentum GD first.
        
        # Gradient step
        v = momentum * v + grad
        z = z - lr * v
        
        # Project back to unit norm
        z = project_z(z)
        
        # Adaptive learning rate
        # If objective increases significantly, reduce lr
        # (Simple heuristic)
        
    # Final refinement with L-BFGS-B on the real heights h
    # Convert best_z to h
    h_best = np.abs(best_z)**2
    s = np.sum(h_best)
    h_best = h_best / s  # Normalize sum to 1 for L-BFGS-B
    
    # Define objective for L-BFGS-B
    def compute_raw_max(h):
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e15
        norm = (s * dx)**2
        if norm < 1e-15:
            return 1e15
        
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        P = np.abs(H)**2
        c = np.fft.ifft(P).real[:N]
        
        max_c = np.max(c)
        return max_c / norm

    def objective_lbfgs(x):
        return compute_raw_max(x)
    
    try:
        from scipy.optimize import minimize
        # Use the h from complex optimization as initial guess
        x0 = h_best.copy()
        
        # We can try a few different initializations for L-BFGS-B
        # 1. From complex opt
        # 2. Uniform
        # 3. Hann
        
        candidates = [x0, np.ones(N)/N, 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / (N - 1))) / np.sum(0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / (N - 1))))]
        
        best_h = x0
        best_val = compute_raw_max(x0)
        
        for c in candidates:
            if time.time() > deadline - 5:
                break
            c = np.maximum(c, 0)
            c = c / np.sum(c)
            try:
                result = minimize(objective_lbfgs, c, method='L-BFGS-B', 
                                  bounds=[(0, None)]*N,
                                  options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_h = result.x
            except Exception:
                pass
        best_h = np.maximum(best_h, 0)
        return list(best_h)
        
    except Exception:
        best_h = np.maximum(h_best, 0)
        return list(best_h)

# EVOLVE-BLOCK-END
