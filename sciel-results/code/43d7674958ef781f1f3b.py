import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_ratio(h):
    """
    Compute the ratio (L2^2) / (L1 * SupConv).
    Uses FFT for speed.
    """
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    
    # FFT convolution
    size = 2 * len(h)
    H = np.zeros(size)
    H[:len(h)] = h
    H_fft = np.fft.fft(H)
    Conv_fft = H_fft * H_fft
    Conv = np.real(np.fft.ifft(Conv_fft))
    sup_conv = np.max(Conv)
    
    if sup_conv < 1e-15:
        return 0.0
    
    ratio = l2_sq / (s * sup_conv)
    return ratio

def optimize_fourier_phase(h0, max_iter=2000, ftol=1e-10):
    """
    Optimize using Fourier phase parameterization.
    h = |IFFT(FFT(h) * exp(1j*phase))| with fixed L2 norm.
    """
    h = np.maximum(h0, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0, h0
    
    # Normalize to L2 = 1
    h = h / np.sqrt(np.sum(h**2))
    
    # Get initial spectrum
    size = 2 * N
    H = np.zeros(size, dtype=complex)
    H[:N] = h
    H_fft = np.fft.fft(H)
    
    # Initial phase
    phase = np.angle(H_fft)
    
    def objective_and_grad(phase_vec):
        # Reconstruct h from phase
        H_fft_new = np.abs(H_fft) * np.exp(1j * phase_vec)
        h_new = np.real(np.fft.ifft(H_fft_new))[:N]
        h_new = np.maximum(h_new, 0)
        
        s = np.sum(h_new)
        if s < 1e-15:
            return 0.0, np.zeros_like(phase_vec)
        
        l2_sq = np.sum(h_new**2)
        
        # FFT convolution
        H_new = np.zeros(size, dtype=complex)
        H_new[:N] = h_new
        H_new_fft = np.fft.fft(H_new)
        Conv_fft = H_new_fft * H_new_fft
        Conv = np.real(np.fft.ifft(Conv_fft))
        sup_conv = np.max(Conv)
        
        if sup_conv < 1e-15:
            return 0.0, np.zeros_like(phase_vec)
        
        ratio = l2_sq / (s * sup_conv)
        
        # Numerical gradient (central difference)
        grad = np.zeros_like(phase_vec)
        eps = 1e-6
        for i in range(len(phase_vec)):
            phase_plus = phase_vec.copy()
            phase_plus[i] += eps
            H_plus = np.abs(H_fft) * np.exp(1j * phase_plus)
            h_plus = np.maximum(np.real(np.fft.ifft(H_plus))[:N], 0)
            s_plus = np.sum(h_plus)
            if s_plus < 1e-15:
                r_plus = 0.0
            else:
                l2_plus = np.sum(h_plus**2)
                H_plus_fft = np.fft.fft(np.concatenate([h_plus, np.zeros(N)]))
                Conv_plus = np.real(np.fft.ifft(H_plus_fft * H_plus_fft))
                sup_plus = np.max(Conv_plus)
                r_plus = l2_plus / (s_plus * sup_plus) if sup_plus > 1e-15 else 0.0
            
            phase_minus = phase_vec.copy()
            phase_minus[i] -= eps
            H_minus = np.abs(H_fft) * np.exp(1j * phase_minus)
            h_minus = np.maximum(np.real(np.fft.ifft(H_minus))[:N], 0)
            s_minus = np.sum(h_minus)
            if s_minus < 1e-15:
                r_minus = 0.0
            else:
                l2_minus = np.sum(h_minus**2)
                H_minus_fft = np.fft.fft(np.concatenate([h_minus, np.zeros(N)]))
                Conv_minus = np.real(np.fft.ifft(H_minus_fft * H_minus_fft))
                sup_minus = np.max(Conv_minus)
                r_minus = l2_minus / (s_minus * sup_minus) if sup_minus > 1e-15 else 0.0
            
            grad[i] = -(r_plus - r_minus) / (2 * eps)
        
        return -ratio, grad
    
    # Optimize phase
    try:
        result = opt.minimize(objective_and_grad, phase, method='L-BFGS-B',
                              jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': 1e-10})
        best_phase = result.x
    except Exception:
        best_phase = phase
    
    # Reconstruct final h
    H_final = np.abs(H_fft) * np.exp(1j * best_phase)
    h_final = np.real(np.fft.ifft(H_final))[:N]
    h_final = np.maximum(h_final, 0)
    
    s = np.sum(h_final)
    if s > 0:
        h_final = h_final / s
    else:
        h_final = np.ones(N) / N
    
    r = compute_ratio(h_final)
    return r, h_final

def optimize_sparse(h0, indices, max_iter=1000):
    """
    Optimize the values at 'indices'.
    """
    if len(indices) == 0:
        return 0.0, h0
    
    x0 = h0[indices].copy()
    x0 = np.maximum(x0, 1e-10)
    
    def objective(x):
        h = np.zeros(N)
        h[indices] = x
        return -compute_ratio(h)
    
    bounds = [(0, 1)] * len(indices)
    
    try:
        result = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
        best_x = result.x
    except Exception:
        best_x = x0
    
    h_opt = np.zeros(N)
    h_opt[indices] = best_x
    
    s = np.sum(h_opt)
    if s > 0:
        h_opt = h_opt / s
    else:
        h_opt = np.ones(N) / N
        
    r = compute_ratio(h_opt)
    return r, h_opt

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_ratio = 0.0
    best_h = np.ones(N) / N
    
    # Initial best
    best_ratio = compute_ratio(best_h)
    
    # Try different sparsity levels
    num_trials_per_k = 50
    
    for K in range(2, 26):
        if time.time() > deadline:
            break
            
        for trial in range(num_trials_per_k):
            if time.time() > deadline:
                break
            
            # Randomly select K indices
            indices = np.random.choice(N, size=K, replace=False)
            indices.sort()
            
            # Random initial values
            h0 = np.zeros(N)
            if trial % 3 == 0:
                h0[indices] = np.random.rand(K)
            elif trial % 3 == 1:
                r = np.random.uniform(0.5, 1.0)
                vals = r ** np.arange(K)
                h0[indices] = vals
            else:
                x = np.linspace(0, 1, K)
                center = np.random.uniform(0.2, 0.8)
                width = np.random.uniform(0.1, 0.5)
                h0[indices] = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            
            # First optimize with sparse method
            r, h_opt = optimize_sparse(h0, indices, max_iter=300)
            
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
                
            # Then try Fourier phase optimization if time permits
            if time.time() < deadline - 10:
                r_fourier, h_fourier = optimize_fourier_phase(h_opt, max_iter=500)
                
                if r_fourier > best_ratio:
                    best_ratio = r_fourier
                    best_h = h_fourier.copy()
                    
    # Final polish on best solution
    if time.time() < deadline - 5:
        r_polish, h_polish = optimize_sparse(best_h, np.where(best_h > 1e-6)[0], max_iter=1000)
        if r_polish > best_ratio:
            best_ratio = r_polish
            best_h = h_polish.copy()
            
    return list(best_h)
