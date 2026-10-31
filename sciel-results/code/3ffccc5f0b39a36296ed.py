import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 600
DEADLINE = 140

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    def compute_autoconv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv
    
    def soft_max(x, tau=0.01):
        """Soft-max approximation using log-sum-exp."""
        m = np.max(x)
        return m + tau * np.log(np.sum(np.exp((x - m) / tau)))
    
    def gradient_of_soft_max_conv(h, tau=0.01):
        """Compute gradient of soft-max of autoconvolution w.r.t. h."""
        conv = compute_autoconv(h)
        # Soft-max weight for each element
        m = np.max(conv)
        weights = np.exp((conv - m) / tau)
        weights /= np.sum(weights)
        
        # Gradient of conv[k] w.r.t. h[i] is h[k-i] + h[i-k] (for valid indices)
        # But we need to be careful with boundaries
        # conv[k] = sum_i h[i] * h[i-k] for k in range(-N+1, N)
        # In our padded representation, conv is of length 2N
        # h_pad has h in first N positions
        # conv[k] = sum_{i} h_pad[i] * h_pad[i-k]
        
        # For the soft-max objective: d/dh[i] sum_k w_k * conv[k]
        # = sum_k w_k * (h_pad[k-i] + h_pad[i-k])
        # where we only consider valid indices
        
        # Efficient computation using FFT
        # The gradient is essentially the convolution of weights with h
        # Let's compute it properly
        
        # For each i, grad[i] = sum_k w_k * (h[k-i] + h[i-k])
        # This is like a cross-correlation
        
        # Let's use direct computation for clarity, then optimize if needed
        grad = np.zeros(N)
        
        # conv is indexed from 0 to 2N-1
        # conv[k] corresponds to lag k-N+1 (from -(N-1) to N-1)
        # Actually, with the padding, conv[k] = sum_i h_pad[i] * h_pad[i-k]
        # where h_pad has h in [0, N-1]
        
        # For a given i in [0, N-1], d conv[k] / d h[i] = h_pad[i-k] + h_pad[k-i]
        # (since conv[k] = sum_j h_pad[j] h_pad[j-k], derivative w.r.t. h_pad[i] 
        #  is h_pad[i-k] when j=i, plus h_pad[k-i] when j-k=i i.e. j=k+i)
        
        # So grad[i] = sum_k w_k * (h_pad[i-k] + h_pad[k-i])
        # Note: h_pad[i-k] is non-zero only when 0 <= i-k < N
        #        h_pad[k-i] is non-zero only when 0 <= k-i < N
        
        # This is equivalent to: grad = conv(w, h) + conv(w, h_flipped)
        # where w is the weight vector of length 2N
        
        # Let's compute using FFT
        w_full = weights  # length 2N
        h_full = np.zeros(2 * N)
        h_full[:N] = h
        
        # grad[i] = sum_k w[k] * h[i-k] + sum_k w[k] * h[k-i]
        # First term: (w * h)[i] where * is convolution
        # Second term: (w * h_flipped)[i]
        
        # Actually, let's just do it directly with numpy for N=600
        # It's O(N^2) but N is small enough
        
        for i in range(N):
            # h[i-k] for k where 0 <= i-k < N, i.e., i-N+1 <= k <= i
            # h[k-i] for k where 0 <= k-i < N, i.e., i <= k <= i+N-1
            
            # First sum: sum_k w[k] * h[i-k] for valid k
            # i-k in [0, N-1] => k in [i-N+1, i]
            k_min = max(0, i - N + 1)
            k_max = min(2*N - 1, i)
            if k_min <= k_max:
                idxs = np.arange(k_min, k_max + 1)
                h_idx = i - idxs  # these are in [0, N-1]
                grad[i] += np.sum(weights[k_min:k_max+1] * h[h_idx])
            
            # Second sum: sum_k w[k] * h[k-i] for valid k
            # k-i in [0, N-1] => k in [i, i+N-1]
            k_min2 = max(0, i)
            k_max2 = min(2*N - 1, i + N - 1)
            if k_min2 <= k_max2:
                idxs = np.arange(k_min2, k_max2 + 1)
                h_idx = idxs - i  # these are in [0, N-1]
                grad[i] += np.sum(weights[k_min2:k_max2+1] * h[h_idx])
        
        return grad
    
    # Gaussian basis representation
    # h(x) = sum_{j=1}^{K} w_j * exp(-((x - c_j)^2) / (2 * s_j^2))
    # where x is in [0, N-1] (discrete grid)
    # Parameters: w_j > 0, c_j in [0, N-1], s_j > 0
    # Total parameters: 3*K
    
    K = 10  # Number of Gaussian components
    
    i = np.arange(N, dtype=float)
    
    def params_to_h(params):
        """Convert parameter vector to height array."""
        h = np.zeros(N)
        for j in range(K):
            w_j = params[3*j]
            c_j = params[3*j+1]
            s_j = params[3*j+2]
            if s_j < 1e-6:
                s_j = 1e-6
            h += w_j * np.exp(-((i - c_j)**2) / (2 * s_j**2))
        return h
    
    def objective(params):
        """Objective function: max autoconvolution of normalized h."""
        if time_left() < 1:
            return 1e10
        h = params_to_h(params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        return compute_autoconv_max(h_norm)
    
    def objective_soft(params, tau=0.01):
        """Soft-max objective for smoother optimization."""
        if time_left() < 1:
            return 1e10
        h = params_to_h(params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        conv = compute_autoconv(h_norm)
        return soft_max(conv, tau)
    
    def gradient_soft(params, tau=0.01):
        """Gradient of soft-max objective."""
        h = params_to_h(params)
        s = np.sum(h)
        if s < 1e-10:
            return np.zeros(3*K)
        h_norm = h / s
        
        # Gradient of soft-max of conv w.r.t. h_norm
        grad_h_norm = gradient_of_soft_max_conv(h_norm, tau)
        
        # Chain rule: d/dparams = d/dh_norm * d h_norm / dparams
        # h_norm = h / s, so d h_norm / d h = (1/s) * I - (1/s^2) * 1 * 1^T
        # But we're working with params, so we need d h / dparams
        
        # For simplicity, let's use finite differences for the parametric gradient
        # Or we can compute it analytically
        
        # Let's use the fact that h = sum_j w_j * g_j
        # dh/dw_j = g_j
        # dh/dc_j = w_j * dg_j/dc_j
        # dh/ds_j = w_j * dg_j/ds_j
        
        grad_params = np.zeros(3*K)
        
        for j in range(K):
            w_j = params[3*j]
            c_j = params[3*j+1]
            s_j = params[3*j+2]
            if s_j < 1e-6:
                s_j = 1e-6
            
            g_j = np.exp(-((i - c_j)**2) / (2 * s_j**2))
            dg_j_dc_j = w_j * g_j * (i - c_j) / s_j**2
            dg_j_ds_j = w_j * g_j * (i - c_j)**2 / s_j**3
            
            # d h_norm / d w_j = (g_j * s - h * g_j.sum()) / s^2
            # But this is complex. Let's just use finite differences.
            pass
        
        # Use finite differences for reliability
        eps = 1e-7
        grad_params = np.zeros(3*K)
        base_val = objective_soft(params, tau)
        
        for k in range(3*K):
            params_plus = params.copy()
            params_plus[k] += eps
            val_plus = objective_soft(params_plus, tau)
            grad_params[k] = (val_plus - base_val) / eps
        
        return grad_params
    
    # Define bounds for parameters
    # w_j in [0.01, 10], c_j in [0, N-1], s_j in [1, N/2]
    bounds = []
    for j in range(K):
        bounds.append((0.01, 10.0))   # w_j
        bounds.append((0.0, N-1))     # c_j
        bounds.append((1.0, N/2))     # s_j
    
    # Generate diverse initial populations
    def generate_initial_pop(size):
        """Generate diverse initial parameter vectors."""
        pop = np.zeros((size, 3*K))
        for idx in range(size):
            # Random strategy
            strategy = random.randint(0, 3)
            
            if strategy == 0:
                # Bimodal: two large Gaussians at edges
                for j in range(K):
                    if j < K//2:
                        pop[idx, 3*j] = random.uniform(0.5, 2.0)
                        pop[idx, 3*j+1] = random.uniform(0, N*0.2)
                        pop[idx, 3*j+2] = random.uniform(N/10, N/4)
                    else:
                        pop[idx, 3*j] = random.uniform(0.5, 2.0)
                        pop[idx, 3*j+1] = random.uniform(N*0.8, N-1)
                        pop[idx, 3*j+2] = random.uniform(N/10, N/4)
            elif strategy == 1:
                # Spread out
                for j in range(K):
                    pop[idx, 3*j] = random.uniform(0.1, 1.0)
                    pop[idx, 3*j+1] = j * (N-1) / (K-1) + random.uniform(-N/10, N/10)
                    pop[idx, 3*j+2] = random.uniform(N/8, N/3)
            elif strategy == 2:
                # Central concentration
                for j in range(K):
                    pop[idx, 3*j] = random.uniform(0.1, 0.5)
                    pop[idx, 3*j+1] = random.uniform(N*0.3, N*0.7)
                    pop[idx, 3*j+2] = random.uniform(N/6, N/2)
            else:
                # Fully random
                for j in range(K):
                    pop[idx, 3*j] = random.uniform(0.01, 5.0)
                    pop[idx, 3*j+1] = random.uniform(0, N-1)
                    pop[idx, 3*j+2] = random.uniform(2, N/2)
        return pop
    
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)
    
    # Stage 1: Differential Evolution
    if time_left() > 60:
        max_iter = int(min(100, time_left() * 0.5))
        pop_size = min(50, max(20, int(time_left() * 0.3)))
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if result.success or result.fun < best_val:
                h_de = params_to_h(result.x)
                s = np.sum(h_de)
                if s > 1e-10:
                    h_de = h_de / s
                    val_de = compute_autoconv_max(h_de)
                    if val_de < best_val:
                        best_val = val_de
                        best_h = h_de.copy()
        except Exception:
            pass
    
    # Stage 2: Local refinement with L-BFGS-B using soft-max objective
    if time_left() > 30:
        # Create several starting points near the best
        start_points = []
        
        # Point 1: fit Gaussians to best_h
        # Simple approach: use the DE result if available
        if 'result' in dir() and hasattr(result, 'x'):
            start_points.append(result.x.copy())
        
        # Point 2: bimodal guess
        p2 = np.zeros(3*K)
        for j in range(K):
            if j < K//2:
                p2[3*j] = 1.0
                p2[3*j+1] = N*0.15
                p2[3*j+2] = N/8
            else:
                p2[3*j] = 1.0
                p2[3*j+1] = N*0.85
                p2[3*j+2] = N/8
        start_points.append(p2)
        
        # Point 3: uniform-ish
        p3 = np.zeros(3*K)
        for j in range(K):
            p3[3*j] = 0.5
            p3[3*j+1] = j * (N-1) / (K-1) if K > 1 else N/2
            p3[3*j+2] = N/4
        start_points.append(p3)
        
        # Point 4: random
        p4 = np.zeros(3*K)
        for j in range(K):
            p4[3*j] = random.uniform(0.1, 2.0)
            p4[3*j+1] = random.uniform(0, N-1)
            p4[3*j+2] = random.uniform(N/10, N/3)
        start_points.append(p4)
        
        for sp in start_points:
            if time_left() < 15:
                break
            
            try:
                # Use soft-max objective for smoother gradient
                res = minimize(
                    objective_soft,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < best_val:
                    h_res = params_to_h(res.x)
                    s = np.sum(h_res)
                    if s > 1e-10:
                        h_res = h_res / s
                        val_res = compute_autoconv_max(h_res)
                        if val_res < best_val:
                            best_val = val_res
                            best_h = h_res.copy()
            except Exception:
                pass
    
    # Stage 3: Water-filling refinement on the grid
    if time_left() > 25:
        current_h = best_h.copy()
        current_val = best_val
        best_h_wf = best_h.copy()
        best_val_wf = best_val
        
        # Water-filling: iteratively adjust h to equalize the convolution peaks
        # For fixed support, reducing total convolution mass lowers peak value
        
        n_iter = int(min(2000, time_left() * 30))
        step_size = 0.01
        
        for k in range(n_iter):
            if time_left() < 5:
                break
            
            conv = compute_autoconv(current_h)
            max_conv = np.max(conv)
            
            # Find positions where convolution is high and low
            # We want to reduce h at positions that contribute to high conv
            # and increase h at positions that contribute to low conv
            
            # Compute sensitivity: how much does conv[k] depend on h[i]?
            # For simplicity, use a heuristic: positions where h is high and 
            # nearby convolution is high should be reduced
            
            # Simple water-filling: move mass from high-conv regions to low-conv regions
            # Identify top and bottom quartiles of convolution
            conv_sorted = np.sort(conv)
            q75 = conv_sorted[int(0.75 * len(conv_sorted))]
            q25 = conv_sorted[int(0.25 * len(conv_sorted))]
            
            # Positions where convolution is high
            high_mask = conv > q75
            low_mask = conv < q25
            
            # For each position i, compute its contribution to high vs low conv
            # This is complex, so use a simpler approach:
            # Perturb h in the direction that reduces max conv
            
            # Gradient-based water-filling
            grad = gradient_of_soft_max_conv(current_h, tau=0.01)
            
            # Update: h_new = h - step * grad, then project to simplex
            h_new = current_h - step_size * grad
            h_new = np.maximum(h_new, 0)
            s = np.sum(h_new)
            if s < 1e-10:
                continue
            h_new /= s
            
            new_val = compute_autoconv_max(h_new)
            
            if new_val < current_val:
                current_h = h_new
                current_val = new_val
                if current_val < best_val_wf:
                    best_val_wf = current_val
                    best_h_wf = current_h.copy()
            
            # Decay step size
            step_size *= 0.999
        
        if best_val_wf < best_val:
            best_val = best_val_wf
            best_h = best_h_wf
    
    # Stage 4: Fine coordinate descent
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(10, time_left() // 2))
        
        for round_num in range(max_rounds):
            if time_left() < 5:
                break
            
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break
                
                for delta in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] + delta * current_h[idx])
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s
                    
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
