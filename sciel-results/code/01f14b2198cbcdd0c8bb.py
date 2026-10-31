import numpy as np
import time
import scipy.optimize as opt

def construct():
    N = 400
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin

    # Helper to compute the smooth objective
    # We minimize max_{k != 0} |A_k| subject to sum(x) = 1
    # Using log-sum-exp for smoothness
    def get_smooth_score(x, tau=0.05):
        # Normalize x to sum 1 for the optimization landscape
        s = np.sum(x)
        if np.isclose(s, 0):
            return 1e10
        x_norm = x / s
        
        # Compute autoconvolution
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x_norm, n=n_fft)
        A = np.fft.irfft(X * np.conj(X), n=n_fft)
        
        # Extract linear convolution part
        # Linear conv of length N is in A[0:N-1] and A[n_fft-(N-2):]
        # Actually, np.fft.irfft with n=n_fft gives circular conv.
        # For linear conv of two length-N sequences, we need length 2N-1.
        # The peak is at index N-1 in the linear conv.
        # In the FFT output of size n_fft, the linear conv components are:
        # A[0:N-1] corresponds to lags -(N-2) to -1 ? No.
        # Let's use np.convolve for correctness in the score function, N=400 is small enough
        # for ~1000 iterations if we are careful, but FFT is faster.
        # Let's stick to FFT but extract correctly.
        # Linear conv y[k] = sum x[i] x[k-i].
        # FFT method: ifft(fft(x, L) * fft(x, L)) gives circular conv.
        # If L >= 2N-1, the first 2N-1 samples are the linear conv.
        # The peak (k=0 in correlation sense, or k=N-1 in conv sense) is at index N-1.
        
        # A_lin = A[:2*N-1]
        # Peak is at index N-1.
        A_lin = A[:2*N-1]
        
        # Mask out the peak at N-1
        A_masked = A_lin.copy()
        A_masked[N-1] = 0.0
        
        # Compute smooth max
        # We want max |A_masked|
        # log-sum-exp of |A_masked|/tau * tau
        abs_A = np.abs(A_masked)
        max_val = np.max(abs_A)
        if max_val == 0:
            return 0.0
        
        # Log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        shifted = abs_A / tau - max_val / tau
        # Clip to avoid overflow in exp, though shifted <= 0
        sum_exp = np.sum(np.exp(shifted))
        smooth_max = max_val + tau * np.log(sum_exp)
        
        return smooth_max

    # Helper to compute the true score for final evaluation
    def get_true_score(x):
        s = np.sum(x)
        if np.isclose(s, 0):
            return np.inf
        x_norm = x / s
        A_lin = np.convolve(x_norm, x_norm, mode='full')
        A_masked = A_lin.copy()
        A_masked[N-1] = 0.0
        max_side = np.max(np.abs(A_masked))
        return max_side # Since sum is 1, ratio is just max_side

    # Initializations
    n = np.arange(N)
    center = (N - 1) / 2.0
    
    candidates = []
    
    # 1. Quadratic Phase Chirps (Real Part)
    # x[n] = cos(pi * (n - center)^2 / T)
    # Try various T values, some irrational
    T_values = [
        5.0, 10.0, 20.0, 50.0, 
        np.pi, 2*np.pi, np.e, np.sqrt(2)*10, 
        7.1, 13.3, 23.5
    ]
    for T in T_values:
        x_chirp = np.cos(np.pi * (n - center)**2 / T)
        # Ensure sum is not zero
        if abs(np.sum(x_chirp)) > 1e-6:
            candidates.append(x_chirp)
            
    # 2. Standard Windows (Fallbacks)
    candidates.append(np.hanning(N))
    candidates.append(np.hamming(N))
    candidates.append(np.blackman(N))
    
    # 3. Random Smooth (Low Pass Filtered Noise)
    np.random.seed(42)
    noise = np.random.randn(N)
    kernel = np.ones(10)/10
    smooth_noise = np.convolve(noise, kernel, mode='same')
    candidates.append(smooth_noise)
    
    best_x = np.ones(N)
    best_score = get_true_score(best_x)
    
    # Optimize each candidate
    for idx, x_init in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
            
        # Normalize initial guess to sum 1
        s_init = np.sum(x_init)
        if abs(s_init) < 1e-6:
            continue
        x_start = x_init / s_init
        
        # Define objective for optimization
        # We minimize smooth_max |A_k| (k!=0) + penalty * (sum(x) - 1)^2
        def objective(x):
            s = np.sum(x)
            penalty = 100.0 * (s - 1.0)**2
            return get_smooth_score(x) + penalty
        
        try:
            # Use L-BFGS-B, it handles large N well
            res = opt.minimize(objective, x_start, method='L-BFGS-B', 
                               options={'maxiter': 50, 'disp': False, 'gtol': 1e-5})
            x_opt = res.x
            
            # Check if constraint is roughly satisfied
            if abs(np.sum(x_opt) - 1.0) < 0.1:
                # Evaluate true score
                true_score = get_true_score(x_opt)
                if true_score < best_score:
                    best_score = true_score
                    best_x = x_opt.copy()
                    
        except Exception:
            pass

    # Final Local Polish: Coordinate Descent on the best candidate
    # We do a few passes of single-element perturbation to fine-tune
    # This helps correct for the log-sum-exp approximation error
    current_x = best_x.copy()
    current_score = best_score
    
    # Small random perturbations + coordinate descent hybrid
    step_size = 0.05
    for _ in range(500):
        if time.time() > deadline - 2.0:
            break
        
        # Try perturbing a random subset
        num_perturb = 5
        idxs = np.random.choice(N, num_perturb, replace=False)
        noise = np.random.normal(0, step_size, num_perturb)
        
        new_x = current_x.copy()
        new_x[idxs] += noise
        
        new_score = get_true_score(new_x)
        
        if new_score < current_score:
            current_x = new_x
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_x = current_x.copy()
        else:
            # If not improved, try a smaller step or different indices
            pass
            
        # Decay step size
        step_size *= 0.95

    return best_x
