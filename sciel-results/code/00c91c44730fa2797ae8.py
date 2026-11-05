import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Simulated Annealing to optimize the phases and amplitudes of the 
    non-DC frequency components of the perturbation u, where h = 1 + u.
    This minimizes the peak of the linear autoconvolution relative to the sum.
    """
    deadline = time.time() + 125.0  # Safety margin
    
    # Baseline: all ones
    best_h = np.ones(N)
    best_value = 2.0
    
    # Parameterization:
    # h = 1 + u
    # u is zero-mean.
    # U = FFT(u)
    # U[0] = 0
    # U[k] = r[k] * exp(i * phi[k]) for k=1..N/2-1
    # U[N/2] = r[N/2] (real)
    # U[N-k] = conj(U[k])
    
    num_r_mid = N // 2 - 1
    num_phi_mid = N // 2 - 1
    # Total params: num_r_mid + num_phi_mid + 1 (nyq)
    
    # Initialize
    r = np.random.rand(num_r_mid) * 0.2 + 0.05
    phi = np.random.rand(num_phi_mid) * 2 * np.pi
    r_nyq = np.random.rand() * 0.2 - 0.1
    
    def params_to_h(params):
        r_mid = params[:num_r_mid]
        phi_mid = params[num_r_mid : num_r_mid + num_phi_mid]
        r_nyq_val = params[num_r_mid + num_phi_mid]
        
        U = np.zeros(N, dtype=complex)
        U[0] = 0.0
        
        for i in range(num_r_mid):
            k = i + 1
            U[k] = r_mid[i] * np.exp(1j * phi_mid[i])
            U[N-k] = np.conj(U[k])
            
        U[N//2] = r_nyq_val
        
        # h = IFFT(FFT(1) + U) = 1 + IFFT(U)
        # FFT(1) is N at 0.
        # H[0] = N. H[k] = U[k].
        # h = IFFT(H)
        H = np.zeros(N, dtype=complex)
        H[0] = N
        H[1:] = U[1:]
        
        h = np.real(ifft(H))
        return h

    def compute_value(h):
        h_sum = np.sum(h)
        if h_sum == 0 or not np.all(np.isfinite(h)):
            return np.inf
        c_lin = np.convolve(h, h)
        max_conv = np.max(c_lin)
        return abs(2 * N * max_conv / (h_sum ** 2))

    # Initial evaluation
    init_params = np.concatenate([r, phi, [r_nyq]])
    h_init = params_to_h(init_params)
    val_init = compute_value(h_init)
    
    if val_init < best_value:
        best_value = val_init
        best_h = h_init.copy()
        
    current_params = init_params.copy()
    current_h = params_to_h(current_params)
    current_val = compute_value(current_h)
    
    # Simulated Annealing
    T = 0.01
    T_min = 1e-5
    cooling = 0.995
    
    steps = 0
    while time.time() < deadline:
        idx = np.random.randint(0, len(current_params))
        
        if idx < num_r_mid:
            delta = np.random.randn() * 0.01
            new_params = current_params.copy()
            new_params[idx] += delta
        elif idx < num_r_mid + num_phi_mid:
            delta = np.random.randn() * 0.1
            new_params = current_params.copy()
            new_params[idx] += delta
        else:
            delta = np.random.randn() * 0.01
            new_params = current_params.copy()
            new_params[idx] += delta
            
        new_h = params_to_h(new_params)
        new_val = compute_value(new_h)
        
        delta_val = new_val - current_val
        
        if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
            current_params = new_params
            current_h = new_h
            current_val = new_val
            
            if current_val < best_value:
                best_value = current_val
                best_h = current_h.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        steps += 1
        
    return best_h
