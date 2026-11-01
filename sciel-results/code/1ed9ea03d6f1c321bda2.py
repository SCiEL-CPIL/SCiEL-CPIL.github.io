def objective(x):
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.abs(conv)
    # Mask the center
    center_idx = N - 1
    mask = np.ones(len(conv), dtype=bool)
    mask[center_idx] = False
    sidelobes = abs_conv[mask]
    
    # Soft max of sidelobes
    T = 5.0
    max_sidelobe = np.max(sidelobes)
    # Handle empty or small arrays
    if len(sidelobes) == 0: return 0
    log_sum_exp = np.log(np.sum(np.exp((sidelobes - max_sidelobe) / T))) + max_sidelobe / T
    soft_max_sidelobe = T * log_sum_exp
    
    sum_abs_h = np.sum(np.abs(h))
    
    # Avoid division by zero
    if sum_abs_h < 1e-6:
        return 1e6
    
    value = soft_max_sidelobe / (sum_abs_h**2)
    return value
