import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sparse signed step function to minimize the autoconvolution peak.
    Uses a random subset of intervals with optimized signs via greedy local search.
    Normalized such that L2 norm is 1.
    """
    M = 50  # Number of active intervals
    
    # 1. Select sparse support
    perm = np.random.permutation(N)
    active_indices = perm[:M]
    
    # 2. Initialize signs
    signs = np.random.choice([-1.0, 1.0], size=M)
    
    # Map index in x to sign index in signs array
    # x is length N, active_indices are the positions.
    # We need a fast way to update autoconvolution.
    
    # Initialize x
    x = np.zeros(N)
    scale = 1.0 / np.sqrt(M)
    x[active_indices] = signs * scale
    
    # Function to compute autoconvolution peak
    # Linear convolution of x with x
    # Since x is sparse, we can compute A_k = sum_i x_i x_{i-k}
    # A has length 2*N - 1, centered at N-1?
    # Standard np.convolve(x, x) returns length 2*N - 1.
    # Indices 0 to 2*N-2.
    # Peak is max(abs(conv))
    
    def get_peak(x):
        conv = np.convolve(x, x, mode='full')
        return np.max(np.abs(conv))
    
    current_peak = get_peak(x)
    
    # Greedy optimization
    # For each active index, try flipping sign
    # We need to map x-index to sign-array-index
    # Create a dictionary or array for this
    sign_map = {idx: i for i, idx in enumerate(active_indices)}
    
    # Precompute contributions?
    # Flipping sign at index i changes x[i] by -2*x[i]
    # This changes conv[k] for all k.
    # conv[k] = sum_j x[j] x[k-j]
    # Change in conv[k] due to x[i] flip:
    # -2*x[i]*x[k-i] - 2*x[k-i]*x[i] = -4*x[i]*x[k-i] ?
    # Wait.
    # conv_new[k] = conv_old[k] - 2*x[i]*x[k-i] - 2*x[k-i]*x[i] + 2*x[i]*x[k-i] ? No.
    # Let x' = x - 2*x[i]*e_i.
    # conv(x') = conv(x - 2*x[i]*e_i, x - 2*x[i]*e_i)
    # = conv(x, x) - 2*x[i]*conv(e_i, x) - 2*x[i]*conv(x, e_i) + 4*x[i]^2*conv(e_i, e_i)
    # conv(e_i, x)[k] = x[k-i]
    # conv(x, e_i)[k] = x[k-i]
    # conv(e_i, e_i)[k] = 1 if k==i else 0
    # So delta_conv[k] = -4*x[i]*x[k-i] + 4*x[i]^2 * (1 if k==i else 0)
    
    # We can update the conv array directly.
    conv = np.convolve(x, x, mode='full')
    current_peak = np.max(np.abs(conv))
    
    # Iterate a few times
    for _ in range(10):
        improved = False
        for i_idx in range(M):
            i = active_indices[i_idx]
            xi = x[i]
            
            # Calculate delta conv
            # delta[k] = -4 * xi * x[k-i] + 4 * xi^2 * (k==i)
            # Note: x is sparse.
            # x[k-i] is non-zero only if k-i is in active_indices.
            
            # We can just compute the new peak for this flip?
            # Or update conv array.
            # Updating conv array is O(N) if we do it naively, but we can do it sparsely.
            # However, N=400 is small. Let's just recompute conv for the flip?
            # np.convolve is fast.
            # x_temp = x.copy()
            # x_temp[i] = -x_temp[i]
            # conv_temp = np.convolve(x_temp, x_temp, mode='full')
            # peak_temp = np.max(np.abs(conv_temp))
            # if peak_temp < current_peak:
            #     x = x_temp
            #     conv = conv_temp
            #     current_peak = peak_temp
            #     improved = True
            
            # This is M * (Cost of convolve).
            # Cost of convolve for N=400 is very low.
            # 10 iterations * 50 flips * 400*400 ops (if naive) or FFT.
            # np.convolve uses direct method for small N? Or FFT?
            # Even if 400*400 = 160k ops.
            # 10 * 50 * 160k = 80M ops. Might be slow in Python if not vectorized well?
            # But np.convolve is C-optimized. It should be fast.
            
            # Let's try the update method to be safe and faster.
            # delta_conv = np.zeros(2*N - 1)
            # This is allocating memory.
            
            # Let's stick to the simple recompute if it's fast enough.
            # 120s limit is generous.
            
            x_temp = x.copy()
            x_temp[i] = -x_temp[i]
            conv_temp = np.convolve(x_temp, x_temp, mode='full')
            peak_temp = np.max(np.abs(conv_temp))
            
            if peak_temp < current_peak - 1e-10:
                x = x_temp
                conv = conv_temp
                current_peak = peak_temp
                improved = True
        
        if not improved:
            break
            
    return x
# EVOLVE-BLOCK-END
