import numpy as np

N = 400

def construct():
    # Based on the directive: Model as sparse combinatorial design.
    # Select a small subset of intervals and assign random signs.
    # This decouples from continuous space and focuses on structure.
    
    # Choose a sparsity level M.
    # The directive suggests "small subset".
    # Let's try M = 20.
    M = 20
    
    # We will evaluate a few random sparse configurations and pick the best one
    # based on a proxy metric: L2 norm of the autoconvolution.
    # This is often related to "energy" bounds.
    
    best_x = None
    best_score = float('inf')
    
    # Try a few random realizations
    for _ in range(50):
        # Select M random indices
        indices = np.random.choice(N, M, replace=False)
        
        # Assign random signs
        signs = np.random.choice([-1, 1], M)
        
        # Construct vector
        x = np.zeros(N)
        x[indices] = signs / np.sqrt(M)
        
        # Calculate autoconvolution
        # Note: np.convolve does full convolution. 
        # For autoconvolution of length N, we care about the part that fits in N.
        # "Full" length is 2N-1.
        # The "peak" or norm might depend on the definition.
        # Standard autoconvolution in signal processing often refers to the circular one or the linear one truncated.
        # Let's use linear autoconvolution.
        conv = np.convolve(x, x, mode='full')
        
        # Proxy metric: L2 norm of the autoconvolution
        # Or maybe the max absolute value excluding the center?
        # Let's use L2 norm as a robust "size" metric.
        score_proxy = np.linalg.norm(conv)
        
        if score_proxy < best_score:
            best_score = score_proxy
            best_x = x
            
    return best_x
