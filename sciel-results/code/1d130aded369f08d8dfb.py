import numpy as np
import time

N = 100
HALF = N / 2.0

def _fft_hankel(h):
    """
    Compute the full circular autocorrelation of h via FFT.
    Returns ac of length N where ac[k] = sum_i h[i] * h[(i+k) % N].
    """
    H = np.fft.fft(h)
    power = H * np.conj(H)
    ac = np.fft.ifft(power).real
    return ac

def _overlap_profile(h):
    """
    For each shift k in 0..N-1, the overlap is:
        O_k = sum_i h[i] * (1 - h[(i+k)%N])
            = sum_i h[i] - sum_i h[i] h[(i+k)%N]
            = N/2 - ac[k]
    The maximum overlap is N/2 - min_k ac[k].
    We want to minimize this, i.e. maximize min_k ac[k].
    """
    ac = _fft_hankel(h)
    return HALF - ac  # This is O_k for each k

def _max_overlap(h):
    """Return the maximum overlap value (the quantity to minimize)."""
    ac = _fft_hankel(h)
    return HALF - np.min(ac)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
    Uses a simple iterative approach: clip, then distribute residual.
    """
    h = np.clip(h, 0.0, 1.0)
    for _ in range(50):
        residual = HALF - np.sum(h)
        if abs(residual) < 1e-12:
            break
        # Distribute residual to elements with room
        if residual > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        n_mask = np.sum(mask)
        if n_mask == 0:
            break
        h[mask] += residual / n_mask
        h = np.clip(h, 0.0, 1.0)
    return h

def _gradient_max_overlap(h):
    """
    Compute the subgradient of the maximum overlap with respect to h.
    O_k = HALF - ac[k], where ac[k] = sum_i h[i] h[(i+k)%N].
    dO_k/dh[j] = -sum_i (delta_{i,j} h[(i+k)%N] + delta_{i+k,j} h[i])
               = -h[(j-k)%N] - h[j-k]  ... wait let me redo this.
    
    ac[k] = sum_i h[i] * h[(i+k) % N]
    d(ac[k])/d(h[j]) = sum_i [delta_{i,j} h[(i+k)%N] + delta_{(i+k)%N, j} h[i]]
                      = h[(j+k)%N] + h[(j-k)%N]  (using circular indexing)
    
    So dO_k/dh[j] = -[h[(j+k)%N] + h[(j-k)%N]]
    
    The max overlap is max_k O_k. Its subgradient is the gradient of O_k*
    where k* is the argmax (or a convex combination of all argmaxes).
    """
    ac = _fft_hankel(h)
    O = HALF - ac
    max_val = np.max(O)
    # Find all k achieving max (within tolerance)
    tol = 1e-9
    active_k = np.where(np.abs(O - max_val) < tol)[0]
    
    grad = np.zeros(N)
    for k in active_k:
        # dO_k/dh[j] = -[h[(j+k)%N] + h[(j-k)%N]]
        shifted_plus = np.roll(h, -k)  # h[(j+k)%N] for j=0..N-1
        shifted_minus = np.roll(h, k)   # h[(j-k)%N] for j=0..N-1
        grad -= (shifted_plus + shifted_minus)
    
    # Normalize by number of active shifts
    grad /= len(active_k)
    return grad

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Start with constant solution (known local minimum with overlap = 25)
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_overlap = _max_overlap(h)  # Should be 25.0
    
    np.random.seed(42)
    
    # Strategy: Use projected gradient descent on the spectral domain
    # Represent h as: h = 0.5 + sum_{m=1}^{M} (a_m cos(2*pi*m*i/N) + b_m sin(2*pi*m*i/N))
    # This ensures sum(h) = N/2 automatically.
    # We optimize the coefficients a_m, b_m.
    
    # Choose number of harmonics to include
    # Low frequencies have more impact on the overlap profile
    M = 20  # Number of harmonic pairs
    
    # Build the DCT/FFT basis matrix
    # B[i, 2m] = cos(2*pi*m*i/N), B[i, 2m+1] = sin(2*pi*m*i/N)
    idx = np.arange(N)
    B = np.zeros((N, 2 * M))
    for m in range(1, M + 1):
        angle = 2 * np.pi * m * idx / N
        B[:, 2 * (m - 1)] = np.cos(angle)
        B[:, 2 * (m - 1) + 1] = np.sin(angle)
    
    # Optimize coefficients using projected gradient descent
    # Initial coefficients are zero (constant h)
    coeffs = np.zeros(2 * M)
    
    # Adaptive step size
    step_size = 0.1
    prev_overlap = best_overlap
    stagnation_count = 0
    
    max_iter = 50000
    
    for iteration in range(max_iter):
        if time.time() - start_time > deadline - 5:
            break
        
        # Current h
        h = 0.5 + B @ coeffs
        
        # Project to feasible set
        h = _project_to_feasible(h)
        
        # Compute overlap and gradient
        overlap = _max_overlap(h)
        
        if overlap < best_overlap - 1e-10:
            best_overlap = overlap
            best_h = h.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
        
        # Compute gradient w.r.t. h
        grad_h = _gradient_max_overlap(h)
        
        # Project gradient to the subspace spanned by B
        # grad_coeffs = B^T @ grad_h
        grad_coeffs = B.T @ grad_h
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_coeffs)
        if grad_norm < 1e-12:
            break
        grad_coeffs /= grad_norm
        
        # Adaptive step size: backtracking line search
        alpha = step_size
        for _ in range(20):
            new_coeffs = coeffs - alpha * grad_coeffs
            new_h = 0.5 + B @ new_coeffs
            new_h = _project_to_feasible(new_h)
            new_overlap = _max_overlap(new_h)
            if new_overlap < overlap:
                break
            alpha *= 0.5
        else:
            alpha = 0.0  # No improvement possible
        
        if alpha > 0:
            coeffs = coeffs - alpha * grad_coeffs
            h = 0.5 + B @ coeffs
            h = _project_to_feasible(h)
        else:
            # Stagnation: try a random perturbation in the coefficient space
            if stagnation_count > 50:
                # Restart with a random perturbation
                coeffs += np.random.normal(0, 0.05, size=2 * M)
                step_size = max(step_size * 0.5, 0.001)
                stagnation_count = 0
            continue
        
        # Occasionally try to increase step size
        if overlap < prev_overlap:
            step_size = min(step_size * 1.1, 1.0)
        prev_overlap = overlap
    
    # Final refinement: local search in the time domain
    if time.time() - start_time < deadline - 10:
        h = best_h.copy()
        cur_overlap = _max_overlap(h)
        local_best_h = h.copy()
        local_best_overlap = cur_overlap
        
        for step in range(20000):
            if time.time() - start_time > deadline - 3:
                break
            
            # Try modifying a random subset of entries
            n_modify = np.random.randint(1, 6)
            idxs = np.random.choice(N, n_modify, replace=False)
            
            original_vals = h[idxs].copy()
            
            # Random perturbation
            deltas = np.random.uniform(-0.02, 0.02, n_modify)
            h[idxs] += deltas
            h = _project_to_feasible(h)
            
            new_overlap = _max_overlap(h)
            
            if new_overlap < local_best_overlap - 1e-10:
                local_best_overlap = new_overlap
                local_best_h = h.copy()
                if local_best_overlap < best_overlap:
                    best_overlap = local_best_overlap
                    best_h = local_best_h.copy()
            else:
                h[idxs] = original_vals
                h = _project_to_feasible(h)
    
    # Ensure exact sum and bounds
    diff = HALF - np.sum(best_h)
    if abs(diff) > 1e-15:
        # Distribute residual
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        n_mask = np.sum(mask)
        if n_mask > 0:
            best_h[mask] += diff / n_mask
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
