"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE_MARGIN = 5.0  # seconds to spare before the 150s wall-clock kill

# EVOLVE-BLOCK-START

def _elapsed_deadline(start_time):
    """Return True if we should stop to respect the deadline."""
    return (time.time() - start_time) >= (150.0 - DEADLINE_MARGIN)

def _cross_corr_max_overlap(h):
    """
    Compute the maximum over shifts k of the overlap between h and 1-h shifted.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum of this over k != 0.
    """
    N_local = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(1, N_local):
        # Overlap: sum_i h[i] * (1 - h[(i+k) % N])
        # = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
        # = (sum h) - corr_hh[k]
        # sum h = N/2 (by constraint)
        # So overlap = N/2 - autocorr_hh[k]
        # To minimize max overlap, we want to maximize min autocorr? No...
        # Actually overlap = sum_i h[i]*(1-h[(i+k)%N])
        # Let's just compute directly
        shifted = np.roll(h, k)
        ov = np.dot(h, 1.0 - shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _cross_corr_max_overlap_fast(h):
    """
    Faster: use FFT to compute all autocorrelations at once.
    overlap[k] = sum_i h[i]*(1 - h[(i+k)%N]) = sum(h) - sum_i h[i]*h[(i+k)%N]
    sum_i h[i]*h[(i+k)%N] is the circular autocorrelation of h at lag k.
    """
    N_local = len(h)
    sum_h = np.sum(h)
    # Circular autocorrelation via FFT
    F = np.fft.fft(h)
    acf = np.real(np.fft.ifft(F * np.conj(F)))
    # acf[k] = sum_i h[i] * h[(i+k) % N]
    # overlap[k] = sum_h - acf[k]
    overlaps = sum_h - acf
    # Exclude k=0 (shift 0)
    max_ov = np.max(overlaps[1:])
    return max_ov

def _bang_bang_init(N_local, seed):
    """
    Initialize with a bang-bang (0/1) structure.
    Erdos problem: we want h in [0,1], sum = N/2, to minimize max overlap.
    A natural starting point: alternating blocks or random 0/1 with sum N/2.
    """
    rng = np.random.RandomState(seed)
    # Start with a random permutation of N/2 ones and N/2 zeros
    arr = np.zeros(N_local)
    ones = rng.choice(N_local, size=N_local // 2, replace=False)
    arr[ones] = 1.0
    return arr

def _local_search(h, start_time, max_iters=5000, tol=1e-8):
    """
    Local search: try flipping individual elements or small groups.
    Since we need sum = N/2, we can swap a 0-position with a 1-position.
    """
    N_local = len(h)
    h = h.copy()
    cur_ov = _cross_corr_max_overlap_fast(h)
    
    # Identify indices where h is close to 0 and close to 1
    # For bang-bang, h[i] is 0 or 1.
    # A move: pick i where h[i]=1, j where h[i]=0, swap them.
    # This keeps the sum constant.
    
    pos_ones = np.where(h > 0.5)[0]
    pos_zeros = np.where(h < 0.5)[0]
    
    for it in range(max_iters):
        if _elapsed_deadline(start_time):
            break
        if len(pos_ones) == 0 or len(pos_zeros) == 0:
            break
        
        best_improvement = 0.0
        best_move = None
        
        # Sample a subset of moves to keep it fast
        n_sample = min(200, len(pos_ones) * len(pos_zeros))
        for _ in range(n_sample):
            i = pos_ones[np.random.randint(len(pos_ones))]
            j = pos_zeros[np.random.randint(len(pos_zeros))]
            if i == j:
                continue
            
            # Try swap
            h[i], h[j] = h[j], h[i]
            new_ov = _cross_corr_max_overlap_fast(h)
            h[i], h[j] = h[j], h[i]  # undo
            
            improvement = cur_ov - new_ov
            if improvement > best_improvement:
                best_improvement = improvement
                best_move = (i, j)
        
        if best_move is not None:
            i, j = best_move
            h[i], h[j] = h[j], h[i]
            cur_ov -= best_improvement
            # Update pos_ones and pos_zeros
            pos_ones = np.where(h > 0.5)[0]
            pos_zeros = np.where(h < 0.5)[0]
            
            if best_improvement < tol:
                break
        else:
            break
    
    return h, cur_ov

def _smooth_refine(h, start_time, max_iters=2000, lr=0.01):
    """
    Gradient-based refinement on the smooth surrogate.
    Objective: max_k overlap_k, approximated by soft-max.
    Constraint: sum(h) = N/2, h in [0,1].
    """
    N_local = len(h)
    h = h.copy()
    h = np.clip(h, 0.0, 1.0)
    
    # Enforce sum constraint via projection
    for _ in range(max_iters):
        if _elapsed_deadline(start_time):
            break
        
        # Compute overlaps via FFT
        F = np.fft.fft(h)
        acf = np.real(np.fft.ifft(F * np.conj(F)))
        overlaps = np.sum(h) - acf
        overlaps = overlaps[1:]  # exclude k=0
        
        # Soft-max approximation of max
        beta = 50.0
        log_sum_exp = np.log(np.sum(np.exp(beta * overlaps)))
        soft_max = (log_sum_exp) / beta
        
        # Gradient of soft-max w.r.t. h
        # d(soft_max)/d(h[i]) = sum_k exp(beta * overlap_k) / sum_m exp(beta * overlap_m) * d(overlap_k)/d(h[i])
        # overlap_k = sum_j h[j] - sum_j h[j]*h[(j+k)%N]
        # d(overlap_k)/d(h[i]) = 1 - sum_j [delta(j,i)*h[(j+k)%N] + h[j]*delta((j+k)%N, i)]
        # = 1 - h[(i+k)%N] - h[(i-k)%N]  (for circular)
        # Wait, let me re-derive.
        # overlap_k = sum_i h[i]*(1 - h[(i+k)%N])
        # d(overlap_k)/d(h[i]) = (1 - h[(i+k)%N]) + sum_j h[j] * d(1-h[(j+k)%N])/d(h[i])
        # = (1 - h[(i+k)%N]) - h[(i-k)%N]
        # So grad_k[i] = 1 - h[(i+k)%N] - h[(i-k)%N]
        
        weights = np.exp(beta * overlaps - log_sum_exp)  # normalized
        # weights[k-1] corresponds to overlap at shift k
        
        grad = np.zeros(N_local)
        for k in range(1, N_local):
            w = weights[k-1]
            # grad_k[i] = 1 - h[(i+k)%N] - h[(i-k)%N]
            grad += w * (1.0 - np.roll(h, k) - np.roll(h, -k))
        
        # Project gradient to satisfy sum constraint: subtract mean
        grad -= np.mean(grad)
        
        # Update
        h_new = h - lr * grad
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Project back to sum = N/2
        diff = np.sum(h_new) - N_local / 2.0
        h_new -= diff / N_local
        h_new = np.clip(h_new, 0.0, 1.0)
        
        # Re-project
        diff = np.sum(h_new) - N_local / 2.0
        h_new -= diff / N_local
        h_new = np.clip(h_new, 0.0, 1.0)
        
        if np.max(np.abs(h_new - h)) < 1e-10:
            h = h_new
            break
        h = h_new
    
    # Final projection to ensure exact sum
    diff = np.sum(h) - N_local / 2.0
    h -= diff / N_local
    h = np.clip(h, 0.0, 1.0)
    
    return h

def construct():
    start_time = time.time()
    
    best_h = None
    best_score = float('inf')
    
    # Multiple restarts with bang-bang initializations
    n_restarts = 20
    seed_base = 42
    
    for r in range(n_restarts):
        if _elapsed_deadline(start_time):
            break
        
        seed = seed_base + r * 1000
        h0 = _bang_bang_init(N, seed)
        
        # Local search
        h_ls, ov_ls = _local_search(h0, start_time, max_iters=3000)
        
        if ov_ls < best_score:
            best_score = ov_ls
            best_h = h_ls.copy()
        
        # Smooth refinement
        if not _elapsed_deadline(start_time):
            h_ref = _smooth_refine(h_ls, start_time, max_iters=1000, lr=0.05)
            # Project to exact sum
            diff = np.sum(h_ref) - N / 2.0
            h_ref -= diff / N
            h_ref = np.clip(h_ref, 0.0, 1.0)
            
            ov_ref = _cross_corr_max_overlap_fast(h_ref)
            if ov_ref < best_score:
                best_score = ov_ref
                best_h = h_ref.copy()
    
    # Final high-precision polish if we have time
    if best_h is not None and not _elapsed_deadline(start_time):
        h_polish = _smooth_refine(best_h, start_time, max_iters=3000, lr=0.01)
        diff = np.sum(h_polish) - N / 2.0
        h_polish -= diff / N
        h_polish = np.clip(h_polish, 0.0, 1.0)
        
        ov_polish = _cross_corr_max_overlap_fast(h_polish)
        if ov_polish < best_score:
            best_score = ov_polish
            best_h = h_polish
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure exact sum = N/2
    diff = np.sum(best_h) - N / 2.0
    best_h -= diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    
    # Final tiny correction
    diff = np.sum(best_h) - N / 2.0
    if abs(diff) > 1e-12:
        best_h[0] += diff
    
    return best_h

# EVOLVE-BLOCK-END
