import numpy as np
import time

N = 400
DEADLINE = 115.0

def _compute_c3_max(h):
    """
    Compute max |C3[k]| for a binary sequence h.
    C3[k] = sum_i h[i] h[i+k] h[i+2k]
    """
    h_float = h.astype(np.float64)
    size = 1 << (2 * N - 1).bit_length()
    
    h_fft = np.fft.fft(h_float, n=size)
    c3_fft = h_fft * h_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    
    return np.max(np.abs(c3_full[:N]))

def _score_normalized(h):
    return _compute_c3_max(h) / N

def _compute_c3_full(h):
    """Compute full C3 array."""
    h_float = h.astype(np.float64)
    size = 1 << (2 * N - 1).bit_length()
    h_fft = np.fft.fft(h_float, n=size)
    c3_fft = h_fft * h_fft * h_fft
    c3_full = np.fft.ifft(c3_fft).real
    return c3_full[:N]

def _batched_flip_deltas(h, c3):
    """
    For each index i, compute the change in max|C3| if we flip h[i].
    Uses the fact that flipping h[i] changes C3[k] by:
    ΔC3[k] = -2*h[i]*(h[i+k]*h[i+2k] + h[i-k]*h[i-2k] + h[i-2k]*h[i-k])
    (with terms zeroed when indices are out of range)
    
    Returns array of shape (N,) where entry i is the new max|C3| after flipping h[i].
    """
    h_float = h.astype(np.float64)
    N_local = N
    
    # Precompute the "interaction" terms
    # For flip at index i, the delta to C3[k] is:
    # -2*h[i] * (h[i+k]*h[i+2k] + h[i-k]*h[i-2k] + h[i-2k]*h[i-k])
    # for valid indices.
    
    # We can compute this for all i and k simultaneously using shifts.
    # Let's compute the contribution for each i.
    
    # For efficiency, compute the delta C3 for each flip as a vector of length N.
    # delta_c3[i, k] = -2*h[i] * (h[i+k]*h[i+2k] + h[i-k]*h[i-2k] + h[i-2k]*h[i-k])
    
    # This is expensive to compute naively. Instead, let's just evaluate
    # a subset of flips per iteration to keep it fast.
    
    # Actually, let's use a simpler approach: evaluate all flips in a batch
    # by doing N FFTs? No, that's too slow.
    
    # Better: use the analytical formula but vectorize over i.
    # For each i, compute delta_c3_i[k] for k=0..N-1.
    
    # Let's precompute shifted versions of h.
    # h_shifted_plus_k[i] = h[i+k] if i+k < N else 0
    # h_shifted_minus_k[i] = h[i-k] if i-k >= 0 else 0
    
    # For a given k, the delta for flip at i is:
    # -2*h[i] * (h[i+k]*h[i+2k] + h[i-k]*h[i-2k] + h[i-2k]*h[i-k])
    
    # Let's compute this for all i for a given k, then take max over k.
    
    # Actually, the cleanest approach: for each candidate flip i, compute the
    # new C3 array and take max. But that's O(N^2) per iteration.
    
    # Let's just do greedy with a limited number of flips per pass, but use
    # a smarter initialization and more restarts.
    
    # For now, return a simple estimate: assume the flip doesn't change max much.
    # This is a placeholder - we'll use the direct method in the main loop.
    
    # Compute the "sensitivity" of each position: how much it contributes to C3.
    # pos_weight[i] = sum_k |contribution of h[i] to C3[k]|
    
    # Contribution of h[i] to C3[k]:
    # h[i] appears in C3[k] when i = j, i = j+k, or i = j+2k.
    # So C3[k] = ... + h[i]*h[i+k]*h[i+2k] + h[i-k]*h[i]*h[i+k] + h[i-2k]*h[i-k]*h[i]
    
    # The weight of position i is sum_k |h[i+k]*h[i+2k] + h[i-k]*h[i+k] + h[i-2k]*h[i-k]|
    
    # Let's compute this.
    weights = np.zeros(N_local)
    for k in range(N_local):
        # Term 1: i, i+k, i+2k all in range
        valid1 = (np.arange(N_local) + k < N_local) & (np.arange(N_local) + 2*k < N_local)
        t1 = np.zeros(N_local)
        idx1 = np.where(valid1)[0]
        if len(idx1) > 0:
            t1[idx1] = h_float[idx1 + k] * h_float[idx1 + 2*k]
        
        # Term 2: i-k, i, i+k all in range
        valid2 = (np.arange(N_local) - k >= 0) & (np.arange(N_local) + k < N_local)
        t2 = np.zeros(N_local)
        idx2 = np.where(valid2)[0]
        if len(idx2) > 0:
            t2[idx2] = h_float[idx2 - k] * h_float[idx2 + k]
        
        # Term 3: i-2k, i-k, i all in range
        valid3 = (np.arange(N_local) - 2*k >= 0) & (np.arange(N_local) - k >= 0)
        t3 = np.zeros(N_local)
        idx3 = np.where(valid3)[0]
        if len(idx3) > 0:
            t3[idx3] = h_float[idx3 - 2*k] * h_float[idx3 - k]
        
        weights += np.abs(t1 + t2 + t3)
    
    # The expected improvement from flipping i is proportional to 2*|h[i]|*weights[i] = 2*weights[i]
    # (since h[i] = ±1)
    # New max |C3| will be roughly max(|C3[k]| - 2*weights[i], ...) but this is rough.
    
    # For ranking flips, use weights as a proxy for potential improvement.
    return weights

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    best_score = float('inf')
    best_h = np.random.choice([1, -1], N)
    
    # Van der Corput / low-discrepancy initialization
    def van_der_corput(n, base=2):
        """Generate first n terms of van der Corput sequence."""
        result = []
        for i in range(1, n + 1):
            x = 0.0
            f = 1.0 / base
            while i > 0:
                x += (i % base) * f
                i //= base
                f /= base
            result.append(x)
        return np.array(result)
    
    # Use van der Corput to determine signs: sign = 1 if frac < 0.5 else -1
    vd = van_der_corput(N)
    h_vd = np.where(vd < 0.5, 1.0, -1.0)
    
    # Also try a few other structured initializations
    initializations = [
        np.random.choice([1, -1], N).astype(np.float64),
        h_vd.copy(),
        np.sign(np.sin(2 * np.pi * 3 * np.linspace(0, 1, N, endpoint=False))).astype(np.float64),
        np.tile(np.array([1, 1, -1, -1]), N // 4 + 1)[:N].astype(np.float64),
        np.tile(np.array([1, -1]), N // 2 + 1)[:N].astype(np.float64),
    ]
    
    # Pick the best initialization
    for h_init in initializations:
        s = _score_normalized(h_init)
        if s < best_score:
            best_score = s
            best_h = h_init.copy()
    
    h_work = best_h.copy()
    current_score = best_score
    
    # Greedy coordinate descent with full C3 computation
    # To speed up, we'll do multiple passes, and within each pass,
    # we'll evaluate flips in a shuffled order.
    
    max_passes = 50
    for pass_num in range(max_passes):
        if time_left() < 5.0:
            break
        
        improved = False
        order = np.random.permutation(N)
        
        for idx in order:
            if time_left() < 2.0:
                break
            
            # Flip
            h_work[idx] *= -1
            new_score = _score_normalized(h_work)
            
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_work.copy()
            else:
                h_work[idx] *= -1
        
        if not improved:
            break
        
        # After a full pass with no improvement, try a few random restarts
        if pass_num > 0 and pass_num % 10 == 0:
            # Random perturbation: flip 5% of positions
            n_flips = max(1, N // 20)
            flip_indices = np.random.choice(N, n_flips, replace=False)
            h_work[flip_indices] *= -1
            new_score = _score_normalized(h_work)
            if new_score < current_score:
                current_score = new_score
            else:
                h_work[flip_indices] *= -1
    
    # Final polish: try all single flips once more
    if time_left() > 3.0:
        for idx in range(N):
            if time_left() < 1.0:
                break
            h_work[idx] *= -1
            new_score = _score_normalized(h_work)
            if new_score < current_score - 1e-12:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h_work.copy()
            else:
                h_work[idx] *= -1
    
    best_h = np.sign(best_h).astype(np.float64)
    if np.all(best_h == 0):
        best_h = np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print("Constructed:", h[:20])
    print("Score:", _score_normalized(h))
