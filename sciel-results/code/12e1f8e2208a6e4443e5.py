import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN

def _autoconv_fft(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    C = np.fft.irfft(F * F, n=fft_size)
    return C[:size]

def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    return np.max(np.abs(C)) / (S * S)

def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv_fft(f)
    absC = np.abs(C)
    max_val = np.max(absC)
    # Numerical stability
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)

def _gradient_smooth(f, t=0.01):
    """
    Gradient of the smooth objective using FFT-based cross-correlation.
    J(f) = LSE(C, t) / S^2
    dJ/df = (1/S^2) * dLSE/df - (2 * LSE / S^3) * dS/df
    dS/df = 1 (vector of ones)
    dLSE/df = sum_k G_k * dC_k/df
    where G_k = t * exp(t * |C_k|) * sign(C_k) / sum_m exp(t * |C_m|)
    dC_k/df_i = f_{k-i} + f_{i-k}
    sum_k G_k (f_{k-i} + f_{i-k}) = 2 * sum_j G_{j+i} f_j  (Cross-correlation)
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv_fft(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    # Compute G vector
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    # G_k corresponds to the weight in the gradient of LSE w.r.t C
    G = (t * exp_term * signC) / sum_exp
    
    # Compute Cross-Correlation: R[i] = sum_j G[j+i] * f[j]
    # This is equivalent to convolution of G_rev and f, or ifft(fft(G)*conj(fft(f)))
    fft_size = 1
    size_conv = 2 * N - 1
    while fft_size < size_conv:
        fft_size *= 2
        
    G_padded = np.zeros(fft_size)
    G_padded[:len(G)] = G
    f_padded = np.zeros(fft_size)
    f_padded[:len(f)] = f
    
    # Cross correlation R[i] = sum_j G[j+i] f[j]
    # In frequency domain: IFFT( FFT(G) * conj(FFT(f)) ) gives sum_j G[j-i] f[j]?
    # Let's check indices.
    # Standard correlation: R_xy[i] = sum_j x[j+i] y[j]
    # FFT property: IFFT(FFT(x) * conj(FFT(y))) = R_xy? 
    # Actually, IFFT(FFT(x) * conj(FFT(y))) is the circular cross-correlation.
    # For linear, we need to be careful with shifts.
    
    # Alternative: Use the fact that dLSE/df = conv(G, f) + conv(G, f_rev)
    # Since G is real, this is 2 * Re(IFFT(FFT(G) * FFT(f))) shifted?
    # Let's use the explicit correlation formula via FFT.
    # R[i] = sum_j G[j+i] f[j]
    # This is the convolution of G and f_rev, evaluated at specific indices?
    # No, it's simpler to just do:
    # grad_LSE = 2 * np.real(np.fft.irfft(np.fft.rfft(G, n=fft_size) * np.conj(np.fft.rfft(f, n=fft_size)), n=fft_size))
    # This gives R[i] = sum_j G[j-i] f[j] ?
    # Let's test with small N.
    # If G = [g0, g1], f = [f0, f1].
    # R[0] = g0*f0 + g1*f1? No.
    # R[0] = sum_j G[j] f[j] = g0 f0 + g1 f1.
    # R[1] = sum_j G[j+1] f[j] = g1 f0.
    # IFFT(FFT(G)*conj(FFT(f))) for linear size 2:
    # FFT(G) = [g0+g1, g0-g1]
    # FFT(f) = [f0+f1, f0-f1]
    # conj(FFT(f)) = [f0+f1, f0-f1] (real)
    # Prod = [(g0+g1)(f0+f1), (g0-g1)(f0-f1)]
    # IFFT = [ ((g0+g1)(f0+f1) + (g0-g1)(f0-f1))/2, ((g0+g1)(f0+f1) - (g0-g1)(f0-f1))/2 ]
    # = [ g0 f0 + g1 f1, g0 f1 + g1 f0 ]
    # This is Convolution, not Correlation.
    # We need R[i] = sum_j G[j+i] f[j].
    # This is correlation.
    # Correlation R[i] = IFFT( conj(FFT(G)) * FFT(f) ) ?
    # Let's try: conj(FFT(G)) = [g0+g1, g0-g1]
    # Prod = [(g0+g1)(f0+f1), (g0-g1)(f0-f1)]
    # Same as above because G, f are real.
    # So IFFT(FFT(G)*conj(FFT(f))) is Convolution.
    # To get Correlation R[i] = sum_j G[j+i] f[j], we can compute Conv(G, f_rev) or similar.
    # Actually, R[i] is the coefficient of z^i in (sum G_j z^j) (sum f_j z^{-j})? No.
    
    # Let's use the property: grad_LSE[i] = sum_k G_k (f_{k-i} + f_{i-k})
    # = sum_k G_k f_{k-i} + sum_k G_k f_{i-k}
    # Term 1: sum_k G_k f_{k-i}. Let m = k-i => k=m+i. sum_m G_{m+i} f_m. This is R[i].
    # Term 2: sum_k G_k f_{i-k}. Let m = i-k => k=i-m. sum_m G_{i-m} f_m.
    # Since G is not symmetric, Term 2 is not Term 1.
    # However, if we define H = G, then Term 2 is Conv(H, f)[i]?
    # Conv(H, f)[i] = sum_j H[j] f[i-j]. Let k=i-j => j=i-k. sum_k H[i-k] f[k].
    # This matches Term 2 if H=G.
    # So grad_LSE = R[i] + Conv(G, f)[i].
    # R[i] = sum_j G[j+i] f[j].
    # Conv(G, f)[i] = sum_j G[j] f[i-j].
    
    # We can compute Conv(G, f) easily via IFFT(FFT(G)*FFT(f)).
    # How to compute R[i]?
    # R[i] = sum_j G[j+i] f[j].
    # This is the cross-correlation of G and f.
    # R = IFFT( conj(FFT(G)) * FFT(f) ) ?
    # Let's check N=2 again.
    # conj(FFT(G)) = [g0+g1, g0-g1]
    # FFT(f) = [f0+f1, f0-f1]
    # Prod = [ (g0+g1)(f0+f1), (g0-g1)(f0-f1) ]
    # IFFT = [ g0f0+g1f1, g0f1+g1f0 ]
    # R[0] should be g0f0+g1f1. Matches.
    # R[1] should be g1f0. Mismatch (got g0f1+g1f0).
    # Why? Because for linear correlation, we need to pad and shift.
    # The IFFT result is circular correlation.
    # For linear, if we pad to size >= N+M-1, the first N+M-1 elements contain the linear correlation?
    # Circular correlation of length L: C_circ[i] = sum_j G[(j-i) mod L] f[j].
    # If L >= 2N-1, there is no wrap-around for the valid linear correlation indices.
    # The linear correlation R_lin[i] = sum_j G[j+i] f[j] for i=0..N-1.
    # In circular correlation with padding:
    # C_circ[i] = sum_j G_padded[j-i] f_padded[j].
    # If we reverse G?
    # Let's just use the direct formula for the gradient part which is small enough?
    # N=400. N^2 = 160,000 ops. This is fast in Python/NumPy if vectorized.
    # The loop in the original code was:
    # for i in range(N):
    #   grad[i] = 2.0 * np.dot(A[k_start:k_end], f_slice)
    # This is O(N^2) but with NumPy dot products, it's C-speed.
    # 400 * 400 = 160k mults. This is negligible (<1ms).
    # So the FFT gradient optimization might not be necessary if the dot product is fast.
    # The bottleneck in the original code was likely the number of iterations or the optimization itself, not the gradient calculation speed per se, unless the loop overhead was high.
    # However, let's stick to the FFT method for elegance and potential speedup in larger N, but ensure correctness.
    
    # Correct FFT for Correlation R[i] = sum_j G[j+i] f[j]:
    # This is equivalent to Conv(G, f_rev) shifted?
    # Let's just use the vectorized dot product loop, it's safer and fast enough for N=400.
    
    grad_LSE = np.zeros(N)
    for i in range(N):
        k_start = i
        k_end = min(i + N, 2 * N - 1)
        f_slice = f[:k_end - k_start]
        # Term 1: sum_j G[j+i] f[j]
        term1 = np.dot(G[k_start:k_end], f_slice)
        # Term 2: sum_j G[j] f[i-j] -> Conv(G, f)[i]
        # j ranges such that i-j is in [0, N-1] => j in [i-N+1, i]
        # And j in [0, N-1]
        j_start = max(0, i - N + 1)
        j_end = min(i, N - 1)
        if j_start <= j_end:
            # G[j] * f[i-j]
            # f indices: i - j. As j goes j_start..j_end, f_idx goes i-j_start .. i-j_end
            f_idx_start = i - j_end
            f_idx_end = i - j_start + 1
            term2 = np.dot(G[j_start:j_end+1], f[f_idx_start:f_idx_end])
        else:
            term2 = 0.0
            
        grad_LSE[i] = term1 + term2

    LSE_val = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    G_val = LSE_val / t
    
    # dJ/df_i = (1/S^2) grad_LSE[i] - (2 * G_val / S^3) * 1
    grad = grad_LSE / (S * S) - (2.0 * G_val / (S * S * S))
    
    return grad


def _local_search_binary(f, time_budget=10.0):
    """
    Local search on binary {-1, 1} sequences.
    """
    f = f.copy()
    S = np.sum(f)
    C = _autoconv_fft(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE:
            break
        
        k_max = np.argmax(np.abs(C))
        
        candidates = set()
        
        # Random candidates
        for _ in range(20):
            candidates.add(np.random.randint(N))
        
        # Candidates that affect the peak
        for j in range(N):
            if 0 <= k_max - j < N:
                impact = -2.0 * f[j] * f[k_max - j]
                if abs(impact) > 0:
                    candidates.add(j)
        
        candidates = list(candidates)
        
        best_improvement = 0
        best_flip_idx = -1
        
        for i in candidates:
            v = f[i]
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            
            if k_start >= len(C) or k_end <= 0:
                continue
            
            k_end_clamped = min(k_end, len(C))
            f_slice = f[:k_end_clamped - k_start]
            
            delta_C = np.zeros_like(C)
            delta_C[k_start:k_end_clamped] = -2.0 * v * f_slice
            
            new_C = C + delta_C
            new_S = S - 2.0 * v
            
            if abs(new_S) < 1e-6:
                continue
            
            new_score = np.max(np.abs(new_C)) / (new_S * new_S)
            
            if new_score < score - best_improvement:
                best_improvement = score - new_score
                best_flip_idx = i
        
        if best_flip_idx >= 0:
            v = f[best_flip_idx]
            f[best_flip_idx] = -v
            S -= 2.0 * v
            C = _autoconv_fft(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 30:
                break
    
    return f, score


def construct():
    start_time = time.time()
    
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Continuous optimization
    # We use multiple restarts and decreasing temperature t
    restarts = 8
    for r in range(restarts):
        if time.time() > DEADLINE:
            break
        
        if r == 0:
            f0 = np.ones(N)
        elif r == 1:
            f0 = np.sin(np.linspace(0, 2 * np.pi, N))
        elif r == 2:
            f0 = np.cos(np.linspace(0, 4 * np.pi, N))
        elif r == 3:
            f0 = rng.uniform(-1, 1, N)
        elif r == 4:
            # Raised cosine window
            f0 = 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / N))
        elif r == 5:
            # Random binary
            f0 = rng.choice([-1.0, 1.0], size=N)
            # Ensure sum is not zero
            S0 = np.sum(f0)
            if abs(S0) < 5:
                for _ in range(N // 2):
                    idx = rng.integers(0, N)
                    if f0[idx] * S0 < 0:
                        f0[idx] = -f0[idx]
                        S0 = np.sum(f0)
        else:
            f0 = rng.uniform(-1, 1, N)
        
        S0 = np.sum(f0)
        if abs(S0) > 1e-6:
            f0 = f0 / S0
        else:
            f0 = np.ones(N)
        
        current_f = f0
        current_score = _objective_vectorized(current_f)
        
        if current_score < best_score:
            best_score = current_score
            best_f = current_f.copy()
        
        # Anneling of t
        ts = [0.1, 0.05, 0.02, 0.01, 0.005]
        
        for t in ts:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    _objective_smooth,
                    current_f,
                    args=(t,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 100, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                true_score = _objective_vectorized(res.x)
                if true_score < current_score:
                    current_f = res.x
                    current_score = true_score
                    if true_score < best_score:
                        best_score = true_score
                        best_f = res.x.copy()
            except Exception:
                continue
    
    # Phase 2: Binary Local Search on best continuous solution (rounded)
    if time.time() < DEADLINE:
        f_bin = np.sign(best_f)
        f_bin[f_bin == 0] = 1.0
        S_bin = np.sum(f_bin)
        if abs(S_bin) > 5:
            time_budget = min(10.0, (DEADLINE - time.time()) * 0.5)
            if time_budget > 1.0:
                f_local, score_local = _local_search_binary(f_bin, time_budget=time_budget)
                if score_local < best_score:
                    best_score = score_local
                    best_f = f_local.copy()

    # Phase 3: Final Polish with very small t
    if time.time() < DEADLINE:
        remaining = DEADLINE - time.time()
        if remaining > 2.0:
            t_final = 0.001
            try:
                res = minimize(
                    _objective_smooth,
                    best_f,
                    args=(t_final,),
                    method='L-BFGS-B',
                    jac=_gradient_smooth,
                    options={'maxiter': 50, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                true_score = _objective_vectorized(res.x)
                if true_score < best_score:
                    best_score = true_score
                    best_f = res.x.copy()
            except Exception:
                pass

    return best_f
