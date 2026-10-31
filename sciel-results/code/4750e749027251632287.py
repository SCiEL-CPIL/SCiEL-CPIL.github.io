import numpy as np
import time
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 115.0


def _autoconv_max_abs(h):
    """Compute max |autoconvolution| of h."""
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))


def _squared_integral(h):
    """Compute (sum h)^2."""
    s = np.sum(h)
    return s * s


def _score(h):
    """Compute the objective: max|autoconv| / (sum h)^2. Lower is better."""
    si = _squared_integral(h)
    if abs(si) < 1e-12:
        return np.inf
    return _autoconv_max_abs(h) / si


def _square_wave_autoconv_max_abs(a, b, offset):
    """
    Compute max|autoconv| for a square wave: +1 on [offset, offset+a), -1 on [offset+a, offset+a+b).
    The autoconvolution of such a signal is piecewise quadratic.
    We compute it via FFT for correctness but this function is for quick evaluation.
    """
    h = np.zeros(N)
    start_pos = offset
    end_pos = offset + a
    if end_pos > N:
        return np.inf
    h[start_pos:end_pos] = 1.0
    start_neg = end_pos
    end_neg = end_pos + b
    if end_neg > N:
        return np.inf
    h[start_neg:end_neg] = -1.0
    ac = fftconvolve(h, h, mode='full')
    return np.max(np.abs(ac))


def _square_wave_sum(a, b):
    """Sum of the square wave: a - b."""
    return a - b


def _score_square_wave(a, b, offset):
    """Score for square wave with params a, b, offset."""
    s = a - b
    if abs(s) < 1e-12:
        return np.inf
    max_ac = _square_wave_autoconv_max_abs(a, b, offset)
    return max_ac / (s * s)


def _optimize_square_wave():
    """
    Optimize over square wave parameters (a, b, offset).
    Use grid search + local refinement.
    """
    best_score = np.inf
    best_params = (N // 2, N // 2 - 1, 0)  # default: mostly +1 with one -1
    
    # Grid search over a, b, offset
    # a: length of +1 block, b: length of -1 block
    # offset: starting position
    
    # To keep sum nonzero, we need a != b
    # Try various configurations
    
    a_values = list(range(1, N + 1, 2))
    b_values = list(range(1, N + 1, 2))
    offset_values = list(range(0, N, 10))
    
    # Coarse grid
    for a in a_values:
        for b in b_values:
            if a == b:
                continue
            for offset in offset_values:
                if offset + a + b > N:
                    continue
                sc = _score_square_wave(a, b, offset)
                if sc < best_score:
                    best_score = sc
                    best_params = (a, b, offset)
    
    # Local refinement around best
    a0, b0, off0 = best_params
    for da in range(-5, 6):
        for db in range(-5, 6):
            for doff in range(-5, 6):
                a = a0 + da
                b = b0 + db
                off = off0 + doff
                if a < 1 or b < 1 or off < 0 or off + a + b > N:
                    continue
                if a == b:
                    continue
                sc = _score_square_wave(a, b, off)
                if sc < best_score:
                    best_score = sc
                    best_params = (a, b, off)
    
    # Build the sequence
    a, b, offset = best_params
    h = np.zeros(N)
    h[offset:offset + a] = 1.0
    h[offset + a:offset + a + b] = -1.0
    
    return h, best_score


def _continuous_refine(h, max_iters=200, lr=0.01):
    """
    Gradient-based refinement on continuous heights in [-1, 1].
    Objective: max|autoconv| / (sum h)^2
    """
    h = h.copy().astype(float)
    best_h = h.copy()
    best_score = _score(h)
    
    for it in range(max_iters):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        s = np.sum(h)
        if abs(s) < 1e-6:
            break
        
        # Compute autoconvolution
        ac = fftconvolve(h, h, mode='full')
        max_abs = np.max(np.abs(ac))
        
        # Find which lag achieves the max
        idx = np.argmax(np.abs(ac))
        sign_ac = np.sign(ac[idx])
        
        # Gradient of max|ac| w.r.t. h: 
        # ac[k] = sum_i h[i] h[i+k-N+1] (for full mode, need to be careful)
        # d(ac[k])/d(h[j]) = h[j+k-N+1] + h[j-k+N-1] (approximately)
        # For the max term, gradient is sign(ac[idx]) * d(ac[idx])/d(h)
        
        # Simpler: use subgradient. The max is achieved at lag idx.
        # ac[idx] = sum_i h[i] * h[idx - i + N - 1] (in full convolution indexing)
        # Let's compute gradient numerically for stability
        
        # Actually, let's use a simpler approach: projected gradient on the log objective
        # f(h) = log(max|ac|) - 2*log(|sum h|)
        
        # Numerical gradient
        grad = np.zeros_like(h)
        eps = 1e-4
        for i in range(N):
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                break
            h_plus = h.copy()
            h_plus[i] += eps
            s_plus = np.sum(h_plus)
            if abs(s_plus) < 1e-8:
                continue
            ac_plus = fftconvolve(h_plus, h_plus, mode='full')
            max_plus = np.max(np.abs(ac_plus))
            f_plus = np.log(max_plus + 1e-12) - 2 * np.log(abs(s_plus) + 1e-12)
            
            h_minus = h.copy()
            h_minus[i] -= eps
            s_minus = np.sum(h_minus)
            if abs(s_minus) < 1e-8:
                continue
            ac_minus = fftconvolve(h_minus, h_minus, mode='full')
            max_minus = np.max(np.abs(ac_minus))
            f_minus = np.log(max_minus + 1e-12) - 2 * np.log(abs(s_minus) + 1e-12)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        # Projected gradient step
        h_new = h - lr * grad
        h_new = np.clip(h_new, -1.0, 1.0)
        
        new_score = _score(h_new)
        if new_score < best_score:
            best_score = new_score
            best_h = h_new.copy()
            h = h_new
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_h, best_score


def _bipartite_refine(h, max_rounds=50):
    """
    Faster refinement: flip pairs of elements to reduce max autoconv.
    """
    h = h.copy().astype(float)
    best_score = _score(h)
    
    for _ in range(max_rounds):
        if time.time() - _START_TIME > _TIME_LIMIT - 5:
            break
        
        improved = False
        # Try flipping individual elements
        for i in range(N):
            if time.time() - _START_TIME > _TIME_LIMIT - 5:
                return h, best_score
            
            h[i] *= -1
            new_score = _score(h)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                h[i] *= -1
        
        if not improved:
            break
    
    return h, best_score


def construct():
    """Construct a signed step function minimizing max|autoconv|/(sum h)^2."""
    
    best_h = None
    best_score = np.inf
    
    # Step 1: Optimize square wave family
    h_sw, sc_sw = _optimize_square_wave()
    if sc_sw < best_score:
        best_score = sc_sw
        best_h = h_sw.copy()
    
    # Step 2: Also try Thue-Morse as a starting point
    tm = np.array([0])
    while len(tm) < N:
        tm = np.concatenate([tm, 1 - tm])
    tm = tm[:N]
    h_tm = np.where(tm == 0, 1.0, -1.0)
    sc_tm = _score(h_tm)
    if sc_tm < best_score:
        best_score = sc_tm
        best_h = h_tm.copy()
    
    # Step 3: Bipartite refinement on best
    if best_h is not None:
        h_ref, sc_ref = _bipartite_refine(best_h, max_rounds=30)
        if sc_ref < best_score:
            best_score = sc_ref
            best_h = h_ref.copy()
    
    # Step 4: Continuous refinement if time allows
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 30:
        h_cont, sc_cont = _continuous_refine(best_h, max_iters=50, lr=0.005)
        if sc_cont < best_score:
            best_score = sc_cont
            best_h = h_cont.copy()
        
        # Snap to ±1
        best_h = np.sign(best_h)
        best_h[best_h == 0] = 1.0
        sc_snap = _score(best_h)
        if sc_snap < best_score:
            best_score = sc_snap
    
    # Step 5: Final bipartite polish
    if best_h is not None and time.time() - _START_TIME < _TIME_LIMIT - 10:
        h_final, sc_final = _bipartite_refine(best_h, max_rounds=20)
        if sc_final < best_score:
            best_score = sc_final
            best_h = h_final.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h

# EVOLVE-BLOCK-END
