import time
import numpy as np
from scipy.optimize import linprog

N = 600
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_score(heights):
    """Compute the normalized max autoconvolution: max_k conv[k] / (sum h)^2"""
    h = np.array(heights, dtype=np.float64)
    s = h.sum()
    if s <= 0:
        return np.inf
    # Linear convolution
    conv = np.convolve(h, h, mode='full')
    return conv.max() / (s * s)

def _try_two_rectangles(a, b, c, d):
    """
    Try a two-rectangle structure:
    Rectangle 1: indices [a, b) with height h1
    Rectangle 2: indices [c, d) with height h2
    Optimize h1, h2 >= 0 to minimize score.
    """
    if a >= b or c >= d:
        return None, None
    
    # Build the pattern
    pattern = np.zeros(N)
    pattern[a:b] = 1.0
    pattern[c:d] = 1.0
    
    # For two rectangles, the optimal heights can be found analytically
    # The score is max_k conv[k] / (h1*len1 + h2*len2)^2
    # conv is linear in (h1, h2) for each lag, so we can optimize
    
    len1 = b - a
    len2 = d - c
    
    # For simplicity, try equal heights first, then optimize ratio
    best_score = np.inf
    best_h1 = 1.0
    best_h2 = 1.0
    
    # Grid search over ratio h2/h1
    for ratio in np.linspace(0.1, 5.0, 50):
        h1 = 1.0
        h2 = ratio
        h = np.zeros(N)
        h[a:b] = h1
        h[c:d] = h2
        s = h.sum()
        if s <= 0:
            continue
        conv = np.convolve(h, h, mode='full')
        score = conv.max() / (s * s)
        if score < best_score:
            best_score = score
            best_h1 = h1
            best_h2 = h2
    
    if best_score == np.inf:
        return None, None
    
    h = np.zeros(N)
    h[a:b] = best_h1
    h[c:d] = best_h2
    return h, best_score

def _optimize_flat_with_gaps():
    """
    Try a flat-top structure with gaps at the edges.
    f(x) = h for x in [-1/4+g, 1/4-g], 0 otherwise.
    """
    best_score = np.inf
    best_h = None
    
    # Try different gap fractions
    for gap_frac in np.linspace(0.0, 0.3, 30):
        gap = int(N * gap_frac)
        if gap >= N // 2:
            continue
        h = np.zeros(N)
        h[gap:N-gap] = 1.0
        s = h.sum()
        if s <= 0:
            continue
        conv = np.convolve(h, h, mode='full')
        score = conv.max() / (s * s)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h, best_score

def _optimize_cosine_taper():
    """
    Try a cosine-tapered distribution: h[i] = 0.5*(1 + cos(pi * (2i-N)/(N-1)))
    or similar smooth profiles.
    """
    best_score = np.inf
    best_h = None
    
    for alpha in np.linspace(0.0, 2.0, 20):
        i = np.arange(N)
        # Taper: 1 in center, goes to 0 at edges with parameter alpha
        x = 2.0 * i / (N - 1) - 1.0  # range [-1, 1]
        h = np.maximum(0.0, 1.0 - alpha * x * x)
        s = h.sum()
        if s <= 0:
            continue
        conv = np.convolve(h, h, mode='full')
        score = conv.max() / (s * s)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h, best_score

def _optimize_three_rectangles():
    """
    Try a three-rectangle structure: two outer rectangles and one inner.
    """
    best_score = np.inf
    best_h = None
    
    # Try symmetric three-rectangle patterns
    for outer_len_frac in np.linspace(0.05, 0.25, 10):
        for inner_len_frac in np.linspace(0.1, 0.4, 10):
            outer_len = int(N * outer_len_frac)
            inner_len = int(N * inner_len_frac)
            if outer_len < 1 or inner_len < 1:
                continue
            
            # Place: outer_left, gap, inner, gap, outer_right
            # Total: 2*outer_len + inner_len <= N
            if 2 * outer_len + inner_len > N:
                continue
            
            gap = (N - 2 * outer_len - inner_len) // 2
            if gap < 0:
                continue
            
            h = np.zeros(N)
            h[:outer_len] = 1.0
            h[outer_len + gap : outer_len + gap + inner_len] = 1.0
            h[N - outer_len:] = 1.0
            
            s = h.sum()
            if s <= 0:
                continue
            conv = np.convolve(h, h, mode='full')
            score = conv.max() / (s * s)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    return best_h, best_score

def construct():
    """
    Returns a sequence of nonnegative step heights on equal intervals of [-1/4, 1/4].
    Uses structural search over two-rectangle, flat-with-gaps, cosine-taper,
    and three-rectangle families to find a good solution.
    """
    start_time = time.time()
    
    # Baseline: uniform
    uniform = np.ones(N)
    best_h = uniform.copy()
    best_score = _compute_score(uniform)
    
    # Try flat with gaps
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        h, s = _optimize_flat_with_gaps()
        if h is not None and s < best_score:
            best_score = s
            best_h = h
    
    # Try cosine taper
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        h, s = _optimize_cosine_taper()
        if h is not None and s < best_score:
            best_score = s
            best_h = h
    
    # Try two rectangles (coarse grid)
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        # Try symmetric two-rectangle: [0, a) and [N-a, N)
        for a in range(10, N // 2 - 10, 15):
            h1, s1 = _try_two_rectangles(0, a, N - a, N)
            if h1 is not None and s1 < best_score:
                best_score = s1
                best_h = h1
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
        
        # Try two rectangles with gap in middle: [0, a) and [b, N)
        for a in range(50, N // 3, 20):
            for b in range(N // 2, N - 50, 20):
                if b - a < 10:
                    continue
                h1, s1 = _try_two_rectangles(0, a, b, N)
                if h1 is not None and s1 < best_score:
                    best_score = s1
                    best_h = h1
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - 10:
                    break
            if elapsed > DEADLINE - 10:
                break
    
    # Try three rectangles
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        h, s = _optimize_three_rectangles()
        if h is not None and s < best_score:
            best_score = s
            best_h = h
    
    # Ensure nonnegative
    best_h = np.maximum(0.0, best_h)
    
    return list(best_h)

if __name__ == "__main__":
    # Simple self-check
    res = construct()
    assert len(res) == N
    assert all(h >= 0 for h in res)
    print(f"Constructed step function. Score: {_compute_score(res)}")
