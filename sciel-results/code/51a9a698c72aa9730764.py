"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Cyclic Coordinate Descent with Ternary Search on each coordinate.
This exploits the convexity of the max-of-linear-functions objective with respect to each individual coordinate.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def _project_onto_box_and_sum(h):
    """
    Project h onto the set {x : 0 <= x_i <= 1, sum(x) = N/2} using
    alternating projections (POCS).
    """
    h = np.array(h, dtype=np.float64, copy=True)
    for _ in range(30):
        # Project onto box
        h = np.clip(h, 0.0, 1.0)
        # Project onto sum constraint
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    # Initialize with a good starting point
    # Cosine wave is a good initial guess for low overlap
    i = np.arange(N)
    h = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h = _fix_sum(h)
    
    best_h = h.copy()
    best_score = _score(h)
    
    # Cyclic Coordinate Descent
    # We iterate through coordinates and optimize each one.
    # Since the objective is convex in each coordinate, we can use ternary search.
    
    def _optimize_coordinate(h, idx):
        """
        Optimize h[idx] while keeping other elements fixed.
        We need to handle the sum constraint.
        If we change h[idx], the sum changes.
        To maintain the sum constraint exactly, we can either:
        1. Optimize h[idx] and then re-project the whole vector (slow, breaks coordinate descent purity).
        2. Optimize h[idx] and adjust one other element to maintain sum (complex).
        3. Optimize h[idx] assuming the sum constraint is handled by the projection at the end of each full cycle.
        
        Given N=100, we can afford to re-project after each coordinate update or after a full cycle.
        Re-projecting after each coordinate update is safer but slower.
        Let's try optimizing h[idx] in [0,1] minimizing score, then fixing the sum by adjusting the element with the most "room" or just re-projecting the whole vector.
        
        Actually, a better way:
        The constraint is sum(h) = N/2.
        If we vary h[idx], we must vary some other h[j] to keep sum constant?
        Or we can just vary h[idx] and then fix the sum by scaling/adjusting others?
        
        Let's stick to a simpler approach:
        1. Pick idx.
        2. Find best h[idx] in [0,1] that minimizes max_overlap, assuming other h's are fixed.
        3. Update h[idx].
        4. Re-project the entire vector onto {0<=x<=1, sum=N/2}.
        
        This is effectively a projected subgradient step but with exact 1D minimization.
        """
        old_val = h[idx]
        
        # Ternary search on [0, 1]
        lo = 0.0
        hi = 1.0
        # We need a function that takes h_val and returns score.
        # But we must fix the sum.
        # If we just change h[idx], the sum is off.
        # We can define an effective function:
        #   f(x) = score(_fix_sum(h_with_h_idx_set_to_x))
        # This is convex.
        
        def f(x):
            h_temp = h.copy()
            h_temp[idx] = x
            h_temp = _fix_sum(h_temp)
            return _score(h_temp)
        
        # Ternary search
        for _ in range(30): # ~1e-9 precision
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if f(m1) < f(m2):
                hi = m2
            else:
                lo = m1
        
        new_val = (lo + hi) / 2.0
        h[idx] = new_val
        # Re-project to ensure constraints are met
        h = _project_onto_box_and_sum(h)
        return h
    
    # Main loop
    max_iters = 200
    for it in range(max_iters):
        if time.time() - t0 > _DEADLINE - 5:
            break
            
        # Shuffle order to avoid bias? Or cyclic? Cyclic is fine.
        for idx in range(N):
            h = _optimize_coordinate(h, idx)
            
        # Check improvement
        current_score = _score(h)
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        else:
            # If no improvement in a full cycle, we might be stuck or converged
            # We can break early if improvement is tiny
            if best_score - current_score < 1e-9:
                break

    return best_h
