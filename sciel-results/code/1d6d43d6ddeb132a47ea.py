"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _overlap_max(h, N):
    """Compute the maximum overlap of h with shifted 1-h for shifts 1..N-1."""
    max_ov = 0.0
    for shift in range(1, N):
        # overlap at shift: sum_i h[i] * (1 - h[i-shift mod N])
        # = sum_i h[i] - sum_i h[i] * h[i-shift mod N]
        h_shifted = np.roll(h, shift)
        cross = np.dot(h, h_shifted)
        ov = h.sum() - cross
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _construct_gradient(N, time_limit=120.0):
    """
    Use gradient-based optimization with log-sum-exp smoothing to minimize
    the maximum overlap.
    """
    deadline = time.time() + time_limit - 5.0  # 5s safety margin
    rng = np.random.default_rng(42)
    
    # Parameterize h with unconstrained parameters: h_i = sigmoid(u_i)
    # Sum constraint: sum(h) = N/2. We'll use a Lagrange multiplier approach
    # or simply project after each step.
    
    best_h = np.full(N, 0.5)
    best_val = _overlap_max(best_h, N)
    
    # Initialize with random perturbation
    u = rng.normal(0.0, 0.5, N)
    h = 1.0 / (1.0 + np.exp(-u))
    # Project to satisfy sum constraint
    h = (h - h.mean()) * N / (N * h.sum()) + 0.5  # rough projection
    h = np.clip(h, 0.0, 1.0)
    # Re-project properly: find alpha, beta such that h' = clip(alpha*h + beta, 0, 1) has sum N/2
    # Simpler: use bisection on the level set
    def project_to_sum(h_raw, target_sum):
        """Project h_raw onto {h in [0,1]^N : sum(h) = target_sum} via soft clipping."""
        # Use iterative bisection on threshold
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            h_proj = np.clip(h_raw + mid, 0.0, 1.0)
            s = h_proj.sum()
            if s > target_sum:
                lo = mid
            else:
                hi = mid
        h_proj = np.clip(h_raw + (lo + hi) / 2, 0.0, 1.0)
        # Fine-tune: distribute remaining mass uniformly among non-saturated elements
        s = h_proj.sum()
        diff = target_sum - s
        if abs(diff) > 1e-12:
            mask = (h_proj > 0.01) & (h_proj < 0.99)
            if mask.any():
                h_proj[mask] += diff / mask.sum()
                h_proj = np.clip(h_proj, 0.0, 1.0)
        return h_proj
    
    h = project_to_sum(h, N / 2.0)
    
    lr = 0.1
    beta = 10.0  # log-sum-exp temperature
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute smoothed max overlap using log-sum-exp
        # Overlaps for all shifts
        overlaps = np.zeros(N - 1)
        for shift in range(1, N):
            h_shifted = np.roll(h, shift)
            cross = np.dot(h, h_shifted)
            overlaps[shift - 1] = h.sum() - cross
        
        # Log-sum-exp of overlaps
        max_ov = overlaps.max()
        if max_ov > 0:
            lse = max_ov + np.log(np.sum(np.exp(overlaps - max_ov)))
            # Gradient of lse w.r.t. overlaps: softmax(overlaps)
            weights = np.exp(overlaps - lse)  # softmax
        else:
            lse = 0.0
            weights = np.ones(N - 1) / (N - 1)
        
        # Gradient of overlap w.r.t. h:
        # overlap_shift = sum_i h[i] - sum_i h[i]*h[i-shift]
        # d(overlap_shift)/dh[j] = 1 - h[j-shift] - h[j+shift]  (with modular indexing)
        # Actually: d/dh[j] of sum_i h[i]*h[i-shift] = h[j-shift] + h[j+shift]
        # So gradient of overlap_shift w.r.t. h[j] = 1 - h[j-shift] - h[j+shift]
        
        grad = np.zeros(N)
        for shift in range(1, N):
            w = weights[shift - 1]
            if w < 1e-10:
                continue
            grad += w * (1.0 - np.roll(h, shift) - np.roll(h, -shift))
        
        # But h is a function of u: h = sigmoid(u), dh/du = h*(1-h)
        # We're optimizing over u, but with the sum constraint.
        # Simplification: work directly with h and project.
        # Gradient descent on h directly, then project.
        
        h_new = h - lr * grad
        h_new = project_to_sum(h_new, N / 2.0)
        
        val = _overlap_max(h_new, N)
        if val < best_val:
            best_val = val
            best_h = h_new.copy()
        h = h_new
        
        # Adaptive learning rate
        lr *= 0.999
        
    return best_h, best_val


def _construct_binary(N, time_limit=120.0):
    """
    Try binary sequences (h_i in {0,1}) with sum N/2 = N/2 ones.
    For N=100, this is 50 ones and 50 zeros.
    Minimize max overlap.
    """
    deadline = time.time() + time_limit - 5.0
    rng = np.random.default_rng(123)
    
    best_h = None
    best_val = float('inf')
    
    # Start with alternating pattern
    h = np.zeros(N)
    h[::2] = 1.0
    # For N=100, h[::2] gives 50 ones. Sum = 50 = N/2. Good.
    val = _overlap_max(h, N)
    best_h = h.copy()
    best_val = val
    
    # Local search: flip pairs (one 1->0 and one 0->1)
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Pick two random indices to swap
        idx = np.where(h == 1.0)[0]
        jdx = np.where(h == 0.0)[0]
        if len(idx) == 0 or len(jdx) == 0:
            break
        
        i = idx[rng.integers(len(idx))]
        j = jdx[rng.integers(len(jdx))]
        
        h[i] = 0.0
        h[j] = 1.0
        val = _overlap_max(h, N)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        else:
            # Revert
            h[i] = 1.0
            h[j] = 0.0
    
    return best_h, best_val


def _construct_spectral(N, time_limit=120.0):
    """
    Spectral-based initialization: use Fiedler vector of circulant graph.
    The overlap structure depends on autocorrelation. We want to minimize
    max over shifts of (sum - autocorrelation). Equivalently maximize min autocorrelation.
    Use spectral methods on the circulant graph.
    """
    deadline = time.time() + time.time() + time_limit - 5.0
    rng = np.random.default_rng(77)
    
    # Build circulant graph Laplacian with edge weights based on shifts
    # For a circulant graph on N nodes, the Laplacian eigenvalues are known.
    # Let's use a simpler approach: try to make h as "random-like" as possible
    # to minimize autocorrelation at all shifts.
    
    # Generate a sequence with low autocorrelation (like a Barker sequence or m-sequence)
    # For N=100, use an m-sequence from a primitive polynomial.
    
    # Use a simple approach: generate pseudo-random binary sequence with sum = N/2
    best_h = None
    best_val = float('inf')
    
    for trial in range(20):
        if time.time() > deadline:
            break
        h = (rng.random(N) > 0.5).astype(float)
        # Fix sum to be exactly N/2
        ones = np.sum(h)
        target = N // 2
        if ones > target:
            # Remove (ones - target) ones
            idx = np.where(h == 1.0)[0]
            to_remove = rng.choice(idx, ones - target, replace=False)
            h[to_remove] = 0.0
        elif ones < target:
            idx = np.where(h == 0.0)[0]
            to_add = rng.choice(idx, target - ones, replace=False)
            h[to_add] = 1.0
        
        val = _overlap_max(h, N)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def construct():
    """
    Construct the optimal sequence by trying multiple approaches
    and returning the best one found.
    """
    start_time = time.time()
    time_limit = 120.0
    deadline = start_time + time_limit - 5.0  # 5s safety margin
    
    candidates = []
    
    # 1. Constant sequence (baseline)
    h_const = np.full(N, 0.5)
    val_const = _overlap_max(h_const, N)
    candidates.append((val_const, h_const))
    
    # 2. Binary alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    val_alt = _overlap_max(h_alt, N)
    candidates.append((val_alt, h_alt))
    
    # 3. Gradient-based optimization
    if time.time() < deadline - 60:
        try:
            h_grad, val_grad = _construct_gradient(N, time_limit=60)
            candidates.append((val_grad, h_grad))
        except Exception:
            pass
    
    # 4. Binary local search
    if time.time() < deadline - 30:
        try:
            h_bin, val_bin = _construct_binary(N, time_limit=30)
            candidates.append((val_bin, h_bin))
        except Exception:
            pass
    
    # 5. Spectral/random binary
    if time.time() < deadline - 10:
        try:
            h_spec, val_spec = _construct_spectral(N, time_limit=10)
            candidates.append((val_spec, h_spec))
        except Exception:
            pass
    
    # Return the best candidate
    candidates.sort(key=lambda x: x[0])
    best_val, best_h = candidates[0]
    
    # Ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    # Verify sum
    s = best_h.sum()
    if abs(s - N / 2.0) > 1e-6:
        # Rescale or revert to constant
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
