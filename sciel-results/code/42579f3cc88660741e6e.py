"""Optimized program for second autocorrelation inequality task."""
import numpy as np
import time
import random

N = 50
DELTA = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, divided into 50 intervals

def _compute_ratio(h):
    """Compute ||h||^2_2 / (||h||_1 * ||h*h||_inf) for step function heights h."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * DELTA
    l2_sq = np.sum(h ** 2) * DELTA
    
    if l1 < 1e-15:
        return -np.inf
    
    # Compute autoconvolution h*h
    # For step functions on equal intervals, the autoconvolution is piecewise linear
    # with breakpoints at t = k * DELTA for k = -(N-1), ..., N-1
    # The maximum occurs at one of these breakpoints
    
    d = DELTA
    max_val = 0.0
    
    # Precompute for efficiency
    # (h*h)(t) = sum_i sum_j h[i] h[j] * overlap(i, j, t)
    # where overlap is the length of intersection of [x_i, x_{i+1}) and [t-x_j-d, t-x_j)
    # x_i = -0.25 + i * d
    
    # At t = k * d:
    # overlap(i, j, k*d) = max(0, min(x_i+d, k*d - x_j) - max(x_i, k*d - x_j - d))
    # = max(0, min(-0.25+(i+1)*d, k*d+0.25-j*d) - max(-0.25+i*d, k*d+0.25-(j+1)*d))
    
    # Let's compute this more efficiently using the structure
    # For fixed k, the overlap depends on i and j
    
    # Actually, let me use a vectorized approach
    # The autoconvolution of a discrete signal can be computed via FFT
    # But we need the continuous autoconvolution of the step function
    
    # For a step function with values h[0..N-1] on intervals of width d,
    # the autoconvolution at t = k*d can be computed as:
    # (h*h)(k*d) = d * sum_m c_m where c_m are coefficients
    
    # Let me think differently. The autoconvolution (h*h)(t) for the step function
    # can be related to the discrete autocorrelation.
    
    # Actually, for step functions, (h*h)(t) is a piecewise linear function.
    # At t = 0: (h*h)(0) = sum_i h[i]^2 * d
    # At t = d: (h*h)(d) = sum_i h[i]*h[i+1] * d (for i where i+1 < N) + boundary terms
    # etc.
    
    # Let me just compute it directly but more efficiently
    # Using the fact that for each k, we can compute the sum
    
    # Precompute x_i
    # x_i = -0.25 + i * d
    
    # For efficiency, let's note that:
    # overlap(i, j, k*d) = max(0, d - |k*d - (x_i - x_j)|) if the intervals overlap properly
    # Wait, that's not quite right.
    
    # Let me just do the O(N^3) computation but with numpy vectorization
    # For each k in range(-N+1, N):
    #   Compute all overlaps at once
    
    # Actually, let me use a smarter approach.
    # The autoconvolution (h*h)(t) for a step function can be computed as:
    # (h*h)(t) = d * (h * h_discrete)(t/d) where * is discrete convolution
    # but that's not quite right either.
    
    # Let me just compute it directly with some optimization
    # For N=50, N^3 = 125000 operations, which is fast
    
    # But let me try to be smarter.
    # Note that (h*h)(k*d) = sum_i sum_j h[i] h[j] * overlap(i,j,k)
    # where overlap(i,j,k) = max(0, min(i+1, k-j) - max(i, k-j-1)) * d
    # Wait, let me re-derive.
    
    # x_i = -0.25 + i*d, so x_i - x_j = (i-j)*d
    # t = k*d
    # overlap = max(0, min(x_i+d, t-x_j) - max(x_i, t-x_j-d))
    # = max(0, min(-0.25+(i+1)*d, k*d+0.25-j*d) - max(-0.25+i*d, k*d+0.25-(j+1)*d))
    
    # Let a = -0.25 + i*d, b = -0.25 + j*d
    # left = max(a, k*d - b - d) = max(a, (k-j-1)*d + 0.25)
    # right = min(a+d, k*d - b) = min(a+d, (k-j)*d + 0.25)
    
    # overlap = max(0, right - left)
    
    # This is still O(N^2) per k, so O(N^3) total.
    # For N=50, that's 125000 * 2 = 250000 operations, which is fine.
    
    # Let me vectorize over i and j for each k
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    I, J = np.meshgrid(i_idx, j_idx, indexing='ij')
    
    # x_i = -0.25 + i*d
    # For each k, compute overlap
    
    max_val = 0.0
    for k in range(-N + 1, N):
        # left = max(-0.25 + I*d, (k - J - 1)*d + 0.25)
        # right = min(-0.25 + (I+1)*d, (k - J)*d + 0.25)
        
        left = np.maximum(-0.25 + I * d, (k - J - 1) * d + 0.25)
        right = np.minimum(-0.25 + (I + 1) * d, (k - J) * d + 0.25)
        overlap = np.maximum(0, right - left)
        
        val = np.sum(h[I] * h[J] * overlap)
        if val > max_val:
            max_val = val
    
    if max_val < 1e-15:
        return -np.inf
    
    return l2_sq / (l1 * max_val)


def _find_good_sidon_set(n_points, N):
    """Find a Sidon-like set of n_points positions in [0, N) with distinct pairwise differences."""
    # Use a greedy approach: pick positions that maximize minimum distance
    # and have distinct pairwise differences
    
    positions = []
    used_diffs = set()
    
    # Start with position 0
    positions.append(0)
    
    # Try to add positions greedily
    candidates = list(range(1, N))
    random.shuffle(candidates)
    
    for pos in candidates:
        if len(positions) >= n_points:
            break
        
        # Check if adding pos creates any duplicate difference
        valid = True
        for p in positions:
            diff = abs(pos - p)
            if diff in used_diffs:
                valid = False
                break
            used_diffs.add(diff)
        
        if valid:
            positions.append(pos)
    
    # If we didn't get enough, try a different approach
    if len(positions) < n_points:
        # Use a more systematic approach
        positions = []
        used_diffs = set()
        
        # Try positions in order of increasing "spread"
        for start in range(N):
            if len(positions) >= n_points:
                break
            # Try to build a set starting from this position
            candidate_set = [start]
            candidate_diffs = set()
            
            for pos in range(start + 1, N):
                if len(candidate_set) >= n_points:
                    break
                valid = True
                for p in candidate_set:
                    diff = abs(pos - p)
                    if diff in candidate_diffs:
                        valid = False
                        break
                    candidate_diffs.add(diff)
                
                if valid:
                    candidate_set.append(pos)
            
            if len(candidate_set) >= n_points:
                positions = candidate_set[:n_points]
                break
    
    if len(positions) < n_points:
        # Fallback: just pick evenly spaced
        positions = list(np.linspace(0, N - 1, n_points).astype(int))
    
    return positions


def _optimize_weights(positions, n_iter=5000, deadline=None):
    """Optimize weights for a given set of positions using coordinate descent."""
    n = len(positions)
    h = np.ones(n)
    
    # Normalize
    h = h / np.sum(h)
    
    # Create full height vector
    def make_full_heights(weights):
        full_h = np.zeros(N)
        for i, pos in enumerate(positions):
            full_h[pos] = weights[i]
        return full_h
    
    # Coordinate descent
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
        
        # Try to improve each weight
        for i in range(n):
            old_w = h[i]
            
            # Try a few perturbations
            best_w = old_w
            best_score = _compute_ratio(make_full_heights(h))
            
            # Try multiplicative updates
            for factor in [0.5, 0.7, 1.3, 1.5, 2.0, 0.3, 3.0]:
                new_w = old_w * factor
                if new_w < 0:
                    continue
                h[i] = new_w
                score = _compute_ratio(make_full_heights(h))
                if score > best_score:
                    best_score = score
                    best_w = new_w
            
            h[i] = best_w
    
    return make_full_heights(h), _compute_ratio(make_full_heights(h))


def _local_search(h, n_iter=2000, deadline=None):
    """Local search: try to improve by modifying individual heights."""
    best_h = h.copy()
    best_score = _compute_ratio(h)
    
    for it in range(n_iter):
        if deadline and time.time() > deadline:
            break
        
        # Pick a random position to modify
        pos = random.randint(0, N - 1)
        old_val = best_h[pos]
        
        # Try a few new values
        candidates = []
        if old_val > 0:
            candidates.extend([old_val * 0.5, old_val * 0.7, old_val * 1.3, old_val * 1.5, 0.0])
        else:
            candidates.extend([0.1, 0.5, 1.0, 2.0])
        
        # Also try random values
        candidates.extend([random.uniform(0, 2)])
        
        best_candidate = old_val
        best_candidate_score = best_score
        
        for val in candidates:
            if val < 0:
                continue
            best_h[pos] = val
            score = _compute_ratio(best_h)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = val
        
        best_h[pos] = best_candidate
        best_score = best_candidate_score
    
    return best_h, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    random.seed(42)
    np.random.seed(42)
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Strategy: Try various Sidon-like sets and optimize their weights
    
    # Try different numbers of nonzero positions
    for n_points in [5, 8, 10, 12, 15, 20, 25, 30]:
        if time.time() > deadline - 20:
            break
        
        # Find a good Sidon-like set
        positions = _find_good_sidon_set(n_points, N)
        
        if len(positions) < 3:
            continue
        
        # Optimize weights for this set
        h_opt, score_opt = _optimize_weights(positions, n_iter=2000, deadline=deadline - 10)
        
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()
        
        # Local search from the optimized weights
        if time.time() < deadline - 15:
            h_ls, score_ls = _local_search(h_opt, n_iter=1000, deadline=deadline - 10)
            if score_ls > best_score:
                best_score = score_ls
                best_h = h_ls.copy()
    
    # Also try some structured patterns
    patterns = []
    
    # Uniform
    patterns.append(np.ones(N))
    
    # Sparse: only a few nonzero, well-separated
    for k in [5, 8, 10, 12, 15]:
        h = np.zeros(N)
        # Use positions that are well-separated
        idx = np.linspace(0, N - 1, k).astype(int)
        h[idx] = 1.0
        patterns.append(h)
    
    # Geometric spacing
    for base in [2, 3]:
        h = np.zeros(N)
        pos = 0
        while pos < N:
            h[pos] = 1.0
            pos *= base
        if np.sum(h) > 0:
            patterns.append(h)
    
    # Evaluate all patterns
    for h in patterns:
        if time.time() > deadline - 5:
            break
        score = _compute_ratio(h)
        if score > best_score:
            best_score = score
            best_h = h.copy()
    
    # Final local search
    if time.time() < deadline - 5:
        h_final, score_final = _local_search(best_h, n_iter=2000, deadline=deadline - 2)
        if score_final > best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Return as list of floats
    return [float(x) for x in best_h]
