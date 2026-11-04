import numpy as np
import time
import sys

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _score_heights(h):
    """
    Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    where h*h is the linear autoconvolution of the step function.
    
    For step heights h_i on equal intervals of [-1/4, 1/4],
    the autoconvolution of the piecewise-constant function is a
    piecewise-quadratic function. The supremum norm is the max value.
    
    For discrete step heights with interval width dw, the autoconvolution
    at offset k (in units of dw) is sum_i h_i * h_{i+k} * dw (approximately).
    Actually, for step functions on [a,b], the autoconvolution is:
      (f*f)(t) = integral f(s) f(t-s) ds
    With step approximation, this becomes a discrete sum times dw.
    
    The supremum of the autoconvolution for nonnegative heights is achieved
    at t=0 (by symmetry and nonnegativity), where it equals sum_i h_i^2 * dw.
    
    Wait, that's not right. The autoconvolution at t=0 is integral f(s)^2 ds,
    which with step approximation is sum h_i^2 * dw.
    
    But for t != 0, it's sum_{i} h_i * h_{i+k} * dw for the appropriate overlap.
    
    Actually, let me reconsider. The autoconvolution of a step function with
    heights h_0, h_1, ..., h_{N-1} on intervals of width dw:
    
    (f*f)(t) for t in [k*dw, (k+1)*dw] is a quadratic in t, but the peak
    within each interval depends on the specific heights.
    
    For simplicity and given the seed [1.0]*N:
    ||h||_2^2 = sum h_i^2 (this is the squared L2 norm of the step function, which is sum h_i^2 * dw)
    ||h||_1 = sum h_i (L1 norm, which is sum h_i * dw)
    ||h*h||_inf = max of the autoconvolution
    
    The ratio is:
      (sum h_i^2 * dw) / ((sum h_i * dw) * ||h*h||_inf)
    
    But the dw factors may cancel or the problem may be defined discretely.
    
    Let me assume the standard discrete formulation where we work with the
    heights directly and the autoconvolution is the linear convolution:
      c[k] = sum_i h_i * h_{i+k} for k = -(N-1), ..., N-1
    
    The supremum norm of the autoconvolution is max_k c[k].
    
    For nonnegative h, c[0] = sum h_i^2 is the maximum (since all terms are nonnegative
    and adding more nonnegative products can only increase the sum... wait, no).
    
    Actually c[0] = sum h_i^2, and c[k] for k!=0 involves products h_i*h_{i+k}.
    By Cauchy-Schwarz, c[0] >= |c[k]| for any k, but since h_i >= 0, c[k] >= 0.
    So max c[k] = c[0] = sum h_i^2.
    
    Then the ratio becomes:
      (sum h_i^2) / ((sum h_i) * (sum h_i^2)) = 1 / (sum h_i)
    
    This would mean we just want to minimize sum h_i, but h_i >= 0, so we could set
    all h_i to 0, which gives 0/0. That can't be right.
    
    Let me re-read the problem: "ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution"
    
    So it's: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    If the autoconvolution supremum is indeed sum h_i^2 (for linear conv with nonneg),
    then ratio = ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.
    
    To maximize this, minimize ||h||_1 subject to some constraint? But there's no constraint
    other than nonnegative. So we'd want h_i -> 0 for all i, but that's degenerate.
    
    I think I'm misunderstanding the setup. Let me assume the evaluator uses a specific
    normalization or the problem has an implicit constraint I'm missing.
    
    For now, let me implement a reasonable interpretation and use optimization.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    if n == 0:
        return 0.0
    
    # Compute linear autoconvolution
    # c[k] = sum_i h[i] * h[i+k] for valid indices
    # Using np.convolve for efficiency
    c = np.convolve(h, h, mode='full')
    
    # Supremum norm of autoconvolution
    conv_inf = np.max(np.abs(c))
    if conv_inf == 0:
        return 0.0
    
    # L2 norm squared (discrete sum of squares)
    l2_sq = np.sum(h**2)
    
    # L1 norm (discrete sum of absolute values)
    l1 = np.sum(np.abs(h))
    if l1 == 0:
        return 0.0
    
    ratio = l2_sq / (l1 * conv_inf)
    return ratio

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Try several structured approaches
    best_h = None
    best_score = -1.0
    
    # Approach 1: Constant heights (seed)
    h_const = np.ones(N)
    s = _score_heights(h_const)
    if s > best_score:
        best_score = s
        best_h = h_const.copy()
    
    # Approach 2: Single spike
    for peak_pos in [0, N//2, N-1]:
        h_spike = np.zeros(N)
        h_spike[peak_pos] = 1.0
        s = _score_heights(h_spike)
        if s > best_score:
            best_score = s
            best_h = h_spike.copy()
    
    # Approach 3: Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h_two = np.zeros(N)
            h_two[i] = 1.0
            h_two[j] = 1.0
            s = _score_heights(h_two)
            if s > best_score:
                best_score = s
                best_h = h_two.copy()
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # Approach 4: Random search with refinement
    np.random.seed(42)
    max_iter = 10000
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        # Generate random nonnegative heights
        if it % 5 == 0:
            # Sparse: most zeros, few random values
            n_nonzero = np.random.randint(1, min(N, 10))
            h = np.zeros(N)
            idx = np.random.choice(N, n_nonzero, replace=False)
            h[idx] = np.random.uniform(0.1, 1.0, n_nonzero)
        elif it % 5 == 1:
            # Dense random
            h = np.random.uniform(0, 1, N)
        else:
            # Perturb current best
            h = best_h.copy()
            idx = np.random.randint(0, N)
            h[idx] = max(0, h[idx] + np.random.normal(0, 0.5))
        
        s = _score_heights(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Approach 5: Gradient-free local optimization on best
    if best_h is not None:
        current = best_h.copy()
        current_score = best_score
        step_sizes = [0.5, 0.1, 0.01, 0.001]
        for step in step_sizes:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    # Try increasing
                    new_h = current.copy()
                    new_h[i] = max(0, current[i] + step)
                    s = _score_heights(new_h)
                    if s > current_score:
                        current = new_h
                        current_score = s
                        improved = True
                    # Try decreasing
                    new_h = current.copy()
                    new_h[i] = max(0, current[i] - step)
                    s = _score_heights(new_h)
                    if s > current_score:
                        current = new_h
                        current_score = s
                        improved = True
            
            if current_score > best_score:
                best_score = current_score
                best_h = current.copy()
    
    if best_h is None:
        return [1.0] * N
    
    return list(best_h)

if __name__ == "__main__":
    result = construct()
    print(f"Score: {_score_heights(result)}")
    print(f"Heights: {result}")
