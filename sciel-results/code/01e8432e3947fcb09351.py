import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative step heights.
    
    The objective is scale-invariant: if we scale h by c, numerator scales by c^2,
    denominator by c^2, so ratio is unchanged. We can fix one scale (e.g., max(h)=1)
    and optimize the shape.
    
    Strategy:
    1. Start with several diverse initializations.
    2. Use gradient-free local search (Nelder-Mead style or simple coordinate descent)
       to refine the shape.
    3. Keep a deadline guard.
    
    The key insight: we want to maximize ||h||_2^2 / (||h||_1 * max_k (h*h)[k]).
    For nonnegative h, (h*h)[k] >= 0. The supremum of the autoconvolution is
    max_k sum_i h[i]*h[k-i] (with appropriate indexing for the discrete convolution).
    
    To maximize the ratio, we want ||h||_2^2 large relative to ||h||_1 * max_conv.
    Since ||h||_2^2 <= ||h||_1 * max(h) (by Cauchy-Schwarz or direct: sum h_i^2 <= (max h_i)(sum h_i)),
    the ratio <= max(h) / max_conv.
    
    Actually: ||h||_2^2 = sum h_i^2. ||h||_1 = sum h_i. max_conv = max_k (h*h)[k].
    
    If h is a spike: h = [0,...,1,...,0], then ||h||_2^2 = 1, ||h||_1 = 1, max_conv = 1 (at lag 0).
    Ratio = 1/1 = 1.
    
    If h is uniform: h = [1,...,1] (all ones), then ||h||_2^2 = N, ||h||_1 = N, max_conv = N (at lag 0, sum of N*1*1).
    Ratio = N/(N*N) = 1/N.
    
    So spike gives ratio 1, uniform gives 1/N. Spike is better.
    
    But wait, can we do better than 1? Let's think.
    Ratio = (sum h_i^2) / ((sum h_i) * max_k sum_i h_i h_{k-i}).
    
    Let's try a two-spike configuration: h has two peaks.
    Or maybe a smooth bump?
    
    Actually, let me reconsider. The seed program (score 1.0) uses a single spike at center.
    The score is value/seed, and seed value is the ratio for the spike.
    
    For a single spike of height H: ratio = H^2 / (H * H) = 1.
    So seed value = 1.0, and score = value/1.0 = value.
    
    So we need to find a configuration with ratio > 1.
    
    Let's think about what maximizes sum(h_i^2) / (sum(h_i) * max_conv).
    
    Note that max_conv >= max_i h_i^2 (at lag 0, the term i=0 in the sum gives h_i^2... wait, no).
    Actually, (h*h)[k] = sum_i h[i] h[k-i]. At k=0, (h*h)[0] = sum_i h[i] h[-i]. For a function supported on indices 0..N-1, we need to be careful about indexing.
    
    Let me define: h is a vector of length N. The autoconvolution (full) has length 2N-1.
    (h*h)[k] for k = 0..2N-2 is sum_{i} h[i] * h[k-i] where h[j] = 0 for j < 0 or j >= N.
    
    At k=0: (h*h)[0] = h[0]*h[0] = h[0]^2.
    At k=1: (h*h)[1] = h[0]*h[1] + h[1]*h[0] = 2*h[0]*h[1].
    ...
    At k=N-1: (h*h)[N-1] = sum_{i=0}^{N-1} h[i]*h[N-1-i].
    
    For a single spike at position p: h[p]=1, others 0.
    (h*h)[k] is nonzero only when k-p = p, i.e., k=2p. And (h*h)[2p] = h[p]*h[p] = 1.
    So max_conv = 1. Ratio = 1/(1*1) = 1.
    
    Can we get ratio > 1?
    We need sum(h_i^2) > sum(h_i) * max_conv.
    
    Since max_conv >= max_i h_i^2 (is this true? At the lag corresponding to 2i, we have h[i]^2 as a term, but there are other terms too, so max_conv >= h[i]^2 for all i, hence max_conv >= max_i h_i^2).
    
    Also sum(h_i^2) <= max_i h_i^2 * N (trivially) but more usefully sum(h_i^2) <= max_i h_i * sum(h_i) = max_i h_i * ||h||_1.
    
    So ratio <= max_i h_i / max_conv <= max_i h_i / max_i h_i^2 = 1/max_i h_i.
    
    If we scale h so that max_i h_i = 1, then ratio <= 1. And the spike achieves ratio = 1.
    
    Wait, that suggests the maximum is 1, achieved by any spike. But the task says "higher is better" and "improvement > 1.0". Maybe I'm misunderstanding the scoring.
    
    Let me re-read: "score is the lower bound on C2" and "the evaluator measures the raw value (higher is better). The search score is value/seed".
    
    So if seed value is 1.0, and we achieve value 1.0, score is 1.0. To improve, we need value > 1.0.
    
    But my analysis suggests max ratio is 1. Let me double-check.
    
    Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    For nonnegative h:
    ||h||_2^2 = sum h_i^2
    ||h||_1 = sum h_i
    ||h*h||_inf = max_k (h*h)[k]
    
    We know (h*h)[k] >= h[i]^2 for any i such that the term appears. Specifically, (h*h)[2i] includes h[i]*h[i] = h[i]^2. So ||h*h||_inf >= max_i h_i^2.
    
    Also, sum h_i^2 <= (max h_i)(sum h_i) = (max h_i)||h||_1.
    
    So ratio <= (max h_i)||h||_1 / (||h||_1 * ||h*h||_inf) = max h_i / ||h*h||_inf <= max h_i / max_i h_i^2 = 1/max h_i.
    
    If we normalize so max h_i = 1, ratio <= 1. Spike achieves 1.
    
    Hmm, but maybe the problem allows the heights to be arbitrary nonnegative, and the "value" is the ratio, and the seed achieves some value, and we need to beat it.
    
    Wait, maybe I should just run the seed and see what value it gets. The seed uses a spike of height 1.0. Ratio = 1.0.
    
    Unless... the indexing of the convolution is different, or the norms are computed differently (e.g., continuous L2 vs discrete sum).
    
    The task says "step heights on equal intervals". So f is a step function. 
    ||f||_2^2 = sum h_i^2 * dx where dx = (1/2)/50 = 1/100.
    ||f||_1 = sum h_i * dx.
    ||f*f||_inf = max of the autoconvolution of the step function.
    
    The autoconvolution of a step function is also a step function (or piecewise). The supremum norm is the max value.
    
    If we use continuous norms:
    ||f||_2^2 = dx * sum h_i^2
    ||f||_1 = dx * sum h_i
    (f*f)(x) = integral f(t) f(x-t) dt. For step functions, this is a piecewise linear function.
    ||f*f||_inf = max of this piecewise linear function.
    
    The ratio = (dx * sum h_i^2) / (dx * sum h_i * ||f*f||_inf) = (sum h_i^2) / (sum h_i * ||f*f||_inf).
    
    The dx cancels. But ||f*f||_inf for the continuous convolution is different from the discrete max_conv.
    
    For a single spike of height 1 at position p (interval [x_p, x_p+dx]):
    f(t) = 1 for t in [x_p, x_p+dx], 0 otherwise.
    (f*f)(x) = integral f(t) f(x-t) dt. This is nonzero when x-t is in [x_p, x_p+dx] and t is in [x_p, x_p+dx].
    So x is in [2*x_p, 2*x_p + 2*dx].
    (f*f)(x) = length of intersection of [x_p, x_p+dx] and [x-dx, x].
    This is a triangle function peaking at x = 2*x_p + dx with value dx.
    So ||f*f||_inf = dx.
    
    Ratio = (dx * 1) / (dx * 1 * dx) = 1/dx = 100.
    
    Ah, so the continuous interpretation gives a different answer! The seed spike gives ratio 100.
    
    So we need to maximize (sum h_i^2) / (sum h_i * ||f*f||_inf_continuous).
    
    For a uniform h = [1,...,1]:
    ||f||_2^2 = dx * N = dx * 50.
    ||f||_1 = dx * 50.
    f is a rectangular pulse of height 1, width N*dx = 1/2.
    (f*f)(x) is a triangle of height (width) = 1/2, centered at the center.
    ||f*f||_inf = 1/2.
    Ratio = (dx*50) / (dx*50 * 1/2) = 1/(1/2) = 2.
    
    So uniform gives 2, spike gives 100. Spike is much better.
    
    Can we do better than 100?
    
    For a spike of height H: ratio = (dx*H^2) / (dx*H * dx*H) = 1/dx = 100. Same.
    
    What about a two-spike configuration?
    h = [0,...,H,0,...,H,0,...] with two spikes.
    ||f||_2^2 = dx * 2*H^2.
    ||f||_1 = dx * 2*H.
    (f*f)(x) will have peaks at the self-convolution of each spike (height dx*H^2 each) and cross-terms.
    The max of ||f*f||_inf is at least dx*H^2 (from self-convolution).
    Ratio = (dx*2*H^2) / (dx*2*H * dx*H^2) = 1/(dx*H).
    
    If H=1, ratio = 1/dx = 100. Same as single spike.
    
    If we make H large, ratio decreases. So H=1 is optimal for spikes.
    
    What if we use a smooth profile?
    Let's think about what shape maximizes (sum h_i^2) / (sum h_i * max_conv_cont).
    
    Since max_conv_cont >= (max h_i) * dx * (something)... actually, the peak of the autoconvolution is at least h_i * dx * h_i = h_i^2 * dx for the self-term at the corresponding lag.
    
    So max_conv_cont >= max_i h_i^2 * dx.
    
    Ratio <= (dx * sum h_i^2) / (dx * sum h_i * max_i h_i^2 * dx) = (sum h_i^2) / (dx * sum h_i * max_i h_i^2).
    
    Let M = max_i h_i. Then sum h_i^2 <= M * sum h_i.
    Ratio <= M * sum h_i / (dx * sum h_i * M^2) = 1/(dx * M).
    
    To maximize, minimize M. But if M -> 0, the function goes to 0 and the ratio is ill-defined (0/0).
    
    Actually, the ratio is scale-invariant. Let's fix sum h_i = 1 (or any constant). Then we want to maximize sum h_i^2 / max_conv_cont.
    
    With sum h_i = 1, sum h_i^2 is maximized by a spike (sum h_i^2 = 1/N... no, if sum h_i = 1 and one h_i = 1, others 0, sum h_i^2 = 1).
    
    Hmm, I'm getting confused. Let me just implement the objective and do a local search.
    
    The key is: we need to compute the continuous autoconvolution supremum for a step function.
    
    For a step function f with heights h_i on intervals [x_i, x_{i+1}], the autoconvolution (f*f)(x) = integral f(t) f(x-t) dt.
    
    This can be computed efficiently. The result is a piecewise linear function. The supremum is the max of this piecewise linear function.
    
    Alternatively, we can compute (f*f) at a dense set of points and take the max. Since f*f is piecewise linear, its max occurs at a vertex or is constant on an interval. The vertices are at x = x_i + x_j for i,j in 0..N-1 (and possibly at the edges).
    
    Actually, the breakpoints of (f*f)(x) are at x = a + b where a is a breakpoint of f and b is a breakpoint of f (with sign considerations). Since f is a step function with breakpoints at x_0, x_1, ..., x_N, the breakpoints of f*f are at x_i + x_j and x_i - x_j for all i,j.
    
    This is O(N^2) breakpoints, which is 2500 for N=50. We can evaluate (f*f) at all these points and take the max.
    
    Let me implement this.
    
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def compute_ratio(h):
        """Compute the ratio ||f||_2^2 / (||f||_1 * ||f*f||_inf) for step heights h."""
        h = np.asarray(h, dtype=float)
        if np.any(h < 0):
            return 0.0
        if np.sum(h) == 0:
            return 0.0
        
        dx = 0.5 / N  # interval length
        x = np.linspace(-0.25, 0.25, N + 1)  # breakpoints
        
        norm2_sq = dx * np.sum(h**2)
        norm1 = dx * np.sum(h)
        
        # Compute autoconvolution supremum
        # Breakpoints of f*f are at x_i + x_j and x_i - x_j
        # But since f is supported on [x_0, x_N], f*f is supported on [2*x_0, 2*x_N]
        # The breakpoints are at x_i + x_j for i,j in 0..N (with x_i + x_j in [2*x_0, 2*x_N])
        
        # Actually, let's just evaluate f*f on a dense grid. Since f*f is piecewise linear,
        # and the slope changes at x_i + x_j, we can evaluate at all such points.
        
        # Simpler: use numpy to compute the full convolution of h with h, but that gives
        # the discrete convolution, not the continuous one.
        
        # For step functions, the continuous autoconvolution can be computed as follows:
        # (f*f)(x) = sum_i h_i * (length of intersection of [x_i, x_{i+1}] and [x - x_{i+1}, x - x_i])
        # This is piecewise linear.
        
        # Let's evaluate at a dense set of points. The function is piecewise linear with
        # breakpoints at x_i + x_j. There are O(N^2) such points.
        
        # Generate candidate x values
        x_vals = set()
        for i in range(N + 1):
            for j in range(N + 1):
                xv = x[i] + x[j]
                if 2 * x[0] - 1e-12 <= xv <= 2 * x[N] + 1e-12:
                    x_vals.add(round(xv, 12))
        
        # Also add some extra points to be safe
        x_dense = np.linspace(2 * x[0], 2 * x[N], 2000)
        x_vals.update(x_dense.tolist())
        
        x_test = np.array(sorted(x_vals))
        
        # Evaluate f*f at x_test
        # (f*f)(x) = integral f(t) f(x-t) dt
        # f(t) = h_i for t in [x_i, x_{i+1}]
        # f(x-t) = h_j for x-t in [x_j, x_{j+1}], i.e., t in [x - x_{j+1}, x - x_j]
        
        # For each x, (f*f)(x) = sum_i h_i * length([x_i, x_{i+1}] intersect [x - x_N, x - x_0])
        # But more precisely, for each i, the contribution is h_i * length([x_i, x_{i+1}] intersect [x - x_{j+1}, x - x_j] for the j such that...)
        
        # Actually, it's easier to think of it as:
        # (f*f)(x) = sum_{i,j} h_i * h_j * length([x_i, x_{i+1}] intersect [x - x_{j+1}, x - x_j])
        
        # This is O(N^2) per x value, and we have O(N^2) x values, so O(N^4) total. N=50, N^4=6.25M, which is fine.
        
        max_conv = 0.0
        for xv in x_test:
            conv_val = 0.0
            for i in range(N):
                # f(t) = h_i for t in [x_i, x_{i+1}]
                # Need f(x-t) != 0, so x-t in [x_0, x_N], i.e., t in [x - x_N, x - x_0]
                # Intersection of [x_i, x_{i+1}] and [x - x_N, x - x_0]
                lo = max(x[i], xv - x[N])
                hi = min(x[i+1], xv - x[0])
                if hi > lo:
                    # Length of intersection is hi - lo
                    # But we also need to account for the fact that f(x-t) varies within this interval
                    # Actually, f(x-t) = h_j where x-t in [x_j, x_{j+1}], so t in [x - x_{j+1}, x - x_j]
                    # The contribution from interval [x_i, x_{i+1}] is:
                    # integral_{t in [x_i, x_{i+1}]} h_i * f(x-t) dt
                    # = h_i * integral_{t in [x_i, x_{i+1}]} f(x-t) dt
                    # f(x-t) is a step function of t, with breakpoints at x - x_j for j=0..N
                    # So we need to integrate f(x-t) over [x_i, x_{i+1}]
                    
                    # This is getting complicated. Let me use a different approach.
                    pass
            
            # This approach is too slow. Let me use a simpler method.
            break
        
        # Simpler approach: use the fact that f*f for step functions can be computed
        # by convolving the heights with a triangle kernel, or by direct computation.
        
        # Actually, the simplest approach: since f is a step function, f*f is the convolution
        # of two step functions. We can compute this by:
        # 1. Create a dense representation of f
        # 2. Use scipy.signal.fftconvolve or np.convolve
        # 3. Take the max
        
        # But the dense representation needs to be fine enough to capture the piecewise linear nature.
        # Since f*f is piecewise linear with O(N^2) segments, we need at least O(N^2) points.
        
        # Let's use a grid of size M = 4*N^2 = 10000
        M = 4 * N * N
        t = np.linspace(2 * x[0], 2 * x[N], M)
        
        # Evaluate f at t - s for each s? No, that's O(M^2).
        
        # Better: use the property that (f*f)(x) = sum_i h_i * (g_i)(x) where g_i(x) = integral f(x-t) dt over [x_i, x_{i+1}]
        # g_i(x) = integral_{x_i}^{x_{i+1}} f(x-t) dt = integral_{x-x_{i+1}}^{x-x_i} f(u) du (with u = x-t)
        # = F(x - x_i) - F(x - x_{i+1}) where F is the CDF of f
        # F(y) = integral_{-inf}^{y} f(u) du = dx * sum_{j: x_j < y} h_j (approximately, for step functions)
        
        # Actually, for a step function, F(y) is piecewise linear.
        # F(y) = sum_{j=0}^{N-1} h_j * max(0, min(y, x_{j+1}) - max(x_j, -inf))
        #       = sum_{j=0}^{N-1} h_j * max(0, min(y - x_j, dx))
        
        # So g_i(x) = F(x - x_i) - F(x - x_{i+1})
        # And (f*f)(x) = sum_i h_i * (F(x - x_i) - F(x - x_{i+1}))
        
        # This is O(N^2) to compute for all x on a grid.
        
        def F(y):
            """CDF of f: F(y) = integral_{-inf}^y f(u) du"""
            # y is an array
            result = np.zeros_like(y, dtype=float)
            for j in range(N):
                # Contribution from interval [x_j, x_{j+1}]
                # integral from -inf to y of f(u) du includes h_j * length([x_j, x_{j+1}] intersect (-inf, y])
                lo = x[j]
                hi = min(x[j+1], y)
                if hi > lo:
                    result += h[j] * (hi - lo)
            return result
        
        conv_vals = np.zeros_like(t)
        for i in range(N):
            conv_vals += h[i] * (F(t - x[i]) - F(t - x[i+1]))
        
        max_conv = np.max(conv_vals)
        
        if max_conv <= 0:
            return 0.0
        
        return norm2_sq / (norm1 * max_conv)
    
    # Initializations
    candidates = []
    
    # Spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    candidates.append(h_spike.copy())
    
    # Uniform
    h_unif = np.ones(N)
    candidates.append(h_unif.copy())
    
    # Gaussian-like
    x_idx = np.arange(N)
    center = N // 2
    sigma = 5
    h_gauss = np.exp(-((x_idx - center)**2) / (2*sigma**2))
    h_gauss /= np.max(h_gauss)
    candidates.append(h_gauss.copy())
    
    # Triangle
    h_tri = np.abs(1 - np.abs(x_idx - center) / (center + 1))
    h_tri /= np.max(h_tri)
    candidates.append(h_tri.copy())
    
    # Random
    np.random.seed(42)
    for _ in range(5):
        h_rand = np.random.rand(N)
        h_rand /= np.max(h_rand)
        candidates.append(h_rand.copy())
    
    # Evaluate initial candidates
    best_h = candidates[0].copy()
    best_val = compute_ratio(best_h)
    
    for h in candidates[1:]:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    # Local search: coordinate descent with random restarts
    np.random.seed(123)
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            h = best_h.copy()
        else:
            # Perturb the best solution
            h = best_h.copy()
            h += np.random.randn(N) * 0.1 * np.max(best_h)
            h = np.maximum(h, 0)
            if np.max(h) > 0:
                h /= np.max(h)
        
        # Coordinate descent
        for _ in range(50):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                
                old_val = compute_ratio(h)
                # Try increasing and decreasing h[i]
                for delta in [0.01, 0.05, 0.1, 0.2, -0.01, -0.05, -0.1, -0.2]:
                    h_new = h.copy()
                    h_new[i] += delta * np.max(h)
                    if h_new[i] < 0:
                        h_new[i] = 0
                    new_val = compute_ratio(h_new)
                    if new_val > old_val + 1e-12:
                        h = h_new
                        old_val = new_val
                        improved = True
                        break
            
            if not improved:
                break
        
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    # Return the best heights
    return list(best_h)
# EVOLVE-BLOCK-END
