<<<<<<< SEARCH
import numpy as np
import time

N = 600
DELTA = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # = 0.5/600

def _autoconv_norm_max(x):
    """Compute the normalized maximum of the autoconvolution of step function x."""
    # Discrete autoconvolution (linear)
    conv = np.convolve(x, x, mode='full')
    max_conv = np.max(conv)
    L2_sq = np.sum(x * x)
    if L2_sq <= 0:
        return 0.0
    # The continuous autoconvolution at a grid point is approx (delta^2) * conv[k]
    # The L2 norm squared of f is delta * sum(x^2)
    # Normalized max = max(f*f) / ||f||_2^2
    # But we need to be careful about what "normalized" means here.
    # Given the problem context, likely: max_t (f*f)(t) / (integral of f^2)
    # (f*f)(t) ~ delta^2 * conv[k]
    # integral f^2 ~ delta * sum(x^2)
    # ratio = delta^2 * max_conv / (delta * sum(x^2)) = delta * max_conv / sum(x^2)
    return delta * max_conv / L2_sq

def _fw_step(x, L2_target):
    """Frank-Wolfe linear minimization step for the smoothed autoconvolution."""
    # Compute gradient of (1/2) ||f*f||_inf^p / ||f||_2^4
    # For simplicity, use p=2: minimize max(autoconv)^2 / L2^4
    # This is non-smooth. Use a smooth surrogate: sum of exp(alpha * autoconv / L2^2) / L2^4
    # Or simpler: just use the subgradient of max(autoconv)/L2^2
    
    conv = np.convolve(x, x, mode='full')
    max_val = np.max(conv)
    L2_sq = np.sum(x * x)
    if L2_sq <= 0:
        return x.copy()
    
    # Subgradient of max(conv) w.r.t. x is conv at the argmax position
    # But conv[k] = sum_i x[i] x[k-i], so d conv[k]/d x[j] = x[k-j] + x[k-j+N-1] (for valid indices)
    # More precisely, if conv = x * x (linear), then d conv[k]/d x[j] = x[k-j] if k-j in [0, N-1]
    # and also x[k-j-(N-1)] if k-j-(N-1) in [0, N-1]... actually for linear conv:
    # conv[k] = sum_{i=max(0,k-N+1)}^{min(N-1,k)} x[i] x[k-i]
    # d conv[k]/d x[j] = x[k-j] + x[k-j-(N-1)] if both in range
    
    k_max = np.argmax(conv)
    # Gradient of conv[k_max] w.r.t. x
    grad_conv = np.zeros(N)
    for j in range(N):
        # x[j] appears in conv[k_max] when i=j or k-i=j
        # i=j: then k_max - j must be in [0, N-1]
        # k-i=j => i=k_max-j, must be in [0, N-1]
        if 0 <= k_max - j <= N - 1:
            grad_conv[j] += x[k_max - j]
        if 0 <= k_max - j - (N - 1) <= N - 1:
            grad_conv[j] += x[k_max - j - (N - 1)]
        # Actually simpler: conv[k] = sum_i x[i]*x[k-i], d/dx[j] = x[k-j] + x[k-j] if j != k-j
        # Let's just use the property: d conv[k]/d x[j] = x[k-j] if 0<=k-j<N else 0
        # Wait, linear conv: (x*x)[k] = sum_{i=0}^{N-1} x[i] x[k-i] where x[n]=0 for n<0 or n>=N
        # So d/dx[j] sum_i x[i]x[k-i] = sum_i (delta_{ij} x[k-i] + x[i] delta_{k-i,j})
        # = x[k-j] + sum_i x[i] delta_{k-i,j} = x[k-j] + x[k-j] if k-j in [0,N-1]
        # = 2*x[k-j] if k-j in [0,N-1] and j != k-j... no
        # Actually: d/dx[j] [x[i]*x[k-i]] for fixed i:
        # if i=j: x[k-j]
        # if k-i=j => i=k-j: x[k-j]
        # So total: x[k-j] + x[k-j] = 2*x[k-j] if j != k-j, else x[k-j] (when 2j=k)
        # This is getting complicated. Let's just use automatic differentiation approach via sparse.
    
    # Simpler approach: use the fact that for linear conv, the gradient can be computed as
    # grad[k] = x_rev[k] + x_rev_rev[k] where x_rev is reversed... 
    # Actually, let's just compute it directly but efficiently.
    # d conv[k_max]/d x[j] = x[k_max - j] + x[k_max - j + (N-1)] if valid
    # No, let's just do it the simple way for correctness
    grad_conv = np.zeros(N)
    for i in range(N):
        # x[i] contributes to conv[i + j] for each j, so d conv[k]/d x[i] includes x[k-i]
        # So d conv[k_max]/d x[i] = x[k_max - i] if 0 <= k_max - i < N
        if 0 <= k_max - i < N:
            grad_conv[i] += x[k_max - i]
        # And also from the other term: x[k_max - j] where k_max - j = i => j = k_max - i
        # Already covered
        # Wait, conv[k] = sum_j x[j] x[k-j]
        # d/dx[i] = sum_j (delta_{ji} x[k-j] + x[j] delta_{k-j,i})
        # = x[k-i] + x[k-i] (if k-i in range) = 2*x[k-i] if i != k-i
        # Hmm, let me just test with small case
    
    # Actually, I'll use a different approach: since the problem is to minimize max(autoconv)/L2^2,
    # and this is a ratio, let's just do projected gradient or a simple optimization.
    
    # For the Frank-Wolfe step, we need to minimize <grad, y> over the constraint set
    # The constraint is sum(y) = 1, y >= 0 (simplex)
    # The minimum of <grad, y> over simplex is min(grad)
    # So the FW step goes to the vertex e_{argmin(grad)}
    
    # But we need the gradient of our objective, not just conv.
    # Let's use a simpler objective: minimize max(conv) subject to sum(x)=1, x>=0
    # and then normalize.
    
    # Actually, let's try a completely different approach: 
    # Use random restarts with local search (coordinate descent)
    return x  # placeholder

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s
    
    # Strategy: Use random initialization + coordinate descent / local search
    # to minimize the normalized max autoconvolution.
    
    best_x = None
    best_score = float('inf')
    
    # First, try the uniform distribution as baseline
    x_uniform = np.ones(N) / N
    score_uniform = _autoconv_norm_max(x_uniform)
    best_x = x_uniform.copy()
    best_score = score_uniform
    
    # Try some structured initializations
    # 1. Concentrated in the middle
    x_mid = np.zeros(N)
    x_mid[N//2 - 50:N//2 + 50] = 1.0 / 100
    score_mid = _autoconv_norm_max(x_mid)
    if score_mid < best_score:
        best_score = score_mid
        best_x = x_mid.copy()
    
    # 2. Two bumps
    x_two = np.zeros(N)
    x_two[:30] = 1.0 / 60
    x_two[N-30:] = 1.0 / 60
    score_two = _autoconv_norm_max(x_two)
    if score_two < best_score:
        best_score = score_two
        best_x = x_two.copy()
    
    # 3. Exponential decay from center
    centers = np.linspace(0, N-1, N)
    exp_x = np.exp(-((centers - N/2)**2) / (2 * (N/4)**2))
    exp_x /= np.sum(exp_x)
    score_exp = _autoconv_norm_max(exp_x)
    if score_exp < best_score:
        best_score = score_exp
        best_x = exp_x.copy()
    
    # Now do local search / coordinate descent
    def _local_search(x, max_iters=2000, time_limit=None):
        x = x.copy()
        current_score = _autoconv_norm_max(x)
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            # Try moving mass from one coordinate to another
            # Choose a random pair (i, j) and try to shift mass
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            # Try shifting a small amount from i to j and vice versa
            # Also try zeroing out a coordinate and redistributing
            # Simple: try setting x[i] = 0 and redistributing its mass
            orig_i = x[i]
            if orig_i < 1e-10:
                continue
            
            # Try moving all mass from i to j
            x[i] = 0
            x[j] += orig_i
            new_score = _autoconv_norm_max(x)
            if new_score < current_score - 1e-12:
                current_score = new_score
            else:
                x[i] = orig_i
                x[j] -= orig_i
            
            # Also try small perturbations
            for delta_frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                amount = orig_i * delta_frac
                x[i] = orig_i - amount
                x[j] = x[j] + amount
                new_score = _autoconv_norm_max(x)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    orig_i = x[i]
                else:
                    x[i] = orig_i
                    x[j] = x[j] - amount
            
            # Try zeroing and spreading
            if it % 10 == 0:
                # Try to set x[i] to 0 and spread mass evenly
                orig_i = x[i]
                if orig_i > 1e-10:
                    x[i] = 0
                    x += orig_i / N
                    new_score = _autoconv_norm_max(x)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                    else:
                        x -= orig_i / N
                        x[i] = orig_i
        
        return x, current_score
    
    # Do local search from each good initialization
    inits = [best_x.copy()]
    
    for init in inits:
        if time.time() > deadline:
            break
        improved_x, improved_score = _local_search(init, max_iters=5000, time_limit=deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_x = improved_x
    
    # Random restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization on simplex
        x_rand = np.random.dirichlet(np.ones(N) * 0.5)
        score_rand = _autoconv_norm_max(x_rand)
        
        if score_rand < best_score:
            best_score = score_rand
            best_x = x_rand.copy()
        
        # Local search from this initialization
        if time.time() > deadline:
            break
        improved_x, improved_score = _local_search(x_rand, max_iters=3000, time_limit=deadline)
        if improved_score < best_score:
            best_score = improved_score
            best_x = improved_x
    
    # Final check
    if best_x is None:
        return [1.0] * N
    
    # Ensure non-negative and sum to 1
    best_x = np.maximum(best_x, 0)
    s = np.sum(best_x)
    if s > 0:
        best_x /= s
    else:
        best_x = np.ones(N) / N
    
    return list(best_x)
=======
import numpy as np
import time

N = 600
DELTA = (1.0 / 4.0 - (-1.0 / 4.0)) / N  # = 0.5/600

def _autoconv_norm_max(x):
    """Compute the normalized maximum of the autoconvolution of step function x."""
    conv = np.convolve(x, x, mode='full')
    max_conv = np.max(conv)
    L2_sq = np.sum(x * x)
    if L2_sq <= 0:
        return 0.0
    return DELTA * max_conv / L2_sq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # Strategy: Multi-start Simulated Annealing
    # Objective: Minimize max(autoconv) / L2^2
    # Constraint: x >= 0, sum(x) = 1 (simplex)
    
    # We work with the simplex constraint.
    # Perturbation: Move mass between two coordinates to maintain simplex strictly.
    
    def _simplex_project(v):
        """Project vector v onto the probability simplex (sum=1, v>=0)."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        w = np.maximum(v - theta, 0)
        return w

    # Initial best: Uniform
    x_best = np.ones(N) / N
    score_best = _autoconv_norm_max(x_best)
    
    # Try a few structured starts
    starts = []
    starts.append(x_best.copy())
    
    # Gaussian centered
    g = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/8)**2))
    g /= np.sum(g)
    starts.append(g)
    
    # Two bumps
    t = np.zeros(N)
    t[:N//4] = 1.0
    t[3*N//4:] = 1.0
    t /= np.sum(t)
    starts.append(t)
    
    # Random Dirichlet
    for _ in range(5):
        r = np.random.dirichlet(np.ones(N) * 0.5)
        starts.append(r)

    # Simulated Annealing
    T_start = 1e-4
    T_end = 1e-10
    alpha = 0.995
    
    x_current = starts[0].copy()
    score_current = _autoconv_norm_max(x_current)
    
    if score_current < score_best:
        score_best = score_current
        x_best = x_current.copy()
        
    T = T_start
    
    # We will iterate until time runs out
    while time.time() < deadline:
        # Choose a random pair of indices to move mass between
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        # Determine amount to move
        if x_current[i] < 1e-12:
            continue
            
        # Try moving a random fraction f of x[i] to x[j]
        f = np.random.rand()
        eps = x_current[i] * f
        
        x_new = x_current.copy()
        x_new[i] -= eps
        x_new[j] += eps
        
        score_new = _autoconv_norm_max(x_new)
        
        delta_score = score_new - score_current
        
        if delta_score < 0 or np.random.rand() < np.exp(-delta_score / T):
            x_current = x_new
            score_current = score_new
            
            if score_current < score_best:
                score_best = score_current
                x_best = x_current.copy()
        else:
            pass
            
        # Cool down
        T *= alpha
        
        # If T is too low, restart with a new start or reheat
        if T < T_end:
            # Pick a new random start
            x_current = np.random.dirichlet(np.ones(N) * 0.5)
            score_current = _autoconv_norm_max(x_current)
            
            if score_current < score_best:
                score_best = score_current
                x_best = x_current.copy()
                
            T = T_start

    # Final cleanup
    x_best = np.maximum(x_best, 0)
    s = np.sum(x_best)
    if s > 0:
        x_best /= s
    else:
        x_best = np.ones(N) / N
        
    return list(x_best)
>>>>>>> REPLACE
