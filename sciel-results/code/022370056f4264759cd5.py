import numpy as np
import time
from scipy.optimize import linprog

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution peak
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    # 1. Initial candidates to seed the LP refinement
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Single spikes
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        candidates.append(h)
        
    # Pairs of spikes
    for i in range(0, N, 2):
        for j in range(i+1, N, 2):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            candidates.append(h)
            
    # Random sparse sets (Sidon-like intuition: spread out peaks to reduce peak autoconv)
    np.random.seed(42)
    for _ in range(50):
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        candidates.append(h)
        
    # Gaussian-like
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(2, N/2, 5):
            h = np.exp(-0.5 * ((np.arange(N) - c) / w)**2)
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h)

    # 2. Linear Programming Refinement
    # Problem: Maximize R(h) = L2^2 / (L1 * MaxConv)
    # Strategy: 
    #   Fix L2^2 = 1.
    #   Minimize MaxConv = max_k (h*h)[k].
    #   This is a Linear Program: min t subject to (h*h)[k] <= t for all k, sum(h^2)=1, h>=0.
    #   Wait, sum(h^2)=1 is non-linear.
    #   However, if we assume the support is fixed and we only optimize weights, 
    #   the problem is still non-linear due to L2 constraint.
    #   
    #   Alternative LP formulation:
    #   We want to maximize L2^2 / MaxConv. 
    #   Let's fix MaxConv = 1. Maximize L2^2.
    #   Subject to: (h*h)[k] <= 1 for all k, h >= 0.
    #   This is maximizing a convex function (L2^2) over a convex set. Not LP.
    #   
    #   Let's look at the structure again.
    #   R = L2^2 / (L1 * MaxConv).
    #   If we fix the support S (indices where h_i > 0), and let M = MaxConv,
    #   we can solve for the optimal weights on S that minimize MaxConv for a fixed L2?
    #   
    #   Actually, a simpler convex relaxation:
    #   For a fixed support S, consider the subvector h_S.
    #   Let's try to minimize MaxConv subject to L1=1 and L2^2 >= some value? No.
    #   
    #   Let's try this LP:
    #   Minimize t
    #   Subject to:
    #     sum_{i,j} h_i h_j delta_{i+j=k} <= t  for all k
    #     sum h_i = 1
    #     h_i >= 0
    #   This minimizes the peak autoconvolution for a probability distribution.
    #   The resulting h minimizes MaxConv.
    #   Does this maximize R?
    #   R = L2^2 / MaxConv (since L1=1).
    #   Minimizing MaxConv increases R, but it might also decrease L2^2 significantly.
    #   Uniform distribution: L2^2 = 1/N, MaxConv = 1/N (approx, actually 1/N for large N if spread).
    #   Actually for uniform h=1/N, conv peak is 1/N. R = (1/N)/(1/N) = 1.
    #   Spike: L2^2 = 1, MaxConv = 1. R = 1.
    #   
    #   The LP min MaxConv subject to L1=1 will likely find a "flat" distribution.
    #   Let's run this LP for various subsets or the whole space.
    #   Variables: h_0...h_{N-1}, t.
    #   Constraints:
    #     A_eq @ h = b_eq  (sum h = 1)
    #     A_ub @ [h, t] <= b_ub (conv constraints)
    #     
    #   Conv constraint for lag k: sum_{i} h_i h_{i+k} <= t.
    #   This is quadratic! LP requires linear constraints.
    #   (h*h)[k] is quadratic in h.
    #   
    #   CORRECTION: The directive says "minimizing the peak autoconvolution, which is a linear function of the height vector".
    #   This is ONLY true if one argument of the convolution is fixed.
    #   If we fix the support and perhaps the relative weights? No.
    #   
    #   However, if we fix the vector h in one slot of the convolution? No, autoconvolution is h*h.
    #   
    #   Maybe the directive implies: Fix the L2 norm, and the support. Then optimize weights?
    #   Still quadratic.
    #   
    #   Let's re-read: "minimizing the peak autoconvolution, which is a linear function of the height vector".
    #   This statement is mathematically false for autoconvolution h*h.
    #   UNLESS... the "heights" are not the variables, but something else?
    #   Or maybe it means: For a fixed vector v, the function h -> v*h is linear.
    #   If we alternate: Fix h, optimize v? No, it's autoconvolution.
    #   
    #   Perhaps the strategy is:
    #   1. Start with an initial h.
    #   2. Compute the "effective" peak location k*.
    #   3. The constraint (h*h)[k*] <= t is quadratic.
    #   
    #   Let's try a different convex problem:
    #   Maximize L2^2 subject to (h*h)[k] <= 1 for all k, and sum h = 1.
    #   This is maximizing a convex function.
    #   
    #   Let's try: Minimize MaxConv subject to L2^2 = 1.
    #   This is minimizing a convex function over a non-convex set (sphere).
    #   
    #   Okay, let's look at the "linear function" claim again.
    #   Is it possible the evaluator's "autoconvolution" is defined differently?
    #   No, standard autoconvolution.
    #   
    #   Maybe the "linear function" refers to the fact that if we fix the *other* vector in a cross-correlation, it's linear.
    #   But it's autoconvolution.
    #   
    #   Let's ignore the "linear" claim if it's mathematically impossible and use a method that IS convex/efficient.
    #   
    #   Method: Coordinate Descent / Gradient Descent on the log-likelihood or similar?
    #   Or: Use the fact that for a fixed support, the optimal weights to maximize L2^2/MaxConv 
    #   can be found by solving a generalized eigenvalue problem or similar?
    #   
    #   Let's try a simple, robust heuristic that exploits the structure:
    #   The ratio R is maximized when the energy is concentrated in a way that minimizes the peak of the autocorrelation.
    #   "Sidon sets" or "Bh sequences" have low additive energy, which correlates with low peak autocorrelation.
    #   
    #   Strategy:
    #   1. Generate many random subsets of indices (supports).
    #   2. For each support, solve for the weights that maximize R.
    #      Since the support is small, we can use a dense optimizer (SLSQP) or even a grid search if very small.
    #      Or, use the "alternating" linearization:
    #      Fix weights, update support? No.
    #      
    #   Let's stick to the current program's success (SLSQP) but improve the initialization and search space.
    #   The current program uses SLSQP. The score is 1.19.
    #   Inspiration 2 scored 1.30. It likely used a better initialization or a different optimizer.
    #   
    #   Let's try **Simulated Annealing** or **Genetic Algorithm** on the discrete support + continuous weights.
    #   
    #   New Strategy:
    #   1. Represent h as (support_indices, weights).
    #   2. Use a population-based search (Genetic Algorithm) to find good supports.
    #   3. For each support, optimize weights using SLSQP (fast for small N).
    #   
    #   This is a structural change from "dense vector optimization" to "sparse support search + local weight optimization".

    def optimize_weights_on_support(support):
        if len(support) == 0:
            return np.zeros(N), 0.0
        n_supp = len(support)
        # Variables: weights w_i for i in support
        # Objective: Maximize sum(w^2) / max(conv)
        # Constraint: sum(w) = 1 (scale invariance allows this)
        
        def neg_obj(w):
            w = np.maximum(w, 0) # Ensure non-negative during optimization steps if needed, though bounds handle it
            h = np.zeros(N)
            h[support] = w
            l1 = np.sum(h)
            if l1 == 0: return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0
            return -(l2_sq / max_conv)
            
        # Initial guess: uniform on support
        w0 = np.ones(n_supp) / n_supp
        
        bounds = [(0, 1)] * n_supp
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        try:
            res = minimize(neg_obj, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success:
                w_opt = res.x
                h_opt = np.zeros(N)
                h_opt[support] = w_opt
                score = _score_raw(h_opt)
                return h_opt, score
        except Exception:
            pass
        
        # Fallback: uniform
        h_uni = np.zeros(N)
        h_uni[support] = 1.0 / n_supp
        return h_uni, _score_raw(h_uni)

    # Phase 1: Random Support Search
    num_trials = 200
    trial_size = 0
    
    for i in range(num_trials):
        if time.time() > deadline: break
        
        # Random support size between 3 and 20
        k = np.random.randint(3, 20)
        support = np.random.choice(N, k, replace=False)
        
        h_cand, score_cand = optimize_weights_on_support(support)
        
        if score_cand > best_score:
            best_score = score_cand
            best_h = h_cand
            
        # Also try a few "structured" supports:
        # Arithmetic progressions (might have high autocorr peak? or low?)
        # Random subsets of small size (1-5)
        if i % 10 == 0:
            k_small = np.random.randint(1, 6)
            support_small = np.random.choice(N, k_small, replace=False)
            h_cand, score_cand = optimize_weights_on_support(support_small)
            if score_cand > best_score:
                best_score = score_cand
                best_h = h_cand

    # Phase 2: Local Search / Perturbation of Best Support
    # Extract support of best_h (indices where h > 1e-6)
    current_support = np.where(best_h > 1e-6)[0]
    
    for _ in range(50):
        if time.time() > deadline: break
        
        # Perturb support: add, remove, or swap an index
        new_support = current_support.copy()
        op = np.random.choice(3)
        
        if op == 0 and len(new_support) < N: # Add
            available = np.setdiff1d(np.arange(N), new_support)
            if len(available) > 0:
                new_support = np.append(new_support, np.random.choice(available))
        elif op == 1 and len(new_support) > 1: # Remove
            idx_to_remove = np.random.choice(len(new_support))
            new_support = np.delete(new_support, idx_to_remove)
        elif op == 2 and len(new_support) > 0: # Swap
            idx_in = np.random.choice(len(new_support))
            available = np.setdiff1d(np.arange(N), new_support)
            if len(available) > 0:
                new_support[idx_in] = np.random.choice(available)
                # Ensure uniqueness
                if len(np.unique(new_support)) < len(new_support):
                    continue
        
        new_support = np.unique(new_support)
        
        h_cand, score_cand = optimize_weights_on_support(new_support)
        
        if score_cand > best_score:
            best_score = score_cand
            best_h = h_cand
            current_support = np.where(best_h > 1e-6)[0]

    # Phase 3: Final Polish with Full SLSQP on the found support (or full vector if dense)
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_full(h):
            l1 = np.sum(h)
            if l1 == 0: return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0
            return -(l2_sq / max_conv)
            
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(neg_obj_full, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_final = res.x
                s_final = _score_raw(h_final)
                if s_final > best_score:
                    best_score = s_final
                    best_h = h_final
        except Exception:
            pass

    return list(best_h)

# Need to import minimize
from scipy.optimize import minimize
