import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log2(p))

def _compute_score(P):
    n = 10
    ys, xs = np.indices((n, n))
    
    # D = H(x - y)
    diffs = xs - ys  # range -9 to 9
    diff_counts = np.zeros(19)
    for d in range(-9, 10):
        mask = (diffs == d)
        diff_counts[d + 9] = np.sum(P[mask])
    D = _shannon_entropy(diff_counts)
    
    # M = max(H(x), H(x+y), H(x+2y))
    # H(x)
    x_counts = np.zeros(n)
    for x in range(n):
        x_counts[x] = np.sum(P[:, x])
    Hx = _shannon_entropy(x_counts)
    
    # H(x+y)
    sum_counts = np.zeros(19)
    for s in range(19):
        mask = (xs + ys == s)
        sum_counts[s] = np.sum(P[mask])
    Hs = _shannon_entropy(sum_counts)
    
    # H(x+2y)
    sum2_counts = np.zeros(28)
    for v in range(28):
        mask = (xs + 2 * ys == v)
        sum2_counts[v] = np.sum(P[mask])
    Hv = _shannon_entropy(sum2_counts)
    
    M = max(Hx, Hs, Hv)
    
    if M <= 1e-9:
        return -np.inf
        
    return D / M

def _score_from_weights(weights):
    """Compute score from a 100-length weight vector."""
    P = weights.reshape(10, 10)
    return _compute_score(P)

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # 30s margin
    
    n = 10
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    # Initialize with uniform
    score_uniform = _compute_score(best_P)
    best_score = score_uniform
    
    # Strategy: Use differential evolution on the full 100-dimensional space
    # but with a smart initialization and multiple restarts.
    # Also try a structured approach: optimize over a smaller set of parameters.
    
    # Approach 1: Direct optimization of the 100 weights using differential evolution
    # This is expensive but might find global optima.
    # We'll use a constrained version where we normalize after each step.
    
    def objective_de(w_flat):
        w = np.maximum(w_flat, 0)
        s = np.sum(w)
        if s < 1e-15:
            return 1e10
        w = w / s
        P = w.reshape(10, 10)
        score = _compute_score(P)
        if np.isnan(score) or np.isinf(score):
            return 1e10
        return -score
    
    # Try a few differential evolution runs with different seeds
    # DE is stochastic, so multiple restarts help.
    
    num_de_restarts = 5
    for i in range(num_de_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(100 + i)
        
        # Initial population: mix of uniform and some structured starts
        # We'll let DE handle the initialization, but give it a good bounds
        # Since weights are non-negative and sum to 1, we can use bounds [0, 1]
        # and normalize inside the objective.
        
        bounds = [(0, 1)] * 100
        
        # Use a smaller population size for speed
        popsize = 20
        maxiter = 50
        
        try:
            res = differential_evolution(objective_de, bounds, seed=100+i, 
                                          maxiter=maxiter, popsize=popsize,
                                          tol=1e-4, polish=False, 
                                          init='sobol')
            
            w_opt = np.maximum(res.x, 0)
            s = np.sum(w_opt)
            if s > 1e-15:
                w_opt = w_opt / s
                P = w_opt.reshape(10, 10)
                score = _compute_score(P)
                
                if score > best_score:
                    best_score = score
                    best_P = P
                    
        except Exception as e:
            pass
    
    # Approach 2: Structured optimization
    # Consider that the optimal solution might have a specific structure.
    # For example, it might be concentrated on a few diagonals or lines.
    
    # Let's try to optimize over a smaller parameter space.
    # We can parameterize P as a product of marginal distributions or something similar.
    # But that might be too restrictive.
    
    # Alternative: Use L-BFGS-B on the log-weights to ensure positivity.
    # Let w = exp(theta) / sum(exp(theta))
    # Optimize theta in R^100
    
    def objective_theta(theta):
        # Softmax to get probabilities
        theta_max = np.max(theta)
        w = np.exp(theta - theta_max)
        s = np.sum(w)
        if s < 1e-15:
            return 1e10
        w = w / s
        P = w.reshape(10, 10)
        score = _compute_score(P)
        if np.isnan(score) or np.isinf(score):
            return 1e10
        return -score
    
    # Try a few L-BFGS-B runs from different starting points
    num_lbfgs_restarts = 10
    for i in range(num_lbfgs_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(200 + i)
        
        # Different starting points
        if i == 0:
            theta0 = np.zeros(100)
        elif i == 1:
            # Uniform
            theta0 = np.random.normal(0, 1, 100)
        elif i == 2:
            # Concentrated on anti-diagonal
            theta0 = np.zeros(100)
            for x in range(10):
                y = 9 - x
                theta0[y * 10 + x] = 5
        elif i == 3:
            # Concentrated on main diagonal
            theta0 = np.zeros(100)
            for x in range(10):
                theta0[x * 10 + x] = 5
        elif i == 4:
            # Random sparse
            theta0 = np.zeros(100)
            indices = np.random.choice(100, 20, replace=False)
            theta0[indices] = np.random.uniform(0, 5, 20)
        else:
            theta0 = np.random.normal(0, 2, 100)
        
        try:
            res = minimize(objective_theta, theta0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10})
            
            theta_opt = res.x
            # Convert back to weights
            theta_max = np.max(theta_opt)
            w_opt = np.exp(theta_opt - theta_max)
            s = np.sum(w_opt)
            if s > 1e-15:
                w_opt = w_opt / s
                P = w_opt.reshape(10, 10)
                score = _compute_score(P)
                
                if score > best_score:
                    best_score = score
                    best_P = P
                    
        except Exception as e:
            pass
    
    # Approach 3: Greedy support building with local optimization
    # Build a sparse support and then optimize the weights on that support.
    
    def _optimize_weights_on_support(support, w0=None):
        """Optimize weights on a given support using L-BFGS-B."""
        k = len(support)
        if k == 0:
            return np.array([])
        
        if w0 is None:
            w0 = np.ones(k) / k
        
        def obj(w):
            w = np.maximum(w, 0)
            s = np.sum(w)
            if s < 1e-15:
                return 1e10
            w = w / s
            P = np.zeros((10, 10))
            for i, (x, y) in enumerate(support):
                P[y, x] = w[i]
            score = _compute_score(P)
            if np.isnan(score) or np.isinf(score):
                return 1e10
            return -score
        
        try:
            res = minimize(obj, w0, method='L-BFGS-B', bounds=[(0, 1)] * k,
                           options={'maxiter': 500, 'ftol': 1e-10})
            w_opt = np.maximum(res.x, 0)
            s = np.sum(w_opt)
            if s > 1e-15:
                return w_opt / s
            return w0
        except:
            return w0
    
    # Try greedy support building with several restarts
    num_greedy_restarts = 3
    for i in range(num_greedy_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(300 + i)
        
        # Start with a random support of size 10-15
        all_cells = [(x, y) for y in range(10) for x in range(10)]
        start_size = 10 + i * 3
        start_indices = np.random.choice(len(all_cells), start_size, replace=False)
        support = [all_cells[j] for j in start_indices]
        weights = _optimize_weights_on_support(support)
        
        P = np.zeros((10, 10))
        for idx, (x, y) in enumerate(support):
            P[y, x] = weights[idx]
        current_score = _compute_score(P)
        
        # Try to add cells greedily
        max_size = 30
        no_improve = 0
        threshold = 1e-5
        
        while len(support) < max_size and no_improve < 5:
            best_gain = 0
            best_cell = None
            best_weights = None
            
            support_set = set(support)
            # Sample a subset of candidate cells to speed up
            candidates = [c for c in all_cells if c not in support_set]
            if len(candidates) > 50:
                candidate_indices = np.random.choice(len(candidates), 50, replace=False)
                candidates = [candidates[j] for j in candidate_indices]
            
            for cell in candidates:
                new_support = support + [cell]
                new_w0 = np.append(weights, 0.1)
                new_weights = _optimize_weights_on_support(new_support, new_w0)
                
                P_new = np.zeros((10, 10))
                for idx, (x, y) in enumerate(new_support):
                    P_new[y, x] = new_weights[idx]
                new_score = _compute_score(P_new)
                
                gain = new_score - current_score
                if gain > best_gain + threshold:
                    best_gain = gain
                    best_cell = cell
                    best_weights = new_weights
            
            if best_cell is not None:
                support.append(best_cell)
                weights = best_weights
                current_score += best_gain
                no_improve = 0
            else:
                no_improve += 1
        
        if current_score > best_score:
            best_score = current_score
            P = np.zeros((10, 10))
            for idx, (x, y) in enumerate(support):
                P[y, x] = weights[idx]
            best_P = P
    
    # Final check and normalization
    if best_P.sum() <= 1e-9 or np.any(~np.isfinite(best_P)):
        return np.ones((n, n)) / (n * n)
    
    # Clip and normalize
    best_P = np.maximum(best_P, 0)
    s = np.sum(best_P)
    if s > 1e-9:
        best_P = best_P / s
    
    return best_P
