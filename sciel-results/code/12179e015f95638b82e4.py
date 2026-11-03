"""Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm times supremum norm of autoconvolution."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 50

def _autoconv(x):
    """Compute full autoconvolution (linear convolution) of x with itself."""
    return np.convolve(x, x, mode='full')

def _ratio(x):
    """Compute the ratio: ||x||_2^2 / (||x||_1 * max(autoconv(x)))"""
    x = np.asarray(x, dtype=float)
    if np.max(x) < 1e-15:
        return 0.0
    l2_sq = np.sum(x**2)
    l1 = np.sum(x)
    ac = _autoconv(x)
    max_ac = np.max(ac)
    if max_ac < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_ac)

def _neg_ratio(x):
    """Negative ratio for minimization."""
    r = _ratio(x)
    if not np.isfinite(r):
        return 1e10
    return -r

def _optimize_weights_on_support(support, max_time=None):
    """
    Given a binary support pattern, optimize the nonnegative weights on that support
    to maximize the ratio. Uses SLSQP or trust-constr with normalization.
    """
    n = len(support)
    x0 = np.ones(n)
    
    # Normalize initial
    x0 = x0 / np.sum(x0)
    
    bounds = [(0.0, 1.0)] * n
    
    def objective(v):
        x = v
        return _neg_ratio(x)
    
    def constraint_sum(v):
        return np.sum(v) - 1.0
    
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    best_x = x0.copy()
    best_val = _ratio(x0)
    
    try:
        # Try a few different initializations
        initializations = [
            np.ones(n) / n,
            np.random.dirichlet(np.ones(n)),
            np.array([1.0 if i in support else 0.0 for i in range(n)]),
        ]
        # Scale initializations to sum to 1
        inits = []
        for init in initializations:
            s = np.sum(init)
            if s > 0:
                inits.append(init / s)
        
        for init in inits:
            res = minimize(objective, init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success and _ratio(res.x) > best_val:
                best_val = _ratio(res.x)
                best_x = res.x.copy()
            elif _ratio(res.x) > best_val:
                best_val = _ratio(res.x)
                best_x = res.x.copy()
    except Exception:
        pass
    
    return best_x * support, best_val

def _generate_sidon_like_supports(num_trials=200, max_time=None):
    """
    Generate candidate support patterns based on Sidon-like properties.
    A Sidon set has the property that all pairwise sums are distinct,
    which minimizes autoconvolution collisions.
    """
    supports = []
    
    # Start with some structured patterns
    # 1. Random sparse supports
    for _ in range(num_trials):
        if max_time and time.time() > max_time:
            break
        density = np.random.uniform(0.3, 0.7)
        support = (np.random.random(N) < density).astype(int)
        if np.sum(support) >= 5:
            supports.append(support)
    
    # 2. Geometric progression-like supports
    for base in [1.2, 1.3, 1.4, 1.5, 1.6]:
        if max_time and time.time() > max_time:
            break
        idx = []
        val = 0
        while val < N:
            idx.append(int(val))
            val *= base
        support = np.zeros(N, dtype=int)
        for i in idx:
            if i < N:
                support[i] = 1
        if np.sum(support) >= 3:
            supports.append(support)
    
    # 3. Quadratic residues mod p-like patterns
    for p in [47, 53]:
        if max_time and time.time() > max_time:
            break
        support = np.zeros(N, dtype=int)
        for x in range(p):
            r = (x * x) % p
            if r < N:
                support[r] = 1
        if np.sum(support) >= 5:
            supports.append(support)
    
    # 4. Sparse random with bias toward edges
    for _ in range(50):
        if max_time and time.time() > max_time:
            break
        # Bias toward edges
        weights = np.linspace(0.1, 1.0, N)
        weights = weights / np.sum(weights)
        k = np.random.randint(8, 25)
        idx = np.random.choice(N, size=k, replace=False, p=weights)
        support = np.zeros(N, dtype=int)
        support[idx] = 1
        supports.append(support)
    
    return supports

def _optimize_with_skew(support, skew_range=(0.5, 2.0), max_time=None):
    """
    Optimize weights with a geometric skew applied to the support positions.
    This breaks symmetry and can improve the ratio.
    """
    n = len(support)
    positions = np.where(support > 0)[0]
    if len(positions) == 0:
        return None, 0.0
    
    best_x = None
    best_val = 0.0
    
    # Try different skew factors
    skew_factors = np.linspace(skew_range[0], skew_range[1], 10)
    
    for sf in skew_factors:
        if max_time and time.time() > max_time:
            break
        
        # Create skewed initial weights
        x = np.zeros(N)
        for pos in positions:
            x[pos] = sf ** (pos / max(N, 1))
        
        s = np.sum(x)
        if s > 0:
            x = x / s
        
        # Local optimization
        try:
            bounds = [(0.0, 1.0)] * N
            
            def objective(v):
                return _neg_ratio(v)
            
            def constraint_sum(v):
                return np.sum(v) - 1.0
            
            constraints = [{'type': 'eq', 'fun': constraint_sum}]
            
            res = minimize(objective, x, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                val = _ratio(res.x)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
            else:
                val = _ratio(res.x)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x, best_val

def _random_local_search(x0, n_iterations=500, max_time=None):
    """
    Perform random local search from an initial point.
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _ratio(x)
    
    for i in range(n_iterations):
        if max_time and time.time() > max_time:
            break
        
        # Random perturbation
        x_new = x.copy()
        n_perturb = np.random.randint(1, 5)
        indices = np.random.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            if x_new[idx] > 0:
                factor = np.random.uniform(0.5, 1.5)
                x_new[idx] *= factor
            else:
                x_new[idx] = np.random.uniform(0, 0.1)
        
        # Normalize
        s = np.sum(x_new)
        if s > 0:
            x_new = x_new / s
        
        val = _ratio(x_new)
        if val > best_val:
            best_val = val
            best_x = x_new.copy()
            x = x_new.copy()
        elif np.random.random() < 0.1:
            # Occasionally accept worse solutions to escape local optima
            x = x_new.copy()
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    best_result = [1.0] * N
    best_score = _ratio(np.array(best_result))
    
    # Phase 1: Generate and evaluate candidate supports
    supports = _generate_sidon_like_supports(num_trials=100, max_time=deadline)
    
    for support in supports:
        if time.time() > deadline:
            break
        
        # Optimize weights on this support
        x_opt, val = _optimize_weights_on_support(support, max_time=deadline)
        
        if x_opt is not None and val > best_score:
            best_score = val
            best_result = x_opt.tolist()
        
        # Also try skewed optimization
        x_skew, val_skew = _optimize_with_skew(support, max_time=deadline)
        
        if x_skew is not None and val_skew > best_score:
            best_score = val_skew
            best_result = x_skew.tolist()
    
    # Phase 2: Random local search from best result
    if time.time() < deadline:
        x_start = np.array(best_result)
        x_ls, val_ls = _random_local_search(x_start, n_iterations=300, max_time=deadline)
        
        if val_ls > best_score:
            best_score = val_ls
            best_result = x_ls.tolist()
    
    # Phase 3: More random initializations with local optimization
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization with some structure
            x = np.random.dirichlet(np.ones(N))
            
            # Apply random skew
            skew = np.random.uniform(0.5, 2.0)
            x = x * (skew ** (np.arange(N) / N))
            s = np.sum(x)
            if s > 0:
                x = x / s
            
            # Local optimization
            try:
                bounds = [(0.0, 1.0)] * N
                
                def objective(v):
                    return _neg_ratio(v)
                
                def constraint_sum(v):
                    return np.sum(v) - 1.0
                
                constraints = [{'type': 'eq', 'fun': constraint_sum}]
                
                res = minimize(objective, x, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 150, 'ftol': 1e-8})
                
                if res.success:
                    val = _ratio(res.x)
                    if val > best_score:
                        best_score = val
                        best_result = res.x.tolist()
            except Exception:
                pass
    
    # Final check: ensure nonnegative
    best_result = [max(0.0, v) for v in best_result]
    
    return best_result
