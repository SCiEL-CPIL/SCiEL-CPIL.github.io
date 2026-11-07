import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    """
    Computes the objective value for a given height vector h.
    Returns (value, g, m, w, y) to allow reuse in gradient calculation if needed, 
    but primarily returns the scalar value for optimization.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    h_sum = np.sum(h)
    if h_sum == 0:
        return 0.0, None
    
    # We work on the simplex for stability, but the value is scale-invariant 
    # in terms of shape optimization if we consider the ratio properly.
    # Actually, let's verify scale invariance again carefully.
    # g = conv(h, h)
    # l2 ~ sum y^2 ~ (h^2)^2 = h^4? No.
    # h is heights. g_k = sum h_i h_{k-i}.
    # If h -> c h, g -> c^2 g.
    # y -> c^2 y.
    # l2 = sum (w/3) (y^2 + yy' + y'^2) -> c^4 l2.
    # norm_1 = sum |g| / (m+1) -> c^2 norm_1.
    # norm_inf = max |g| -> c^2 norm_inf.
    # value = l2 / (norm_1 * norm_inf) -> c^4 / (c^2 * c^2) = 1.
    # YES, it is scale invariant.
    # So we can fix sum(h) = 1 to reduce the dimensionality to 49.
    
    # Normalize to sum=1 for optimization stability
    h_norm = h / h_sum
    g = np.convolve(h_norm, h_norm, mode='full')
    m = len(g)
    
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    S = np.sum(terms)
    l2 = (w / 3.0) * S
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 * norm_inf == 0:
        return 0.0, None
        
    value = l2 / (norm_1 * norm_inf)
    return value, (g, m, w, y, h_norm)

def _objective_and_grad(h):
    """
    Objective function for minimization (we maximize value, so minimize -value).
    Uses finite differences for gradient if analytic is too complex, 
    but let's try to implement a fast finite difference or analytic gradient.
    Given the complexity of the denominator (max and sum), analytic gradient is tricky.
    However, for L-BFGS-B, providing a good gradient is crucial.
    Let's use a central difference gradient with a small step, computed efficiently.
    To speed it up, we can compute the value for h + eps*e_i and h - eps*e_i.
    """
    val, _ = _compute_value(h)
    
    # If value is 0, gradient is undefined/zero
    if val == 0:
        return -val, np.zeros_like(h)
        
    grad = np.zeros_like(h)
    eps = 1e-6
    
    # To save time, we can use a smaller step or fewer dimensions? 
    # No, L-BFGS-B needs the full gradient.
    # 50 evaluations per gradient step.
    # If we do 100 steps, 5000 evals. Fast enough.
    
    for i in range(N):
        h_plus = h.copy()
        h_minus = h.copy()
        h_plus[i] += eps
        h_minus[i] -= eps
        
        # Project to simplex? 
        # The optimizer works in the ambient space with bounds [0, 1].
        # But we want to respect sum=1. 
        # It is better to use a parameterization that automatically satisfies sum=1, 
        # e.g., softmax or projecting.
        # However, scipy minimize with bounds [0,1] doesn't enforce sum=1.
        # We should use a transformation.
        # Let's use the log-sum-exp trick or just project inside the eval?
        # If we project inside _compute_value, the gradient will be wrong.
        
        # Alternative: Optimize in 49 dimensions.
        # h[0..48] are free, h[49] = 1 - sum(h[0..48]).
        # This is cleaner.
        
        # For now, let's stick to a robust numerical gradient on the 50D vector 
        # but ensure _compute_value handles the normalization internally?
        # If _compute_value normalizes, then the gradient w.r.t raw h is not the 
        # gradient w.r.t the normalized shape.
        # 
        # Better approach: 
        # Define the optimizer on the simplex directly using a parameterization.
        # Or, use the fact that value is scale invariant, so we can just 
        # normalize h inside _compute_value and compute the gradient of the 
        # normalized value w.r.t the input h, then project the gradient to the tangent space.
        
        # Let's try a different tactic: 
        # Since we have time, let's use Nelder-Mead which doesn't need gradients, 
        # but it's slow in 50D.
        # L-BFGS-B is faster if we have gradients.
        
        # Let's implement the gradient of the normalized function.
        # Let u = h / sum(h).
        # f(h) = V(u).
        # grad_h f = (1/sum(h)) * (grad_u V - V * grad(sum(h))/sum(h)^2 ... )
        # This is getting messy.
        
        # Simpler: Use scipy.optimize.minimize with method='Nelder-Mead' on the 49D simplex?
        # Or 'Powell' which doesn't need gradients?
        # Powell is derivative-free and works well for medium dimensions.
        
        # Let's switch strategy in the code: Use Powell's method on the 49D simplex.
        # We will parameterize h[0:49] and set h[49] = max(0, 1 - sum(h[0:49])).
        # Bounds: h[i] in [0, 1].
        
        # For the purpose of this function, if we are using Powell, we don't need grad.
        # But if we want to try L-BFGS-B, we need grad.
        # Let's provide a finite difference gradient for the 50D case, 
        # assuming the optimizer respects the simplex via projection? 
        # No, standard scipy minimize doesn't project.
        
        # Let's use the 49D parameterization in construct().
        # This function is just for internal testing or if we switch methods.
        
        # To keep this function useful for a potential L-BFGS-B call on 50D with bounds,
        # we will compute the gradient numerically on the normalized value.
        
        val_plus, _ = _compute_value(h_plus)
        val_minus, _ = _compute_value(h_minus)
        grad[i] = (val_plus - val_minus) / (2 * eps)
        
    return -val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Generate diverse initial candidates
    # We will use 49D parameterization for the optimizer.
    # x = [x_0, ..., x_48], h_49 = 1 - sum(x)
    # Constraints: x_i >= 0, sum(x) <= 1.
    
    candidates = []
    
    # 1. Seed (uniform)
    h_uniform = np.ones(N) / N
    candidates.append(h_uniform)
    
    # 2. Gaussians
    for width in [1, 2, 5, 10]:
        for center in [10, 25, 40]:
            x = np.arange(N)
            h = np.exp(-((x - center) ** 2) / (2 * width ** 2))
            candidates.append(h)
            
    # 3. Boxcars
    for start in range(0, N, 10):
        for length in [5, 10, 20]:
            h = np.zeros(N)
            h[start:min(start+length, N)] = 1.0
            candidates.append(h)
            
    # 4. Sine waves
    for freq in [1, 2, 5]:
        x = np.linspace(0, 2*np.pi, N)
        h = np.abs(np.sin(freq * x))
        candidates.append(h)
        
    # 5. Power laws
    for p in [2, 4, 8]:
        x = np.linspace(-1, 1, N)
        h = np.abs(x**p)
        candidates.append(h)
        
    # 6. Random
    for _ in range(10):
        h = np.random.rand(N)
        candidates.append(h)
        
    best_h = None
    best_val = -np.inf
    
    # Evaluate all candidates
    for h in candidates:
        val, _ = _compute_value(h)
        if val > best_val:
            best_val = val
            best_h = h.copy()
            
    # Polish the best candidate using Powell's Method (derivative-free, robust)
    # We define a function that takes 49 parameters and returns the negative value.
    # h_49 = 1 - sum(h_0..48)
    
    def objective_49d(x_49):
        if np.any(x_49 < 0):
            return 0.0 # Penalty for invalid
        h = np.zeros(N)
        h[:49] = x_49
        h[49] = 1.0 - np.sum(x_49)
        if h[49] < 0:
            return 0.0
        val, _ = _compute_value(h)
        return -val
        
    # Initial point for optimizer
    h_init = best_h.copy()
    h_init = h_init / np.sum(h_init) # Ensure sum 1
    x_init = h_init[:49].copy()
    
    # Check time
    if time.time() > deadline - 20:
        return list(best_h)
        
    try:
        # Powell's method is good for this
        res = minimize(
            objective_49d, 
            x_init, 
            method='Powell', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.success:
            x_opt = res.x
            h_opt = np.zeros(N)
            h_opt[:49] = x_opt
            h_opt[49] = 1.0 - np.sum(x_opt)
            if np.all(h_opt >= 0) and np.isclose(np.sum(h_opt), 1.0):
                val_opt, _ = _compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    
    except Exception:
        pass
        
    # If we have time left, try a few other local optima from different starts
    # Pick top 5 initial candidates and polish them
    if time.time() < deadline - 30:
        # Sort candidates by value
        scored_candidates = []
        for h in candidates:
            val, _ = _compute_value(h)
            scored_candidates.append((val, h))
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        for val, h in scored_candidates[:5]:
            if time.time() > deadline - 10:
                break
            if val <= best_val:
                break # No point polishing worse starts if we are tight on time, 
                      # but different basins might be better. Let's try 2 more.
                # Actually, if val is close to best_val, it might be a different basin.
                # Let's just try the top 2 distinct ones.
                
            h_init = h.copy()
            h_init = h_init / np.sum(h_init)
            x_init = h_init[:49].copy()
            
            try:
                res = minimize(
                    objective_49d, 
                    x_init, 
                    method='Powell', 
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.success:
                    x_opt = res.x
                    h_opt = np.zeros(N)
                    h_opt[:49] = x_opt
                    h_opt[49] = 1.0 - np.sum(x_opt)
                    if np.all(h_opt >= 0) and np.isclose(np.sum(h_opt), 1.0):
                        val_opt, _ = _compute_value(h_opt)
                        if val_opt > best_val:
                            best_val = val_opt
                            best_h = h_opt.copy()
            except Exception:
                pass

    # Final check
    if best_h is None:
        best_h = np.ones(N) / N
        
    # Ensure non-negative and finite
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
        best_h = np.ones(N) / N
        
    return list(best_h)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
