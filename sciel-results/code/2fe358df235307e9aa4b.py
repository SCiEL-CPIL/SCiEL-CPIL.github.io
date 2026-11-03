"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Minimize max_s ∫ h(x)(1-h(x+s)) dx over shifts s.
    Uses smooth-max (log-sum-exp) formulation with SLSQP optimization.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    def _overlap_values(h):
        """Compute overlap values for all shifts."""
        n = len(h)
        overlaps = np.zeros(n)
        for k in range(n):
            h_shifted = np.roll(h, -k)
            overlaps[k] = np.sum(h * (1 - h_shifted)) / n
        return overlaps
    
    def _score(h):
        """Compute the maximum overlap score."""
        overlaps = _overlap_values(h)
        return np.max(overlaps)
    
    def _smooth_max_value(overlaps, beta):
        """Smooth max using log-sum-exp."""
        max_ov = np.max(overlaps)
        if beta * max_ov < 500:  # Avoid overflow
            return max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
        else:
            return max_ov
    
    def _smooth_max_grad(overlaps, beta):
        """Gradient of smooth max."""
        max_ov = np.max(overlaps)
        if beta * max_ov < 500:
            exps = np.exp(beta * (overlaps - max_ov))
            weights = exps / np.sum(exps)
            return weights
        else:
            # Return one-hot at argmax
            grad = np.zeros_like(overlaps)
            grad[np.argmax(overlaps)] = 1.0
            return grad
    
    def _fix_sum(h):
        """Project h onto the constraint sum(h) = N/2."""
        h = np.clip(h, 0, 1)
        current_sum = np.sum(h)
        target_sum = N / 2
        diff = target_sum - current_sum
        
        if abs(diff) < 1e-10:
            return h
        
        if diff > 0:
            # Need to increase sum: increase values where h < 1
            candidates = np.where(h < 1.0)[0]
            if len(candidates) == 0:
                return h
            # Distribute diff among candidates
            room = 1.0 - h[candidates]
            total_room = np.sum(room)
            if total_room > 0:
                h[candidates] += diff * room / total_room
        else:
            # Need to decrease sum: decrease values where h > 0
            candidates = np.where(h > 0.0)[0]
            if len(candidates) == 0:
                return h
            h[candidates] += diff * h[candidates] / np.sum(h[candidates])
        
        return np.clip(h, 0, 1)
    
    def _epigraph_objective(x):
        """Objective: maximize t subject to smooth_max(overlaps) <= t."""
        h = _fix_sum(x)
        overlaps = _overlap_values(h)
        # We want to minimize the smooth max, so return it
        beta = 50.0
        return _smooth_max_value(overlaps, beta)
    
    def _epigraph_grad(x):
        """Gradient of the epigraph objective."""
        h = _fix_sum(x)
        overlaps = _overlap_values(h)
        beta = 50.0
        weights = _smooth_max_grad(overlaps, beta)
        
        # Gradient with respect to h
        grad_h = np.zeros(N)
        for k in range(N):
            h_shifted = np.roll(h, -k)
            # d/dh[i] of sum(h[j]*(1-h_shifted[j]))/n
            # = (1 - h_shifted[i])/n - h[i]*(1/n)
            # Actually: overlap_k = sum_j h[j] * (1 - h[(j+k)%n]) / n
            # d(overlap_k)/d(h[i]) = (1 - h[(i+k)%n])/n - h[i] * (1/n) if h[(i+k)%n] = h_shifted[i]
            # Wait, h_shifted[i] = h[(i+k)%n]
            # So d(overlap_k)/d(h[i]) = (1 - h[(i+k)%n])/n - h[i]/n
            # = (1 - h_shifted[i] - h[i]) / n
            
            # But we need to be more careful. Let me recompute:
            # overlap_k = (1/n) * sum_j h[j] * (1 - h[(j+k)%n])
            # d(overlap_k)/d(h[i]) = (1/n) * [ (1 - h[(i+k)%n]) + sum_{j: j+k=i} h[j] * (-1) ]
            # The second term: j+k=i means j=i-k, so h[i-k] * (-1)
            # So d(overlap_k)/d(h[i]) = (1/n) * [1 - h[(i+k)%n] - h[(i-k)%n]]
            
            # Actually this is getting complex. Let me use a simpler approach:
            # Just compute the gradient numerically or use the chain rule properly.
            
            # For now, use a simple approximation
            pass
        
        # Use numerical gradient as fallback
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = _epigraph_objective(x_plus)
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = _epigraph_objective(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad
    
    # Start with a reasonable initial guess
    h0 = np.full(N, 0.5)
    # Add some structure: alternating pattern
    h0[1::2] = 0.8
    h0[0::2] = 0.2
    h0 = _fix_sum(h0)
    
    best_h = h0.copy()
    best_score = _score(best_h)
    
    # Try multiple initializations and optimization runs
    initializations = []
    
    # 1. Constant
    initializations.append(np.full(N, 0.5))
    
    # 2. Alternating
    alt = np.zeros(N)
    alt[1::2] = 1.0
    initializations.append(alt)
    
    # 3. Random with sum constraint
    for seed in range(5):
        if time.time() > deadline - 30.0:
            break
        np.random.seed(seed)
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _fix_sum(h_rand)
        initializations.append(h_rand)
    
    # 4. Quadratic residue pattern
    def quadratic_residue_pattern(p):
        qr_set = set()
        for a in range(1, p):
            if pow(a, (p-1)//2, p) == 1:
                qr_set.add(a)
        seq = np.zeros(p)
        for i in range(p):
            if i in qr_set:
                seq[i] = 1.0
        return seq
    
    for p in [5, 7, 11, 13, 17]:
        if time.time() > deadline - 30.0:
            break
        seq = quadratic_residue_pattern(p)
        repeats = (N + p - 1) // p
        tiled = np.tile(seq, repeats)[:N]
        # Smooth it a bit
        tiled = 0.5 + 0.5 * (tiled - 0.5)
        tiled = _fix_sum(tiled)
        initializations.append(tiled)
    
    # Optimize from each initialization
    for init_idx, h_init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_init = _fix_sum(h_init)
        
        # Use SLSQP with smooth max
        def objective(x):
            return _epigraph_objective(x)
        
        def gradient(x):
            return _epigraph_grad(x)
        
        # Constraints: sum(x) = N/2, 0 <= x[i] <= 1
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2, 
             'jac': lambda x: np.ones(N)}
        ]
        bounds = [(0, 1)] * N
        
        try:
            result = minimize(
                objective, h_init,
                method='SLSQP',
                jac=gradient,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8, 'disp': False}
            )
            
            if result.success or result.fun < best_score:
                h_opt = _fix_sum(result.x)
                score = _score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception as e:
            pass
    
    # Polish: use higher beta for more precise max
    if time.time() < deadline - 10.0:
        def objective_polish(x):
            h = _fix_sum(x)
            overlaps = _overlap_values(h)
            beta = 200.0
            return _smooth_max_value(overlaps, beta)
        
        def gradient_polish(x):
            eps = 1e-7
            grad = np.zeros(N)
            for i in range(N):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = objective_polish(x_plus)
                x_minus = x.copy()
                x_minus[i] -= eps
                f_minus = objective_polish(x_minus)
                grad[i] = (f_plus - f_minus) / (2 * eps)
            return grad
        
        try:
            result = minimize(
                objective_polish, best_h,
                method='SLSQP',
                jac=gradient_polish,
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2,
                             'jac': lambda x: np.ones(N)}],
                options={'maxiter': 50, 'ftol': 1e-10, 'disp': False}
            )
            
            if result.fun < best_score:
                h_opt = _fix_sum(result.x)
                score = _score(h_opt)
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final: ensure sum is exactly N/2 and values in [0,1]
    final = _fix_sum(best_h)
    final = np.clip(final, 0, 1)
    
    # Verify sum
    if not np.isclose(np.sum(final), N/2, atol=1e-10):
        diff = N/2 - np.sum(final)
        if diff > 0:
            candidates = np.where(final < 1.0)[0]
            room = 1.0 - final[candidates]
            if np.sum(room) > 0:
                final[candidates] += diff * room / np.sum(room)
        else:
            candidates = np.where(final > 0.0)[0]
            if np.sum(final[candidates]) > 0:
                final[candidates] += diff * final[candidates] / np.sum(final[candidates])
        final = np.clip(final, 0, 1)
    
    return final
# EVOLVE-BLOCK-END
