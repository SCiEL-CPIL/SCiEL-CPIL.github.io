"""Solve Erdos minimum overlap via convex optimization on autocorrelation."""
import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _autocorr(g):
    """Circular autocorrelation of g: (g*g)(k) = sum_j g[j]*g[(j+k) % N]."""
    # Use FFT for speed: autocorr = ifft(fft(g) * conj(fft(g)))
    G = np.fft.fft(g)
    ac = np.fft.ifft(G * np.conj(G)).real
    return ac

def _autocorr_grad(g):
    """Gradient of ac[k] w.r.t. g: 2 * circshift(g, -k)."""
    # ac[k] = sum_j g[j] g[j+k]; d/dg[i] = g[i+k] + g[i-k] (circular)
    # So grad ac[k] = 2 * circshift(g, -k)
    return 2.0 * np.array([np.roll(g, -k) for k in range(N)])

def construct():
    start = time.time()
    
    # Initialize g with small random values summing to 0
    rng = np.random.default_rng(42)
    g = rng.normal(0, 0.01, N)
    g -= g.mean()  # ensure sum 0
    np.clip(g, -0.5, 0.5, out=g)
    g -= g.mean()
    
    best_g = g.copy()
    best_t = -np.inf
    
    # We want to maximize t such that ac[k] >= t for all k, sum(g)=0, |g|<=0.5
    # Use projected gradient ascent on (g, t)
    # Objective: maximize t
    # Constraints: ac[k] >= t, sum(g) = 0, -0.5 <= g <= 0.5
    
    # Alternative: for a fixed t, check feasibility; but let's do direct optimization.
    # Use a barrier/augmented Lagrangian approach.
    
    # Simpler: iterate on t, and for each t, project g onto {sum=0, |g|<=0.5, ac>=t}
    # But projection onto ac>=t is hard.
    
    # Let's use scipy minimize on a smooth objective that penalizes violations.
    # Objective: -t + mu * max(0, t - min_k ac[k]) + penalty for sum(g) != 0 and |g|>0.5
    
    # Better: use a two-level approach.
    # For a given t, minimize ||g||^2 subject to ac[k] >= t, sum(g)=0, |g|<=0.5.
    # If feasible, increase t; else decrease.
    
    # Let's try a direct convex optimization with scipy.
    # Variables: g (N), t (1). Total N+1 variables.
    # Maximize t -> minimize -t
    # Constraints: t - ac[k] <= 0 for all k (N constraints)
    #               sum(g) = 0 (1 equality)
    #               -0.5 <= g[i] <= 0.5 (2N bounds)
    
    # This is a QP with quadratic constraints (ac[k] is quadratic in g).
    # Not a standard QP. Let's use a different approach.
    
    # Since ac[k] = sum_j g[j]g[j+k], the constraint ac[k] >= t is quadratic.
    # This is a QCQP. We can solve with scipy minimize using SLSQP or trust-constr.
    
    # But N=100 means 100 quadratic constraints. Let's try.
    
    x0 = np.concatenate([g, [0.0]])  # [g, t]
    
    def objective(x):
        if time.time() > DEADLINE:
            return 1e10
        g_ = x[:N]
        t_ = x[N]
        return -t_
    
    def grad_objective(x):
        g_ = x[:N]
        t_ = x[N]
        grad = np.zeros_like(x)
        grad[N] = -1.0
        return grad
    
    # Constraints: t - ac[k] <= 0 for all k
    # In scipy, constraints are dicts with 'fun' (>=0 for 'ineq') or 'fun' (==0 for 'eq')
    # Let's use SLSQP which handles nonlinear constraints.
    
    # Actually, SLSQP with 100 nonlinear constraints may be slow.
    # Let's try a different formulation.
    
    # Alternative: Use the fact that we can optimize over g only, with t = min_k ac[k].
    # Maximize min_k ac[k] subject to sum(g)=0, |g|<=0.5.
    # This is a max-min problem. We can use a smooth approximation:
    # maximize -logsumexp(-ac[k]) / N (geometric mean-like) or just use the min directly.
    
    # Let's use scipy minimize to maximize min_k ac[k] with a smoothing.
    
    def smooth_min(x, alpha=100.0):
        g_ = x[:N]
        ac = _autocorr(g_)
        # Smooth min: -1/alpha * log(sum(exp(-alpha * ac)))
        # For numerical stability, subtract max
        m = np.max(ac)
        return -1.0/alpha * np.log(np.sum(np.exp(-alpha * (ac - m)))) - m
    
    def smooth_min_grad(x, alpha=100.0):
        g_ = x[:N]
        ac = _autocorr(g_)
        m = np.max(ac)
        weights = np.exp(-alpha * (ac - m))
        weights /= np.sum(weights)
        # d/dg of smooth_min = sum_k weights[k] * d/dg (-1/alpha * log(sum exp(-alpha ac))) 
        # = sum_k weights[k] * (-1/alpha) * (1/sum) * sum_j (-alpha) exp(-alpha ac[j]) * d ac[j]/dg
        # = sum_k weights[k] * d ac[k]/dg  (since weights = exp(-alpha ac)/sum)
        # Actually: d/dg logsumexp(-alpha ac) = sum_k (exp(-alpha ac[k])/sum) * (-alpha) * d ac[k]/dg
        # So d/dg smooth_min = -1/alpha * d/dg logsumexp = sum_k weights[k] * d ac[k]/dg
        # But we want to MAXIMIZE smooth_min, so we MINIMIZE -smooth_min.
        # Gradient of -smooth_min w.r.t. g = -sum_k weights[k] * d ac[k]/dg
        grad_ac = _autocorr_grad(g_)  # shape (N, N): grad_ac[k] = d ac[k]/dg
        # sum_k weights[k] * grad_ac[k]
        result = np.tensordot(weights, grad_ac, axes=([0], [0]))
        return -result  # gradient of -smooth_min
    
    # Bounds: -0.5 to 0.5 for g
    bounds = [(-0.5, 0.5)] * N
    
    # Equality constraint: sum(g) = 0
    def eq_constraint(x):
        return np.sum(x[:N])
    
    def eq_grad(x):
        grad = np.zeros(N)
        grad[:] = 1.0
        return grad
    
    constraints = [
        {'type': 'eq', 'fun': eq_constraint, 'jac': eq_grad}
    ]
    
    x0 = g.copy()
    
    try:
        result = minimize(
            lambda x: -smooth_min(x, alpha=50.0),
            x0,
            jac=lambda x: -smooth_min_grad(x, alpha=50.0),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        if time.time() > DEADLINE:
            # Use best so far
            pass
        else:
            g_candidate = result.x
            # Project to ensure sum=0 and bounds
            g_candidate = np.clip(g_candidate, -0.5, 0.5)
            g_candidate -= g_candidate.mean()
            ac = _autocorr(g_candidate)
            t_val = np.min(ac)
            if t_val > best_t:
                best_t = t_val
                best_g = g_candidate
    except Exception as e:
        pass
    
    # If SLSQP didn't work well, try a simpler iterative approach
    if best_t < 0 or time.time() > DEADLINE - 5:
        # Fallback: use a simple iterative method
        g = best_g if best_t > 0 else g.copy()
        g = np.clip(g, -0.5, 0.5)
        g -= g.mean()
        
        # Simple gradient ascent on min_k ac[k] with projection
        for it in range(500):
            if time.time() > DEADLINE:
                break
            ac = _autocorr(g)
            k_min = np.argmin(ac)
            # Gradient of ac[k_min] w.r.t. g
            grad = _autocorr_grad(g)[k_min]
            # Ascent step
            lr = 0.01
            g = g + lr * grad
            # Project: clip and recenter
            g = np.clip(g, -0.5, 0.5)
            g -= g.mean()
            # Re-clip after recentering
            g = np.clip(g, -0.5, 0.5)
            g -= g.mean()
        
        ac = _autocorr(g)
        t_val = np.min(ac)
        if t_val > best_t:
            best_t = t_val
            best_g = g
    
    # Convert back to h
    h = 0.5 + best_g
    h = np.clip(h, 0.0, 1.0)
    
    # Ensure sum is exactly N/2
    # Current sum might be off due to clipping
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    
    # Adjust: add diff/N to all, then clip and re-adjust
    # This is a water-filling problem
    h_adj = h + diff / N
    h_adj = np.clip(h_adj, 0.0, 1.0)
    
    # Iterate to get exact sum
    for _ in range(100):
        current_sum = np.sum(h_adj)
        diff = target_sum - current_sum
        if abs(diff) < 1e-12:
            break
        h_adj = h_adj + diff / N
        h_adj = np.clip(h_adj, 0.0, 1.0)
    
    # Final check: if sum is still not exact, do a more careful adjustment
    current_sum = np.sum(h_adj)
    diff = target_sum - current_sum
    if abs(diff) > 1e-10:
        # Find indices where we can adjust
        # Add diff to one element, respecting bounds
        # Use a greedy approach
        remaining = diff
        indices = np.argsort(h_adj)  # most flexible first
        for idx in indices:
            if remaining > 0:
                space = 1.0 - h_adj[idx]
                adj = min(space, remaining)
                h_adj[idx] += adj
                remaining -= adj
            elif remaining < 0:
                space = h_adj[idx]
                adj = max(-space, remaining)
                h_adj[idx] += adj
                remaining -= adj
            if abs(remaining) < 1e-12:
                break
    
    return h_adj
