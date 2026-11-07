import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

N = 50

# EVOLVE-BLOCK-START
def construct():
    # m is the length of g = convolve(h, h)
    # h has length N=50. g has length 2*N - 1 = 99.
    m = 2 * N - 1
    w = 1.0 / (m + 1)
    
    def get_g(h):
        return np.convolve(h, h)

    def calc_l2(g):
        # y = [0, g[0], ..., g[m-1], 0]
        # l2 = (w/3) * sum( y[i]^2 + y[i]y[i+1] + y[i+1]^2 )
        # sum y^2 = sum g^2
        # sum y[i]y[i+1] = sum g[i]g[i+1] for i=0..m-2
        # Note: y[0]=0, y[m+1]=0.
        # Terms:
        # i=0: 0^2 + 0*g0 + g0^2 = g0^2
        # i=1..m-1: g[i-1]^2 + g[i-1]g[i] + g[i]^2
        # i=m: g[m-1]^2 + g[m-1]*0 + 0^2 = g[m-1]^2
        # Total sum = 2*sum(g^2) + sum(g[:-1]*g[1:])
        
        s1 = 2.0 * np.sum(g**2)
        s2 = np.sum(g[:-1] * g[1:])
        sum_val = s1 + s2
        return (w / 3.0) * sum_val

    def objective(h):
        h = np.maximum(h, 0)
        S = np.sum(h)
        if S < 1e-10:
            return 0.0
        g = get_g(h)
        l2 = calc_l2(g)
        # We want to maximize l2 / S^2
        # So minimize -l2 / S^2
        val = l2 / (S * S)
        return -val

    def constraint_g_leq_1(h):
        g = get_g(h)
        # We need g <= 1
        # SLSQP requires fun(x) >= 0 for inequality constraints
        return 1.0 - g

    # Define constraints for SLSQP
    # SLSQP constraints are dicts
    # We have m constraints of type inequality
    cons = []
    for k in range(m):
        cons.append({
            'type': 'ineq',
            'fun': lambda h, k=k: 1.0 - get_g(h)[k]
        })
    
    # This creates many small functions, which is slow.
    # Better to use a single function returning an array?
    # SLSQP supports array-valued inequality constraints?
    # Yes, if 'fun' returns an array, it treats each element as a constraint.
    
    def constraints_array(h):
        g = get_g(h)
        return 1.0 - g

    cons = {
        'type': 'ineq',
        'fun': constraints_array
    }
    
    bounds = [(0.0, None) for _ in range(N)]
    
    # Initial guesses
    # We need to scale them so that max(g) <= 1.
    # Actually, the optimizer will handle scaling, but starting near the boundary helps.
    
    def normalize_peak(h):
        g = get_g(h)
        max_g = np.max(g)
        if max_g > 1e-10:
            # Scale h so that max(g) = 1
            # g scales with h^2, so h scales with 1/sqrt(max_g)
            h = h / np.sqrt(max_g)
        return h

    guesses = []
    
    # 1. Uniform
    h0 = np.ones(N)
    guesses.append(normalize_peak(h0))
    
    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    h1 = np.exp(-x**2 * 5)
    guesses.append(normalize_peak(h1))
    
    # 3. Single Spike
    h2 = np.zeros(N)
    h2[N//2] = 1.0
    guesses.append(normalize_peak(h2))
    
    # 4. Random
    np.random.seed(42)
    h3 = np.random.rand(N)
    guesses.append(normalize_peak(h3))
    
    # 5. Two Spikes
    h4 = np.zeros(N)
    h4[10] = 1.0
    h4[40] = 1.0
    guesses.append(normalize_peak(h4))

    best_val = -np.inf
    best_h = guesses[0]
    
    for h0 in guesses:
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                val = -res.fun
                # Check if constraints are satisfied
                g = get_g(res.x)
                if np.all(g <= 1.0 + 1e-6):
                    if val > best_val:
                        best_val = val
                        best_h = res.x
        except Exception:
            continue

    # Final calculation of value
    # V = (m+1) * l2 / (M * S^2)
    # Here M=1 (approximately), so V = (m+1) * best_val
    # Wait, best_val is l2/S^2.
    # V = (m+1) * best_val
    
    # But we should recalculate exactly to be safe, in case M < 1
    h_final = np.maximum(best_h, 0)
    S = np.sum(h_final)
    g = get_g(h_final)
    M = np.max(g)
    l2 = calc_l2(g)
    
    if S < 1e-10 or M < 1e-10:
        return [1.0/N] * N
        
    value = (m + 1) * l2 / (M * S * S)
    
    # The evaluator expects the heights.
    # The value is computed by the evaluator.
    # We just return the heights.
    return list(h_final)
# EVOLVE-BLOCK-END
