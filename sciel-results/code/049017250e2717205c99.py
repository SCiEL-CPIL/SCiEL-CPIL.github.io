import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def _make_policy(thresholds, n_steps, dt, T):
    class ExercisePolicy:
        def __init__(self, thresholds, n_steps, dt, T):
            self.thresholds = np.asarray(thresholds, dtype=np.float64)
            self.n_steps = n_steps
            self.dt = dt
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=np.float64)
            if time is None:
                t = 0.0
            else:
                try:
                    t = float(time)
                except Exception:
                    t = 0.0
            
            # Determine the step index j (1-based in the contract sense, 0-based in array)
            # Contract says: 12 monthly exercise dates (no time-0 exercise).
            # So time corresponds to dates 1..12.
            if self.dt > 0:
                # t is the current time. If t=dt, it's the first exercise date.
                # j = round(t / dt)
                j = int(round(t / self.dt))
            else:
                j = 1
            
            # Clamp j to valid range [1, n_steps]
            j = max(1, min(j, self.n_steps))
            
            # thresholds array is 0-indexed for steps 1..n_steps
            # So step 1 uses thresholds[0], step 12 uses thresholds[11]
            threshold = self.thresholds[j - 1]
            
            # Exercise if S <= threshold (Put)
            return cs <= threshold

    return ExercisePolicy(thresholds, n_steps, dt, T)


def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    DEADLINE = 150.0
    SAFETY = 5.0

    # Defaults
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12

    def _get(obj, *names, default=None):
        for nm in names:
            if isinstance(obj, dict):
                if nm in obj:
                    return obj[nm]
            elif hasattr(obj, nm):
                return getattr(obj, nm)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', default=T))
        n_steps = int(_get(contract, 'n_dates', 'num_dates', 'exercise_dates', default=n_steps))
    except Exception:
        pass

    if n_steps < 1:
        n_steps = 1
    
    dt = T / n_steps
    mu_log = (r - 0.5 * sigma * sigma) * dt
    vol_log = sigma * math.sqrt(dt)
    disc = math.exp(-r * dt)

    # Grid in Log-Space (x = ln(S))
    # S range: [0.3*K, 1.8*K] is sufficient for a put with K=100, T=1, sig=0.2
    # Lower bound: 30, Upper bound: 180
    S_min = max(1e-3, 0.3 * K)
    S_max = 1.8 * K
    
    n_grid = 1000
    S_grid = np.linspace(S_min, S_max, n_grid)
    x_grid = np.log(S_grid)
    
    # Payoff in log-space
    payoff = np.maximum(K - S_grid, 0.0)
    
    # Gauss-Hermite Quadrature Nodes and Weights
    # We need to integrate E[V(x + mu + vol*Z)] where Z ~ N(0,1)
    # Using GH quadrature: sum w_i * f(z_i) approximates integral f(z) * exp(-z^2) dz
    # But our density is exp(-0.5 z^2) / sqrt(2pi).
    # Standard GH is for exp(-x^2). 
    # Let Y ~ N(0,1). Y = sqrt(2) * Z_gh where Z_gh is GH node.
    # Actually, simpler: use the standard normal CDF/PPF logic or just map GH nodes.
    # np.polynomial.hermite.hermgauss(n) gives nodes x_i and weights w_i for 
    # integral_{-inf}^{inf} f(x) exp(-x^2) dx = sum w_i f(x_i)
    # We want integral f(y) phi(y) dy = integral f(y) (1/sqrt(2pi)) exp(-y^2/2) dy
    # Let y = sqrt(2) x. dy = sqrt(2) dx.
    # Integral becomes (1/sqrt(2pi)) * sqrt(2) * integral f(sqrt(2)x) exp(-x^2) dx
    # = (1/sqrt(pi)) * sum w_i f(sqrt(2) x_i)
    
    n_quad = 64
    zq, wq = np.polynomial.hermite.hermgauss(n_quad)
    zq = zq * math.sqrt(2.0)
    wq = wq / math.sqrt(math.pi)
    
    # V_next on x_grid
    V_next = payoff.copy()
    
    thresholds = np.empty(n_steps)
    
    def elapsed_ok():
        return (time.time() - t_start) < (DEADLINE - SAFETY)

    try:
        # Backward induction
        for j in range(n_steps - 1, -1, -1):
            if not elapsed_ok():
                break
                
            # Compute Continuation Value
            # For each x_i in x_grid:
            # x_next = x_i + mu_log + vol_log * zq
            # V_next(x_next) via interpolation on x_grid
            
            # Vectorized calculation
            # x_next shape: (n_grid, n_quad)
            x_next = x_grid[:, None] + mu_log + vol_log * zq[None, :]
            
            # Clip x_next to grid range for interpolation
            x_next_clipped = np.clip(x_next, x_grid[0], x_grid[-1])
            
            # Interpolate V_next
            # np.interp expects 1D, so we flatten
            V_interp = np.interp(x_next_clipped.ravel(), x_grid, V_next).reshape(x_next.shape)
            
            # Weighted sum
            cont = disc * np.sum(wq[None, :] * V_interp, axis=1)
            
            # Immediate Payoff
            immediate = payoff
            
            # Value
            V_cur = np.maximum(immediate, cont)
            
            # Find Threshold
            # Exercise if immediate >= cont
            diff = immediate - cont
            # Find the largest x where diff >= 0
            # Since payoff decreases with S (and x) and cont increases with S (and x) for a put?
            # Actually, for a put, payoff is decreasing. Continuation is also generally decreasing but less so?
            # Wait, for a put, V(S) is decreasing in S.
            # Payoff K-S. Continuation E[V_next].
            # At low S, Payoff > Cont (Exercise).
            # At high S, Payoff < Cont (Continue).
            # So diff is positive at low x, negative at high x.
            # We want the crossing point.
            
            # Find indices where diff >= 0
            pos_idx = np.where(diff >= 0)[0]
            
            if len(pos_idx) == 0:
                # Never exercise (or only at S=0)
                th_x = x_grid[0]
            elif len(pos_idx) == n_grid:
                # Always exercise
                th_x = x_grid[-1]
            else:
                last_pos = pos_idx[-1]
                # Check if it's strictly inside
                if last_pos + 1 < n_grid:
                    # Linear interpolation between last_pos and last_pos+1
                    d1 = diff[last_pos]
                    d2 = diff[last_pos + 1]
                    if d1 != d2:
                        frac = d1 / (d1 - d2)
                        th_x = x_grid[last_pos] + frac * (x_grid[last_pos + 1] - x_grid[last_pos])
                    else:
                        th_x = x_grid[last_pos]
                else:
                    th_x = x_grid[last_pos]
            
            # Convert back to S
            thresholds[j] = math.exp(th_x)
            
            # Cap threshold at K (exercise in-the-money only usually, but boundary can be slightly above K? 
            # For puts, optimal exercise boundary is always <= K. 
            # However, due to numerical errors, it might exceed K slightly.
            # Also, if S > K, payoff is 0. If cont > 0, we continue.
            # So threshold should be <= K.
            if thresholds[j] > K:
                thresholds[j] = K
            
            V_next = V_cur

    except Exception:
        # Fallback if any error occurs
        thresholds = np.zeros(n_steps)
        for j in range(n_steps):
            # Simple heuristic: threshold approaches K as maturity approaches
            rem = (n_steps - j) * dt
            thresholds[j] = K * (1.0 - math.exp(-2.5 * rem / max(dt, 1e-9)))
            thresholds[j] = min(thresholds[j], K)

    # Ensure monotonicity? 
    # For Bermudan puts, the exercise boundary is non-decreasing over time (as time passes, you are more likely to exercise, so threshold moves up towards K).
    # Wait. 
    # At t=0 (first date), you have more time value, so you require a deeper ITM position to exercise. Threshold is LOW.
    # At t=T (last date), you exercise if ITM. Threshold is K.
    # So Thresholds should be non-decreasing with j (index 0 is first date, index 11 is last).
    # My loop goes j = n_steps-1 (last) down to 0 (first).
    # So thresholds[11] is computed first (should be ~K).
    # thresholds[0] is computed last (should be lower).
    # Let's enforce monotonicity: thresholds[j] <= thresholds[j+1]
    
    for j in range(n_steps - 1):
        if thresholds[j] > thresholds[j + 1]:
            thresholds[j] = thresholds[j + 1]
            
    # Final sanity check
    thresholds = np.clip(thresholds, 0.0, K)

    return _make_policy(thresholds, n_steps, dt, T)
# EVOLVE-BLOCK-END
