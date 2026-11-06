import numpy as np
import time
import random

def fit_exercise_policy(training_paths, contract):
    """
    Fits a Bermudan put exercise policy using a parameterized boundary model
    with a more robust optimization (simulated annealing + coordinate refinement).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Extract parameters from contract
    K = contract.get('strike', 100.0)
    r = contract.get('rate', 0.05)
    T = contract.get('maturity', 1.0)
    sigma = contract.get('volatility', 0.2)
    num_steps = 12

    # Handle training paths shape
    N, D = training_paths.shape
    if D == 13:
        S = training_paths[:, 1:]
    else:
        S = training_paths

    dt = T / num_steps
    times = np.linspace(dt, T, num_steps)
    discount_factors = np.exp(-r * times)

    # Pre-compute for speed: we'll evaluate policy many times
    # Vectorized evaluation of a boundary policy
    def evaluate_policy(B):
        """
        B: array of shape (12,) boundaries
        Returns: mean discounted payoff
        """
        active = np.ones(N, dtype=bool)
        total_cf = np.zeros(N)
        
        for k in range(num_steps):
            mask = active & (S[:, k] <= B[k])
            if np.any(mask):
                total_cf[mask] += discount_factors[k] * (K - S[mask, k])
                active[mask] = False
                if not np.any(active):
                    break
        
        return np.mean(total_cf)

    # Fast evaluation using pre-computed sorted indices per step?
    # Actually, let's try a different approach: for each step k, we need to know
    # which paths are still active and have S[:,k] <= B[k].
    # We can pre-sort paths by S[:,k] for each k to speed up threshold queries,
    # but the "active" mask changes based on previous exercises, so it's not trivial.
    
    # Let's stick with the loop but optimize it slightly.
    # One optimization: if B[k] is very high (close to K), most paths will exercise.
    # If B[k] is 0, none will.
    
    # Better idea: Use a parametric boundary that is monotonic.
    # For a put, the optimal exercise boundary is typically non-decreasing over time
    # (you're more willing to exercise as maturity approaches if the stock is low).
    # Actually, for American puts, the boundary increases with time to maturity.
    # So B[0] (earliest) <= B[1] <= ... <= B[11] (latest).
    # Let's enforce this monotonicity to reduce search space.
    
    # Parameterize B using cumulative sums of positive increments.
    # B[0] = base
    # B[k] = B[k-1] + inc[k-1]
    # Where base is in [0, K] and inc[k] is in [0, K]
    # This ensures B is non-decreasing.
    
    # But is it always optimal for a Bermudan put to have a non-decreasing boundary?
    # Yes, intuitively, as time passes, the option becomes more valuable if the stock
    # stays low, so you'd exercise at a higher price later. Wait, no.
    # For an American put, you exercise when S <= B(t). As t increases (closer to maturity),
    # B(t) typically increases. So B(t1) <= B(t2) if t1 < t2.
    # So yes, non-decreasing.
    
    # Let's use a simpler parameterization: B[k] = K * sigmoid(a + b * k)
    # Or just enforce monotonicity via sorting? No, sorting changes the mapping.
    
    # Let's try a different structural change:
    # Instead of optimizing 12 independent boundaries, use a single threshold
    # that scales with time, or use a linear boundary B(t) = a + b*t.
    # This reduces the dimension from 12 to 2.
    # B_k = a + b * t_k
    # Constraints: B_k >= 0 and B_k <= K (or maybe B_k can exceed K, but it's useless).
    # Actually, B_k should be in [0, K].
    
    # Let's try optimizing (a, b) for a linear boundary.
    # If that's not enough, we can add a quadratic term: B_k = a + b*t_k + c*t_k^2
    
    # Let's start with a linear boundary and see if it's better.
    # This is a structural change: from 12D to 2D search space.
    
    def evaluate_linear_policy(a, b):
        """
        a, b: parameters for B_k = a + b * t_k
        """
        B = a + b * times
        # Clip to [0, K]
        B = np.clip(B, 0, K)
        return evaluate_policy(B)

    # Optimization: Use Nelder-Mead or a simple grid search for (a, b).
    # Since the space is small, a fine grid search might be feasible.
    # a in [0, K], b in [-K/T, K/T]
    
    # Let's use scipy.optimize if available, or a custom implementation.
    # Let's try a coarse grid search first, then refine.
    
    # Grid search for (a, b)
    # a: 0 to K, step 0.5
    # b: -K/T to K/T, step 0.1/T
    
    best_val = -np.inf
    best_a = 0
    best_b = 0
    
    # Coarse grid
    a_vals = np.linspace(0, K, 20)
    b_vals = np.linspace(-K/T, K/T, 20)
    
    for a in a_vals:
        for b in b_vals:
            if time.time() > deadline - 10:
                break
            val = evaluate_linear_policy(a, b)
            if val > best_val:
                best_val = val
                best_a = a
                best_b = b
        if time.time() > deadline - 10:
            break
    
    # Refine around the best (a, b)
    if time.time() < deadline - 5:
        # Local search with smaller steps
        step_a = 0.1
        step_b = 0.01 / T
        
        for _ in range(50):  # Max iterations
            if time.time() > deadline - 5:
                break
            
            improved = False
            for da in [-step_a, 0, step_a]:
                for db in [-step_b, 0, step_b]:
                    na = best_a + da
                    nb = best_b + db
                    if na < 0 or na > K:
                        continue
                    # Check b bounds
                    if nb < -K/T or nb > K/T:
                        continue
                    val = evaluate_linear_policy(na, nb)
                    if val > best_val + 1e-10:
                        best_val = val
                        best_a = na
                        best_b = nb
                        improved = True
            
            if not improved:
                # Shrink step
                step_a *= 0.5
                step_b *= 0.5
                if step_a < 1e-4:
                    break

    # Final boundaries
    best_B = best_a + best_b * times
    best_B = np.clip(best_B, 0, K)

    # Create the policy object
    class BermudanPolicy:
        def __init__(self, boundaries, times, K):
            self.boundaries = boundaries
            self.times = times
            self.K = K
            self.num_steps = len(boundaries)
            self.dt = times[0]

        def exercise(self, current_state, time_val):
            """
            Returns a boolean vector indicating whether to exercise.
            """
            S = np.atleast_1d(current_state)
            
            # Find closest time index
            diffs = np.abs(self.times - time_val)
            idx = np.argmin(diffs)
            
            boundary = self.boundaries[idx]
            
            res = S <= boundary
            
            return res

    return BermudanPolicy(best_B, times, K)
