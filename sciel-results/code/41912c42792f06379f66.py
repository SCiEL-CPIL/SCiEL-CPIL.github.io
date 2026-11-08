"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def compute_max_overlap_linear(h):
    """
    Compute the maximum overlap of h with shifted (1-h) for linear shifts.
    O(s) = sum_{i=0}^{N-s-1} h[i] * (1 - h[i+s])
    """
    n = len(h)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    # We only need to check shifts up to N//2 due to symmetry? No, linear is not symmetric in the same way.
    # But O(N-s) decreases as s increases?
    # Actually, let's just compute for all s to be safe, or use FFT for circular and check if linear is intended.
    # Given the ambiguity, I'll compute circular first as it's faster and the seed implies circular.
    # If circular is optimal, linear is likely similar or worse for non-constant.
    
    # Let's assume circular for now as per the seed's logic, but use a more robust optimizer.
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(H * np.conj(O)).real
    return np.max(corr)


def objective(h):
    """
    Objective function to minimize: max overlap + penalty for out-of-bounds.
    """
    # Penalty for values outside [0, 1]
    penalty = 0.0
    if np.any(h < 0) or np.any(h > 1):
        penalty = 1000.0 * (np.sum(np.clip(h, 0, 1) != h))
    
    return compute_max_overlap_linear(h) + penalty


def construct():
    """
    Construct a sequence of step heights in [0, 1] whose sum is exactly N/2,
    minimizing the maximum overlap with shifted (1-h).
    
    Strategy: Linear Programming (LP) to minimize the maximum overlap T directly.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    target_sum = N / 2.0
    
    # Best solution found so far
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap_linear(best_h)
    
    # The overlap for shift s is:
    # O_s = sum_{i=0}^{N-s-1} h[i] * (1 - h[i+s])
    # This is quadratic. However, note that:
    # O_s = sum_{i=0}^{N-s-1} h[i] - sum_{i=0}^{N-s-1} h[i]h[i+s]
    # This is not linear in h.
    
    # Wait, the task says "overlap of h with shifted 1-h".
    # If the problem implies a linear formulation, perhaps it's not the product?
    # "Overlap" usually means the integral of the product for continuous functions, or sum of products for discrete.
    # Let's re-read carefully: "minimize the maximum overlap of h with shifted 1-h".
    # If h is a probability density (step function), overlap is integral h(x) * (1-h(x+s)) dx.
    # This is quadratic.
    
    # However, if we look at the inspirations, they all use FFT/correlation.
    # Is it possible the problem is actually linear?
    # Maybe "overlap" means something else?
    # Or maybe we can bound it?
    
    # Let's try a different structural change:
    # Instead of optimizing h directly with a non-convex quadratic objective,
    # let's try to find a sequence that minimizes the L2 norm of the correlation?
    # Or, let's use the fact that for small N, we can solve a Quadratic Program (QP) using scipy.optimize.minimize with method='SLSQP' or 'trust-constr' which handles quadratic objectives better?
    # Or, we can reformulate.
    
    # Actually, there is a known result for the Erdos problem.
    # The optimal function is often related to a threshold function or a specific pattern.
    # For N=100, the constant 0.5 gives overlap 0.25 * N / 2? No.
    # For h=0.5, O_s = sum 0.5*0.5 = 0.25 * (N-s).
    # Max is at s=0: 0.25 * N = 25.
    
    # Let's try to minimize the maximum of O_s.
    # We can use a minimax formulation:
    # min T
    # s.t. O_s <= T for all s
    #      sum h_i = N/2
    #      0 <= h_i <= 1
    
    # Since O_s is quadratic, this is a Quadratic Constrained Program (QCP).
    # scipy.optimize.minimize can handle this with 'SLSQP' if we define the objective as T and constraints as O_s - T <= 0.
    # But T is a variable. We can optimize over h and T jointly?
    # Or just minimize max(O_s) directly.
    
    # Let's try SLSQP with the max overlap as the objective.
    # The gradient of max(O_s) is the gradient of the active O_s.
    
    def max_overlap_func(h):
        # Compute all O_s
        # O_s = sum_{i=0}^{N-s-1} h[i] * (1 - h[i+s])
        # This can be computed via FFT as before, but we need the real part of the correlation.
        # corr[s] = sum_{i=0}^{N-1} h[i] * (1-h)[(i+s)%N]
        # For linear, it's a bit different. The seed uses circular. Let's stick to circular for speed and consistency with seed's FFT approach.
        H = np.fft.fft(h)
        O = np.fft.fft(1.0 - h)
        corr = np.fft.ifft(H * np.conj(O)).real
        return np.max(corr)
    
    # SLSQP might be slow for N=100 if we do many iterations.
    # Let's try a different approach:
    # Discretize the search space? No, N=100 is too high for exhaustive.
    
    # What if we use the fact that the optimal solution is often a "bang-bang" control (0 or 1)?
    # If h is binary (0 or 1), sum h_i = N/2 = 50.
    # Then O_s = sum h_i (1-h_{i+s}) = sum h_i - sum h_i h_{i+s}.
    # Since h_i is 0 or 1, h_i h_{i+s} is 1 only if both are 1.
    # So O_s = (number of 1s in window) - (number of pairs of 1s at distance s).
    # This is still complex.
    
    # Let's go back to the LP idea. Is it possible the overlap is defined differently?
    # "overlap of h with shifted 1-h"
    # Maybe it means the area where both are non-zero?
    # If h is a step function, and 1-h is its complement.
    # The "overlap" might be the measure of the set where h(x) > 0 AND (1-h)(x+s) > 0?
    # If h is 0/1, this is the intersection of the support of h and the support of 1-h shifted.
    # Support of h = {x | h(x)=1}. Support of 1-h = {x | h(x)=0}.
    # Intersection of {h=1} and {h(x+s)=0}.
    # This is exactly the number of 1s in h that are followed by a 0 at shift s.
    # This is linear in h if we fix the pattern? No, it's still a count of pairs.
    
    # Let's assume the standard definition (integral of product) and use a robust QP solver.
    # scipy.optimize.minimize with method='SLSQP' is a general purpose solver.
    
    def objective_slsqp(h):
        return max_overlap_func(h)
    
    def constraint_sum(h):
        return np.sum(h) - target_sum
    
    # SLSQP requires a vector of constraints.
    # We have one equality constraint.
    
    # We also need to ensure h is in [0,1]. SLSQP supports bounds.
    
    # Let's try a few random starts with SLSQP.
    
    np.random.seed(42)
    
    # Initial guess: 0.5 everywhere
    h0 = np.full(N, 0.5)
    
    # Bounds for h
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Constraint: sum(h) = N/2
    # In SLSQP, constraints are dicts with 'type', 'fun', 'jac'.
    # fun returns 0 for equality.
    
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - target_sum, 'jac': lambda h: np.ones(N)})
    
    # Run SLSQP
    # SLSQP can be slow. Let's limit iterations.
    
    try:
        res = minimize(
            objective_slsqp,
            h0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res.success:
            h_opt = res.x
            # Ensure sum is exact
            diff = target_sum - np.sum(h_opt)
            if abs(diff) > 1e-9:
                # Adjust
                if diff > 0:
                    idx = np.argmin(h_opt)
                    if 1.0 - h_opt[idx] >= diff:
                        h_opt[idx] += diff
                    else:
                        # Distribute
                        mask = h_opt < 1.0
                        h_opt[mask] += diff / np.sum(mask)
                else:
                    idx = np.argmax(h_opt)
                    if h_opt[idx] >= -diff:
                        h_opt[idx] += diff
                    else:
                        mask = h_opt > 0.0
                        h_opt[mask] += diff / np.sum(mask)
            
            h_opt = np.clip(h_opt, 0, 1)
            
            score = max_overlap_func(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
    except Exception:
        pass
    
    # If SLSQP didn't improve, return the best found (likely constant 0.5)
    return best_h

# EVOLVE-BLOCK-END
