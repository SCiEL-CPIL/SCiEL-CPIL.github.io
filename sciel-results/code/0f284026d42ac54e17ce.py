import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_ratio = _compute_ratio(best_h)
    
    # Strategy: Simulated Annealing
    # We view the problem as maximizing a non-convex function over the simplex (or non-negative orthant).
    # Since the ratio is scale-invariant, we can normalize h to sum to 1 (or keep it unconstrained and rely on the landscape).
    # Actually, scaling h by c changes L2 by c^2, L1 by c, Conv by c^2.
    # Ratio = (c^2 L2) / (c L1 * c^2 Conv) = Ratio / c.
    # Wait.
    # ||c h||_2^2 = c^2 ||h||_2^2
    # ||c h||_1 = c ||h||_1
    # ||c h * c h||_inf = ||c^2 (h*h)||_inf = c^2 ||h*h||_inf
    # Ratio = (c^2 L2) / (c L1 * c^2 Conv) = L2 / (c L1 Conv) = (1/c) * (L2 / (L1 Conv)).
    # So the ratio is inversely proportional to the scale.
    # To maximize the ratio, we want the scale to be as small as possible?
    # But h must be non-negative. If we scale down, the ratio goes up.
    # Is there a lower bound on the scale?
    # The problem says "Choose 50 nonnegative heights". It does not specify a normalization like sum=1 or max=1.
    # However, if I can make h arbitrarily small, the ratio goes to infinity.
    # Let's re-read carefully: "maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
    # If the answer is unbounded, the task would be trivial or ill-posed.
    # Usually in such inequalities, there is an implicit normalization or the constant C is defined such that:
    # ||h||_2^2 >= C ||h||_1 ||h*h||_inf
    # If I scale h by c, LHS ~ c^2, RHS ~ c * c^2 = c^3.
    # So c^2 >= C c^3 => C <= 1/c.
    # If c -> 0, C can be arbitrarily large?
    # No, the inequality must hold for ALL h.
    # If I pick a specific h with small scale, it doesn't constrain C for other h.
    # The constant C is the infimum of the ratio over all h.
    # Ratio(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf).
    # If I scale h by c, Ratio(c h) = Ratio(h) / c.
    # If I take c -> 0, Ratio -> infinity.
    # If I take c -> infinity, Ratio -> 0.
    # So the infimum is 0? That would mean C=0.
    # But the seed program scores 1.14, implying the "score" is the value of the ratio for the specific h returned.
    # "The evaluator measures the raw value (higher is better)."
    # "score is the lower bound on C2".
    # This phrasing is tricky. "maximize the ratio ... score is the lower bound on C2".
    # If I return h, the evaluator computes Ratio(h).
    # If Ratio(h) is large, it suggests C2 could be large?
    # But C2 is the *infimum*. A single large ratio doesn't prove the infimum is large.
    # However, in these coding challenges, "maximize the ratio" usually means "find an h that achieves a high ratio".
    # But if the ratio is unbounded (by scaling down), the problem is trivial: return a very small h.
    # Let's check the seed program. It returns normalized heights (sum ~ 1 or similar).
    # The seed program does not scale down to 1e-100.
    # Why?
    # Maybe there is a constraint I missed?
    # "Choose 50 nonnegative heights on equal intervals of [-1/4,1/4]"
    # Does this imply the function is defined on this interval and the heights are samples?
    # Or does it imply the support is within this interval?
    # "heights on equal intervals" usually means a step function.
    # If the function is f(x) = sum h_i * 1_{[x_i, x_{i+1})}(x), then:
    # L1 = sum h_i * dx
    # L2^2 = sum h_i^2 * dx
    # Convolution: (f*f)(t) = int f(s) f(t-s) ds.
    # For step functions, the convolution is also a piecewise linear function (trapezoids).
    # The supremum norm of the convolution depends on the specific geometry.
    # Crucially, scaling h by c scales f by c.
    # L1 ~ c, L2^2 ~ c^2, Conv ~ c^2.
    # Ratio = c^2 / (c * c^2) = 1/c.
    # If I can choose c arbitrarily small, the ratio is arbitrarily large.
    # Is there a constraint on the magnitude?
    # "Choose 50 nonnegative heights".
    # Maybe the heights must be integers? No, "nonnegative heights" usually implies reals.
    # Maybe the L2 norm is defined differently?
    # "squared L2 norm" = ||h||_2^2.
    # If the problem is indeed unbounded, the seed program's score of 1.14 is just the ratio for that specific h.
    # To beat 1.14, I just need to find an h with a higher ratio.
    # Since Ratio(h/c) = c * Ratio(h), I can trivially beat 1.14 by scaling the seed h by 0.5.
    # Ratio(0.5 h) = 2 * Ratio(h) = 2.28.
    # Why didn't the seed program do this?
    # Perhaps the evaluator normalizes h?
    # "The evaluator measures the raw value".
    # If the evaluator normalizes h (e.g., to L1=1 or L2=1), then scaling doesn't matter.
    # Let's assume the evaluator normalizes h to L1=1.
    # If L1=1, then Ratio = L2^2 / SupConv.
    # This is a well-posed optimization problem on the simplex.
    # The seed program likely operates on the simplex (sum h_i = 1) or the evaluator does.
    # Let's assume the evaluator normalizes to L1=1.
    # Then we want to maximize ||h||_2^2 / ||h*h||_inf subject to sum h_i = 1, h_i >= 0.
    # This is equivalent to minimizing ||h*h||_inf / ||h||_2^2.
    # Or maximizing the "spread" of the convolution relative to the energy.
    # A uniform distribution has low L2 (spread out) and low Conv peak.
    # A spike has high L2 (concentrated) and high Conv peak.
    # Let's test:
    # h = [1, 0, ...]. L1=1, L2^2=1. Conv = [1, 0, ...]. Sup=1. Ratio = 1.
    # h = [1/N, ..., 1/N]. L1=1, L2^2 = N*(1/N^2) = 1/N.
    # Conv is a triangle. Peak is at center.
    # For N=50, uniform h.
    # Conv peak approx (N/2) * (1/N)^2 * N? No.
    # (h*h)_k = sum h_j h_{k-j}.
    # For uniform 1/N, sum is over j where 0<=j<N and 0<=k-j<N.
    # Max overlap is N terms (if k is central).
    # (h*h)_center = N * (1/N * 1/N) = 1/N.
    # Ratio = (1/N) / (1 * 1/N) = 1.
    # So both spike and uniform give 1.
    # The seed gets 1.14. So there are shapes that do better than 1.
    # This confirms the problem is to find a shape on the simplex (or effectively, since scale is fixed by normalization) that maximizes the ratio.
    
    # Strategy: Simulated Annealing on the simplex.
    # Representation: h as a vector.
    # Move: Pick two indices i, j. Move mass from i to j or vice versa.
    # h_new[i] = h[i] - delta
    # h_new[j] = h[j] + delta
    # This keeps sum constant.
    # We need to ensure h_new[i] >= 0.
    
    def _ratio_normalized(h):
        # Assume h is already normalized to sum=1?
        # Let's normalize inside to be safe, or just compute ratio and let the optimizer handle it.
        # Since ratio is scale invariant in the sense that it's defined on the projective space,
        # we can fix sum(h)=1.
        s = np.sum(h)
        if s < 1e-15:
            return 0.0
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        l1 = 1.0
        conv = np.convolve(h_norm, h_norm, mode='full')
        sup_conv = np.max(conv)
        if sup_conv < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_conv)

    # Initial state: Uniform
    h = np.ones(N) / N
    current_ratio = _ratio_normalized(h)
    
    # SA Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 100000
    
    for i_iter in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate a move
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        # Random delta
        # Scale delta relative to current heights
        # To allow large moves, we can sample delta from a distribution dependent on T
        # Or just use a fixed step size and adapt it?
        # Let's try a simple random walk with fixed step size, but scale by T?
        # No, T controls acceptance, not step size.
        # Let's use a step size that is a fraction of the total mass.
        # delta_max = min(h[i], h[j])
        # If we want to explore, we might want to move significant mass.
        # Let's sample delta from Uniform(-delta_max, delta_max) ?
        # Or just pick a random target height for one of them?
        
        # Alternative Move: Pick i, j. Set h[j] to a random value, adjust h[i] to maintain sum.
        # This is a "star" move or "coordinate" move in the simplex.
        # Let's try: Pick i, j.
        # new_h_j = np.random.uniform(0, 1) * (h[i] + h[j])  # Share the mass between i and j
        # new_h_i = (h[i] + h[j]) - new_h_j
        # This is a convex combination, so it stays in the simplex.
        
        old_i, old_j = h[i], h[j]
        total_ij = old_i + old_j
        if total_ij < 1e-12:
            continue
            
        # Random convex combination
        r = np.random.uniform(0, 1)
        new_j = r * total_ij
        new_i = (1 - r) * total_ij
        
        h[i] = new_i
        h[j] = new_j
        
        new_ratio = _ratio_normalized(h)
        
        # Acceptance criterion
        if new_ratio > current_ratio:
            current_ratio = new_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_h = h.copy()
        else:
            # Metropolis criterion
            # delta = current_ratio - new_ratio > 0
            # Accept with prob exp(-delta / T)
            # Since ratio is positive, and we are maximizing,
            # if new < current, delta = current - new.
            delta = current_ratio - new_ratio
            if delta < 1e-12:
                # Numerical noise
                pass
            else:
                accept_prob = np.exp(-delta / T)
                if np.random.uniform(0, 1) < accept_prob:
                    current_ratio = new_ratio
                else:
                    # Reject
                    h[i] = old_i
                    h[j] = old_j
                    
        T *= cooling_rate
        
    # Final check
    final_ratio = _ratio_normalized(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        
    return list(best_h)

if __name__ == "__main__":
    # Quick test
    h = construct()
    print(_compute_ratio(h))
