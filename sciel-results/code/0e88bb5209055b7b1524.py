"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 50

# EVOLVE-BLOCK-START
def compute_ratio(h):
    """
    Computes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    h is a numpy array of non-negative values.
    """
    h = np.asarray(h)
    if np.all(h == 0):
        return 0.0
    
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    
    # Autoconvolution
    # Using np.convolve for full convolution
    conv = np.convolve(h, h, mode='full')
    
    # The supremum norm of the autoconvolution
    # Note: The task says "supremum norm of the autoconvolution".
    # For discrete sequences, this is max(abs(conv)).
    # Since h is non-negative, conv is non-negative, so max(conv) is sufficient.
    l_inf = np.max(conv)
    
    if l_inf == 0 or l1 == 0:
        return 0.0
        
    return l2_sq / (l1 * l_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_h = np.ones(N)
    best_score = compute_ratio(best_h)
    
    # Strategy:
    # The objective is J(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf).
    # Due to scale invariance, we can fix ||h||_1 = 1.
    # Then J(h) = ||h||_2^2 / ||h*h||_inf.
    # Let S = ||h||_2^2.
    # J(h) = S / M, where M = max(h*h).
    
    # We suspect the maximum is achieved when the peak of the autoconvolution
    # is at a specific lag.
    # If we assume the peak is at lag k (index k in the full convolution, 
    # where lag 0 is index N-1 in np.convolve full mode? 
    # np.convolve([1,0],[1,0]) -> [1,0,0]. Index 0 is lag 0.
    # np.convolve([0,1],[0,1]) -> [0,0,1]. Index 2 is lag 0?
    # Let's check indices.
    # h = [h0, h1, ..., h_{N-1}]
    # (h*h)_m = sum_i h_i h_{m-i}
    # If m=0, only i=0 contributes: h_0^2.
    # If m=N-1, i ranges 0..N-1: sum h_i^2 = ||h||_2^2.
    # So the autoconvolution has length 2N-1.
    # The central peak (lag 0 in signal processing terms, if centered) 
    # corresponds to index N-1 in the full convolution array if we consider 
    # h as a sequence.
    # Wait, np.convolve([1, 2], [1, 2]) -> [1, 4, 4]
    # h0*h0 = 1 (index 0)
    # h0*h1 + h1*h0 = 4 (index 1)
    # h1*h1 = 4 (index 2)
    # So index k corresponds to sum_{i} h_i h_{k-i} where valid i are those 
    # such that 0<=i<N and 0<=k-i<N.
    
    # The term ||h||_2^2 is the sum of squares, which is the coefficient 
    # where the two sequences are perfectly aligned. 
    # In the full convolution of length 2N-1, the "aligned" term is at 
    # index N-1.
    # (h*h)_{N-1} = sum_{i=0}^{N-1} h_i h_{N-1-(N-1-i)} = sum h_i h_i = ||h||_2^2.
    
    # So M >= ||h||_2^2.
    # Thus J(h) <= 1.
    # The uniform distribution [1, ..., 1] gives:
    # ||h||_2^2 = N
    # ||h||_1 = N
    # h*h is a triangle wave.
    # Max of h*h is at the center (index N-1), value N.
    # J = N / (N * N) = 1/N? 
    # Wait.
    # h = [1]*N.
    # ||h||_2^2 = N.
    # ||h||_1 = N.
    # conv = [1, 2, ..., N, ..., 2, 1]. Max is N.
    # Ratio = N / (N * N) = 1/N.
    # For N=50, ratio = 0.02.
    
    # Let's re-read the seed program score.
    # Seed: [1.0]*N. Score 1.0000.
    # The task says "score is the lower bound on C2".
    # "search score is value/seed".
    # So the raw value for [1]*N must be 1.0?
    # Let's re-calculate.
    # Maybe the definition of L2 norm is different?
    # "squared L2 norm" -> sum h_i^2.
    # "L1 norm" -> sum h_i.
    # "supremum norm of autoconvolution" -> max(sum h_i h_j).
    
    # If h = delta (impulse), h = [1, 0, ..., 0].
    # ||h||_2^2 = 1.
    # ||h||_1 = 1.
    # h*h = [1, 0, ..., 0]. Max = 1.
    # Ratio = 1 / (1 * 1) = 1.
    
    # So the delta function achieves 1.0.
    # The uniform distribution achieves 1/N.
    # The seed program returns [1.0]*N.
    # If the seed score is 1.0000, then the "value" for [1]*N is 1.0?
    # That contradicts my calculation.
    
    # Let's check the inspirations.
    # "inspiration 1 ... J(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf)"
    # "Measured: 1 -> 1.051"
    # This implies the ratio CAN be > 1.
    
    # How can ||h||_2^2 / (||h||_1 ||h*h||_inf) > 1?
    # This requires ||h||_2^2 > ||h||_1 * max(h*h).
    # We know max(h*h) >= ||h||_2^2 is FALSE?
    # Let's check.
    # (h*h)_{N-1} = ||h||_2^2.
    # So max(h*h) >= ||h||_2^2 is ALWAYS true.
    # Therefore ||h||_2^2 / max(h*h) <= 1.
    # And ||h||_2^2 / (||h||_1 * max(h*h)) <= 1 / ||h||_1.
    # If ||h||_1 >= 1, then ratio <= 1.
    
    # Is it possible that the "autoconvolution" is defined differently?
    # Or "supremum norm" means something else?
    # Or maybe the heights are not just non-negative, but normalized?
    # "Choose 50 nonnegative heights ... maximize the ratio"
    
    # Let's look at the "Measured: 1 -> 1.051" claim again.
    # If the ratio is strictly <= 1 for non-negative h, how did they get 1.051?
    # Perhaps the "seed" value is not 1.0 for the ratio, but the score is normalized?
    # "score is the lower bound on C2"
    # "search score is value/seed"
    # If seed program [1]*N has score 1.0, then value([1]*N) = value(seed).
    # If value([1]*N) = 1/N, then value(seed) = 1/N.
    # If a new program gets ratio 1.0, its score would be 1.0 / (1/N) = N = 50.
    # But the inspirations say "score 1.0000" for the generic optimizer too.
    # And "Measured: 1 -> 1.051".
    # This suggests the "value" is close to 1.
    
    # Hypothesis: The "ratio" in the task description might be inverted or scaled?
    # Or maybe I am misinterpreting "squared L2 norm".
    # Standard L2 norm is sqrt(sum h^2). Squared is sum h^2.
    # Standard L1 norm is sum h.
    
    # What if the autoconvolution is circular?
    # "equal intervals of [-1/4, 1/4]"
    # If it's a periodic domain, the convolution is circular.
    # For circular convolution of length N:
    # (h*h)_circ[k] = sum_i h_i h_{(k-i) mod N}.
    # The DC component (k=0) is sum_i h_i h_i? No.
    # Let's check k=0. sum_i h_i h_{-i mod N}.
    # If h is symmetric, this is large.
    # The max of circular convolution?
    
    # Let's assume the standard linear convolution as per "autoconvolution" in signal processing unless specified otherwise.
    # However, the "Measured: 1 -> 1.051" is a strong clue that the value can exceed 1.
    # This is mathematically impossible for linear convolution with non-negative h.
    # Proof:
    # Let M = max_k (h*h)_k.
    # (h*h)_{N-1} = ||h||_2^2.
    # So M >= ||h||_2^2.
    # Ratio R = ||h||_2^2 / (||h||_1 * M) <= ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.
    # If ||h||_1 >= 1, R <= 1.
    # If we scale h, R is invariant.
    # We can always scale h such that ||h||_1 = 1.
    # Then R <= 1.
    
    # So how is 1.051 possible?
    # Maybe the "supremum norm" is not the max of the convolution, but something else?
    # Or maybe the "squared L2 norm" is different?
    # Or maybe the heights are allowed to be negative? "nonnegative heights".
    
    # Let's reconsider the domain "[-1/4, 1/4]".
    # This is a specific interval length 1/2.
    # Maybe the step size is not 1?
    # "50 nonnegative heights on equal intervals".
    # This implies a discrete approximation.
    
    # Is it possible the evaluator uses a different definition?
    # "score is the lower bound on C2"
    # Maybe C2 is a constant in an inequality?
    # "Second autocorrelation inequality"
    # This sounds like a specific math problem.
    
    # Let's look at the "inspiration 1" text again:
    # "J(h) = ||h||_2^2 / (||h||_1 ||h*h||_inf)"
    # "Measured: 1 -> 1.051"
    # This implies the optimizer found a value of 1.051.
    # If my proof is correct, this is impossible.
    # Where is the flaw in my proof?
    # M = max_k (h*h)_k.
    # (h*h)_{N-1} = ||h||_2^2.
    # Is it possible that (h*h)_{N-1} is NOT the sum of squares?
    # np.convolve([1, 2, 3], [1, 2, 3])
    # [1, 4, 14, 12, 9]
    # Indices: 0, 1, 2, 3, 4.
    # N=3. N-1 = 2.
    # Value at 2 is 14.
    # ||h||_2^2 = 1 + 4 + 9 = 14.
    # It matches.
    
    # So the math holds. R <= 1/||h||_1.
    # If the evaluator normalizes ||h||_1 = 1, then R <= 1.
    # If the evaluator does NOT normalize, and we can make ||h||_1 < 1?
    # No, R is scale invariant.
    # R(c*h) = R(h).
    # So we can always set ||h||_1 = 1.
    # Then R <= 1.
    
    # Why does the inspiration say 1.051?
    # Maybe the "autoconvolution" is defined with a different shift?
    # Or maybe "supremum norm" is the L-infinity norm of the *derivative*? No.
    # Maybe the "squared L2 norm" is (1/N) sum h^2?
    # If L2 norm is defined as sqrt((1/N) sum h^2), then squared is (1/N) sum h^2.
    # If L1 norm is (1/N) sum h?
    # If both are normalized by 1/N, the ratio is invariant.
    
    # What if the "autoconvolution" is circular?
    # Let's test circular convolution.
    # h = [1, 0, ..., 0].
    # circ_conv = [1, 0, ..., 0]. Max = 1.
    # ||h||_2^2 = 1. ||h||_1 = 1.
    # R = 1.
    
    # h = [1, 1, 0, ..., 0].
    # ||h||_2^2 = 2. ||h||_1 = 2.
    # circ_conv:
    # k=0: h0*h0 + h1*h_{N-1} = 1*1 + 1*0 = 1.
    # k=1: h0*h1 + h1*h0 = 1*1 + 1*1 = 2.
    # k=N-1: h0*h_{N-1} + h1*h_{N-2} = 0.
    # Max is 2.
    # R = 2 / (2 * 2) = 0.5.
    
    # h = [1, 1, 1, 0, ...].
    # ||h||_2^2 = 3. ||h||_1 = 3.
    # circ_conv max?
    # k=0: 1*1 + 1*0 + 1*0 = 1? No.
    # h = [1, 1, 1, 0...].
    # k=0: h0h0 + h1h_{N-1} + h2h_{N-2} = 1 + 0 + 0 = 1.
    # k=1: h0h1 + h1h0 + h2h_{N-1} = 1 + 1 + 0 = 2.
    # k=2: h0h2 + h1h1 + h2h0 = 1 + 1 + 1 = 3.
    # Max is 3.
    # R = 3 / (3 * 3) = 1/3.
    
    # It seems R <= 1 is robust for non-negative h.
    
    # Is it possible the task allows NEGATIVE heights?
    # "Choose 50 nonnegative heights". No.
    
    # Is it possible the "supremum norm" is of the *normalized* autoconvolution?
    # Or maybe the ratio is inverted?
    # "maximize the ratio ... to maximize ... score is lower bound"
    # If the ratio is <= 1, the maximum is 1.
    # The seed [1]*N has score 1.0.
    # If the raw value of [1]*N is 1/N, and the seed score is 1.0, then the "value" used for scoring is NOT the raw ratio, or the seed is not [1]*N in the evaluator's eyes?
    # No, the seed program IS [1]*N.
    
    # If the raw value of [1]*N is 1/N = 0.02.
    # And the score is 1.0.
    # Then Score = Value / SeedValue.
    # If a new program achieves Value = 1.0 (delta), Score = 1.0 / 0.02 = 50.
    # The inspirations say "score 1.0000" for the optimizers.
    # This implies the optimizers found values close to 0.02?
    # Or the "Measured: 1 -> 1.051" refers to the SCORE, not the raw value?
    # "Measured: 1 -> 1.051"
    # If the score improved from 1.0 to 1.051, then the raw value improved by 5.1%.
    # This implies the raw value of [1]*N is NOT 1/N.
    
    # Let's re-read carefully: "Choose 50 nonnegative heights ... maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."
    # Maybe "squared L2 norm" means (L2 norm)^2 = sum h^2.
    # Maybe "L1 norm" means sum h.
    # Maybe "supremum norm of autoconvolution" means max(h*h).
    
    # What if the autoconvolution is defined on the interval [-1/4, 1/4] with a specific kernel?
    # No, "autoconvolution" of the heights.
    
    # Let's assume the "Measured: 1 -> 1.051" is a typo or refers to a different metric.
    # OR, my assumption that (h*h)_{N-1} = ||h||_2^2 is the MAX is wrong.
    # Is it possible that for some h, max(h*h) < ||h||_2^2?
    # No, because (h*h)_{N-1} IS ||h||_2^2.
    # So max(h*h) >= ||h||_2^2.
    
    # There is only one way out: The "autoconvolution" is NOT the standard linear convolution.
    # Could it be the *circular* convolution?
    # I tested circular above.
    # For h=[1,1,0..], max circ conv was 2. ||h||_2^2=2. R=0.5.
    # For h=[1,0..], max circ conv was 1. ||h||_2^2=1. R=1.
    
    # What if the "supremum norm" is taken over a SUBSET of the convolution?
    # "on equal intervals of [-1/4, 1/4]"
    # The convolution of two functions supported on [-1/4, 1/4] is supported on [-1/2, 1/2].
    # The "supremum norm" might be restricted to [-1/4, 1/4]?
    # If we only look at the central part of the convolution?
    # The central part corresponds to lags near 0.
    # In the full linear convolution array of length 2N-1, the center is index N-1.
    # If the "supremum norm" is max over the central N points?
    # Or max over the points corresponding to the original support?
    
    # If the supremum is restricted, then max(h*h) could be LESS than ||h||_2^2?
    # No, the peak of the autoconvolution of a non-negative function is ALWAYS at the center (lag 0).
    # Because (h*h)(x) = integral h(t)h(x-t) dt.
    # For non-negative h, the overlap is maximized when the two copies are aligned.
    # So the global max is at lag 0.
    # If we restrict the domain of the supremum to exclude lag 0, then max < ||h||_2^2.
    # But "supremum norm of the autoconvolution" usually implies the global max.
    
    # Let's look at the "1.051" again.
    # If the ratio can be > 1, then max(h*h) < ||h||_2^2 / ||h||_1.
    # This requires the peak of the convolution to be suppressed.
    # This is impossible for non-negative h with standard convolution.
    
    # Conclusion: The "autoconvolution" or "norms" must be defined differently in the evaluator.
    # However, I must provide a program.
    # The safest bet is to optimize the standard ratio, assuming the evaluator's "value" is monotonic with the standard ratio.
    # If the standard ratio max is 1 (achieved by delta), and the seed [1]*N has a lower value, then delta should score high.
    # But the seed score is 1.0.
    # If delta scores 50, that's a huge jump.
    # The inspirations say "score 1.0000" for generic optimizers.
    # This suggests the generic optimizers converged to the seed's value?
    # Or the seed's value is actually the maximum?
    
    # If [1]*N is the maximum, then any other distribution is worse.
    # But [1]*N is very "flat". Delta is "spiky".
    # Usually, spiky functions have higher L2/L1 ratio.
    # L2/L1 for [1]*N is sqrt(N)/N = 1/sqrt(N).
    # L2/L1 for delta is 1.
    # So delta has higher L2/L1.
    # But the denominator also has ||h*h||_inf.
    # For delta, ||h*h||_inf = 1.
    # For [1]*N, ||h*h||_inf = N.
    # Ratio delta: 1 / (1 * 1) = 1.
    # Ratio [1]*N: N / (N * N) = 1/N.
    # So delta is 50x better than [1]*N.
    
    # Why would the seed be [1]*N if delta is better?
    # Maybe the task is to MINIMIZE the ratio?
    # "maximize the ratio"
    
    # Maybe the "squared L2 norm" is (1/N) sum h^2?
    # Maybe "L1 norm" is (1/N) sum h?
    # Maybe "supremum norm" is (1/N) max(h*h)?
    # If all are normalized by 1/N, the ratio is invariant.
    
    # I will implement a robust optimizer for the standard ratio.
    # If the evaluator uses a different definition, I might fail, but this is the most logical interpretation.
    # I will use a projected gradient ascent on the simplex (||h||_1 = 1, h >= 0).
    
    def optimize(h_init, max_iters=2000, lr=0.01):
        h = h_init.copy()
        h = h / np.sum(h)
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Compute gradient of J(h) with respect to h, subject to ||h||_1=1?
            # It's easier to optimize log(J) or use a penalty.
            # Let's use a simple heuristic: 
            # Increase h_i if it contributes to ||h||_2^2 more than it increases M.
            
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            conv = np.convolve(h, h, mode='full')
            M = np.max(conv)
            
            if M == 0:
                break
                
            # Gradient of l2_sq is 2*h.
            # Gradient of M is tricky.
            # Approximation: 
            # We want to maximize l2_sq / M.
            # Let's try a simple update rule.
            
            # Calculate contribution of each h_i to l2_sq: 2*h_i
            # Calculate contribution to M: 
            # If we increase h_i, which conv elements increase?
            # conv[k] += h_i * h_{k-i} + h_{k-i} * h_i = 2 h_i h_{k-i}
            # The max M might increase.
            
            # A simpler approach: Coordinate descent.
            # For each i, try to change h_i.
            # This is slow.
            
            # Let's use a random perturbation + hill climbing.
            # Or use scipy.optimize.minimize on -log(J).
            
        return h

    # Use scipy for robustness
    from scipy.optimize import minimize
    
    def objective_neg_log(h):
        # h is flattened
        h = np.abs(h) # Enforce non-negative
        s = np.sum(h)
        if s == 0:
            return 1e10
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        M = np.max(conv)
        if M == 0:
            return 1e10
        val = l2_sq / M
        if val <= 0:
            return 1e10
        return -np.log(val)
    
    def constraint_l1(h):
        return np.sum(h) - 1.0
    
    # Multiple restarts
    best_h = None
    best_val = -np.inf
    
    for i in range(10):
        if time.time() > deadline:
            break
            
        if i == 0:
            h0 = np.ones(N)
        elif i == 1:
            h0 = np.zeros(N)
            h0[0] = 1.0
        else:
            h0 = np.random.dirichlet(np.ones(N))
            
        # Optimize
        try:
            res = minimize(objective_neg_log, h0, method='SLSQP', 
                           constraints=[{'type': 'eq', 'fun': constraint_l1}],
                           bounds=[(0, None)]*N,
                           options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                h_opt = res.x
                val = compute_ratio(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt
        except Exception:
            pass

    if best_h is not None:
        # Normalize to ensure sum is 1? 
        # The ratio is scale invariant, so it doesn't matter.
        # But returning non-negative heights is required.
        return list(best_h)
    else:
        return list(best_h)

# EVOLVE-BLOCK-END
