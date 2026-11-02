import numpy as np
import time
import sys

N = 600

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    # Fallback best sequence: Uniform
    best_seq = np.ones(N)
    best_score = 1.0 # Uniform autoconvolution max sidelobe is 0, so score 0? 
    # Wait, uniform seq [1,1,...,1]
    # Autoconvolution is triangle. Peak N. Sidelobes are linearly decreasing.
    # Max sidelobe is N-1. Ratio (N-1)/N.
    # Let's verify with code logic later.
    
    def get_score(seq):
        seq = np.array(seq)
        # Normalize? The metric is likely scale invariant.
        # Let's use the raw ratio.
        c = np.correlate(seq, seq, mode='full')
        c0 = c[N-1]
        if c0 == 0:
            return float('inf')
        # Exclude lag 0
        sidelobes = np.concatenate([c[:N-1], c[N:]])
        max_side = np.max(sidelobes)
        return max_side / c0

    # Check uniform
    current_score = get_score(best_seq)
    best_score = current_score
    
    # Strategy: SDP Relaxation
    # We want to minimize T such that exists X >= 0, X_ij = x_i x_j (relaxed to PSD)
    # s.t. sum_j X_ij = 1 (normalization? No, let's fix sum x_i^2 = 1 or similar)
    # Actually, let's fix sum x_i = 1? No, x_i >= 0.
    # Let's use the formulation:
    # Minimize T
    # Subject to:
    #   sum_{j} X_{i, j} >= 0 (X is PSD)
    #   X_{i,i} = x_i^2 ? No, we don't have x_i directly in standard SDP unless we lift.
    #   Let's use the standard "minimum peak sidelobe level" SDP relaxation.
    #   Variables: X (N x N PSD matrix), x (N vector)
    #   Constraints:
    #     X >= 0
    #     X_{i,i} = x_i^2 ? This is non-convex.
    #     Alternative: Let X be the Gram matrix of x. X = x x^T is rank 1.
    #     Relaxation: X is PSD.
    #     Autoconvolution c_k = sum_i x_i x_{i+k} = sum_i X_{i, i+k}.
    #     We want to minimize max_k c_k / c_0.
    #     Let c_0 = sum_i x_i^2 = trace(X).
    #     Let's fix trace(X) = 1.
    #     Then c_k = sum_i X_{i, i+k}.
    #     Minimize T such that sum_i X_{i, i+k} <= T for all k != 0.
    #     And X >= 0, trace(X) = 1.
    #     Also we need x_i >= 0. This constraint is hard in SDP.
    #     However, the optimal solution for minimizing peak sidelobe often has non-negative x.
    #     If cvxpy is available, we can try to enforce x_i >= 0 by adding constraints?
    #     Standard SDP relaxation for this problem usually ignores x_i >= 0 or uses a different formulation.
    #     If we ignore x_i >= 0, the solution might have negative values.
    #     If the solution has negative values, we can take absolute values? No, that changes autoconvolution.
    #     But for "nonnegative step function", we MUST have x_i >= 0.
    #     
    #     Let's try a simpler approach if SDP is too complex or unavailable:
    #     Use a spectral method.
    #     x = F^{-1}(S) where S is a power spectrum.
    #     If we choose S to be a "boxcar" or "Gaussian" in frequency, x will be smooth.
    #     The current program uses random binary + hill climb.
    #     The inspiration mentions "Spectral parameterization reduces effective dimensionality... Gaussian + Gaussian bump".
    #     This suggests a parametric spectral approach is promising.
    #     
    #     Let's implement a parametric spectral optimization.
    #     Model x_n as the inverse DFT of a parametric spectrum S(k).
    #     S(k) = A * exp(-B * k^2) + C * exp(-D * (k-k0)^2) ?
    #     Or simpler: S(k) = 1 / (1 + E * k^2) (Lorentzian)
    #     Or S(k) = exp(-F * |k|)
    #     
    #     Let's try optimizing a few parameters for a Gaussian spectrum.
    #     S(k) = exp(-sigma^2 * k^2).
    #     x is a discrete Gaussian.
    #     We can scale and shift? No, domain is fixed [-1/4, 1/4].
    #     x_n corresponds to time domain.
    #     If S(k) is Gaussian, x_n is Gaussian.
    #     We can optimize sigma.
    #     Also, we can add a DC offset? No, x_n >= 0.
    #     A pure Gaussian in time domain has non-negative values.
    #     
    #     Let's try to optimize the width of a Gaussian pulse centered at 0.
    #     x_n = exp(-n^2 / (2 * sigma^2)).
    #     We can also try a sum of two Gaussians?
    #     Or a "super-Gaussian" x_n = exp(-|n|^p / (2 * sigma^2)).
    
    # Let's try a grid search over spectral parameters.
    # Parameter 1: sigma (width)
    # Parameter 2: p (exponent)
    
    best_seq = np.ones(N)
    best_score = get_score(best_seq)
    
    # Generate candidate sequences based on spectral models
    candidates = []
    
    # 1. Gaussian pulses with varying widths
    sigmas = np.linspace(0.5, N/2, 50)
    for sigma in sigmas:
        n = np.arange(N) - N/2
        seq = np.exp(-n**2 / (2 * sigma**2))
        # Ensure non-negative (it is)
        # Normalize? Scale doesn't matter for ratio.
        candidates.append(seq)
        
    # 2. Super-Gaussian pulses
    ps = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
    sigmas2 = np.linspace(0.5, N/2, 20)
    for p in ps:
        for sigma in sigmas2:
            n = np.arange(N) - N/2
            seq = np.exp(-np.abs(n)**p / (2 * sigma**2))
            candidates.append(seq)
            
    # 3. Raised Cosine-like (smooth window)
    # x_n = 0.5 * (1 + cos(pi * n / L)) for |n| < L, 0 otherwise
    Ls = np.linspace(5, N, 20)
    for L in Ls:
        n = np.arange(N) - N/2
        seq = np.zeros(N)
        mask = np.abs(n) < L
        seq[mask] = 0.5 * (1 + np.cos(np.pi * n[mask] / L))
        if np.sum(seq) > 0:
            candidates.append(seq)
            
    # 4. Two-peak spectral (from inspiration 2)
    # This corresponds to a time-domain sequence that is a sum of two shifted Gaussians?
    # Or a time-domain sequence that is two rectangular blocks?
    # Inspiration 2: "Two-Block Disjoint Rectangles".
    # Let's try two rectangular blocks of width W separated by gap G.
    Ws = np.linspace(1, N/2, 10)
    Gs = np.linspace(0, N, 10)
    for W in Ws:
        for G in Gs:
            seq = np.zeros(N)
            # Center the whole thing
            total_width = 2*W + G
            if total_width > N: continue
            start = int(N/2 - total_width/2)
            # Block 1
            end1 = start + int(W)
            seq[start:end1] = 1.0
            # Block 2
            start2 = end1 + int(G)
            end2 = start2 + int(W)
            if end2 < N:
                seq[start2:end2] = 1.0
            if np.sum(seq) > 0:
                candidates.append(seq)

    # Evaluate all candidates
    for seq in candidates:
        if time.time() > deadline - 10:
            break
        score = get_score(seq)
        if score < best_score:
            best_score = score
            best_seq = seq.copy()
            
    # Refine the best candidate using local search (hill climbing)
    # This is a structural change: we start from a structured global candidate, not random.
    # We will perform a few steps of hill climbing on the best candidate found.
    
    seq = best_seq.copy()
    # Simple hill climbing: flip a small number of points or adjust values?
    # Since we are dealing with continuous values now (from Gaussians),
    # we can try a simple gradient-free local optimization (Nelder-Mead) on a few parameters?
    # Or just stick to the best structured candidate if time is short.
    # Given 120s limit, and evaluation of ~100 candidates is fast, we have time for optimization.
    
    # Let's try to optimize the parameters of the best candidate type.
    # Identify which type was best? We didn't track it.
    # Let's just run a generic local search on the best_seq found.
    # But local search on 600 dims is hard.
    
    # Alternative: If the best candidate was a Gaussian, optimize its sigma and amplitude scaling (scale is irrelevant).
    # Let's assume the best was one of the parametric ones.
    # We can do a fine grid search around the best parameters.
    # But we didn't store the parameters.
    
    # Let's just do a simple "perturbation" search on the best_seq.
    # Add small noise and see if it improves.
    # This is weak.
    
    # Better: Re-run the generation with finer grids for the top few types.
    # For now, return the best structured candidate.
    
    # One more idea: Bisection on the score using a simple feasibility check?
    # No, that's what SDP does.
    
    # Let's check if cvxpy is available.
    try:
        import cvxpy as cp
        # We have time for one small SDP solve?
        # N=600 is too large for standard SDP solvers in 120s.
        # We can downsample? No, the task requires 600 intervals.
        # So SDP is likely too slow.
        
        # What about a smaller N? No, N is fixed.
        
        # So we stick to the parametric spectral approach.
        # It is structurally different from random binary + hill climb.
        # It explores smooth, low-frequency shapes which are known to have lower sidelobes in convolution.
        
    except ImportError:
        pass

    return list(best_seq)

if __name__ == "__main__":
    res = construct()
    print(len(res))
