import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """Compute the normalized maximum of autoconvolution."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # We want to minimize R = max(h*h) / (sum(h) * max(h))
    # Let y = log(h). Then h = exp(y).
    # R = sum(exp(2y)) / (sum(exp(y)) * max(exp(y)))
    #
    # This is not necessarily convex in y. 
    # However, if we fix the maximum H (e.g. H=1), then R = sum(exp(2y)) / (sum(exp(y)) * H).
    # Minimizing sum(exp(2y)) subject to sum(exp(y)) >= A and max(y) <= 0 is a convex problem?
    # No, sum(exp(2y)) is convex. sum(exp(y)) is convex.
    # The ratio of convex functions is not convex.
    #
    # Let's try a different structural approach.
    # The function f(h) = max(h*h) / (sum(h)*max(h)) is homogeneous of degree 0.
    # So we can fix sum(h) = 1.
    # Then we minimize max(h*h) / max(h) = max(h^2 / max(h)).
    # Since max(h) is the maximum element, let M = max(h).
    # Then we minimize max(h^2) / M.
    # Note that max(h^2) = M^2.
    # So R = M^2 / (1 * M) = M.
    # Wait. If sum(h)=1, R = max(h*h) / (1 * max(h)) = max(h^2)/max(h) = max(h) because h>=0.
    # So minimizing R is equivalent to minimizing the maximum height M, subject to sum(h)=1 and h>=0.
    # The minimum possible maximum height is 1/N (uniform distribution).
    # If h is uniform, h_i = 1/N.
    # Then R = (1/N) / (1 * 1/N) = 1.
    # Is it possible to get R < 1?
    # Let's check the definition again.
    # "minimize the normalized maximum of its autoconvolution"
    # Score = max(conv) / (area * max_val)
    # If h is uniform:
    # conv is a triangle. Peak is at center.
    # h_i = 1/N. Area = 1. max_val = 1/N.
    # conv_peak = sum(h_i^2) = N * (1/N^2) = 1/N.
    # Score = (1/N) / (1 * 1/N) = 1.
    #
    # Let's check a two-level distribution.
    # h has k entries of height H and N-k entries of height L.
    # Area = kH + (N-k)L = 1.
    # max_val = H (assuming H > L).
    # conv_peak = sum(h^2) = kH^2 + (N-k)L^2.
    # Score = (kH^2 + (N-k)L^2) / (1 * H) = kH + (N-k)L^2/H.
    # We want to minimize this.
    # From area constraint: L = (1 - kH) / (N-k).
    # Substitute L.
    # Let's try N=4, k=1.
    # L = (1-H)/3.
    # Score = H + 3 * ((1-H)/3)^2 / H = H + (1-H)^2 / (3H).
    # d/dH [ H + (1-2H+H^2)/(3H) ] = 1 + [ (3H)(-2+2H) - (1-2H+H^2)(3) ] / (9H^2)
    # = 1 + [ -6H + 6H^2 - 3 + 6H - 3H^2 ] / (9H^2)
    # = 1 + [ 3H^2 - 3 ] / (9H^2) = 1 + (H^2-1)/(3H^2) = (3H^2 + H^2 - 1)/(3H^2) = (4H^2-1)/(3H^2).
    # Set to 0: 4H^2 = 1 => H = 0.5.
    # L = (1-0.5)/3 = 1/6.
    # Score = 0.5 + 3*(1/36)/0.5 = 0.5 + 1/6 = 2/3 ≈ 0.666.
    # This is better than 1.
    #
    # So the problem is: minimize M such that there exists h with sum(h)=1, max(h)=M, and min(h)=0?
    # No, h can be any non-negative step function.
    # The score is max(conv) / (area * max_val).
    # For any h, max(conv) >= sum(h^2) is false?
    # Actually, for non-negative h, max(conv) is the peak of the autoconvolution.
    # The peak of autoconvolution of a non-negative sequence is NOT necessarily sum(h^2).
    # sum(h^2) is the value of the convolution at lag 0.
    # max(conv) is the maximum over all lags.
    # For a symmetric unimodal h, the peak is at lag 0, so max(conv) = sum(h^2).
    # But for other shapes, the peak might be elsewhere?
    # No, for non-negative sequences, the autoconvolution is maximized at lag 0?
    # Let's check.
    # (h*h)(k) = sum h_i h_{i+k}.
    # (h*h)(0) = sum h_i^2.
    # Is sum h_i h_{i+k} <= sum h_i^2?
    # By Cauchy-Schwarz, (sum h_i h_{i+k})^2 <= (sum h_i^2)(sum h_{i+k}^2) = (sum h_i^2)^2.
    # So sum h_i h_{i+k} <= sum h_i^2.
    # Thus, for non-negative h, max(conv) = sum(h^2) = ||h||_2^2.
    #
    # So the score is ||h||_2^2 / (||h||_1 * max(h)).
    # Let M = max(h). Then ||h||_2^2 <= M ||h||_1.
    # So Score <= 1.
    # We want to minimize Score.
    # Score = ||h||_2^2 / (||h||_1 M).
    # Since h is homogeneous, let ||h||_1 = 1.
    # Then Score = ||h||_2^2 / M.
    # We want to minimize ||h||_2^2 / M subject to sum(h)=1, 0 <= h_i <= M.
    # To minimize ||h||_2^2, we want h to be as uniform as possible.
    # But h_i <= M.
    # If M >= 1/N, we can set h_i = 1/N for all i. Then ||h||_2^2 = 1/N.
    # Score = (1/N) / (1 * 1/N) = 1.
    # If we make M smaller?
    # If M < 1/N, it's impossible to have sum(h)=1 with h_i <= M because sum(h) <= N*M < 1.
    # So M must be at least 1/N.
    # Wait, my previous calculation gave 2/3 for N=4.
    # Let's re-evaluate.
    # h = [0.5, 1/6, 1/6, 1/6].
    # sum = 1. M = 0.5.
    # ||h||_2^2 = 0.25 + 3*(1/36) = 0.25 + 1/12 = 3/12 + 1/12 = 4/12 = 1/3.
    # Score = (1/3) / (1 * 0.5) = 2/3.
    # Why did the uniform case give 1?
    # Uniform: h = [0.25, 0.25, 0.25, 0.25].
    # M = 0.25.
    # ||h||_2^2 = 4 * 0.0625 = 0.25.
    # Score = 0.25 / (1 * 0.25) = 1.
    # So making M larger (and the distribution less uniform) reduced the score?
    # Score = ||h||_2^2 / M.
    # If we increase M, we can decrease ||h||_2^2?
    # No, ||h||_2^2 is minimized when h is uniform.
    # Uniform gives min ||h||_2^2 for fixed sum.
    # But M is also smaller in uniform case.
    # Let's analyze f(M) = min ||h||_2^2 / M subject to sum(h)=1, h_i <= M.
    # The min ||h||_2^2 for fixed sum and upper bound M is achieved by making h_i as equal as possible.
    # If M >= 1/N, the unconstrained min is uniform, h_i=1/N. ||h||_2^2 = 1/N.
    # But if we choose M > 1/N, we can still choose uniform h. Then M_eff = 1/N.
    # The score uses the actual max(h), which is 1/N.
    # So if we pick a non-uniform h with max M > 1/N, the score is ||h||_2^2 / M.
    # We want to minimize this ratio.
    # Let's parameterize by M.
    # For a given M > 1/N, what is the min ||h||_2^2?
    # We want to minimize sum h_i^2 subject to sum h_i = 1, 0 <= h_i <= M.
    # The optimal solution is to have as many h_i = M as possible, and one fractional, rest 0?
    # No, to minimize sum of squares, we want values to be close to each other.
    # The constraint is h_i <= M.
    # If we are forced to have max h_i = M, then at least one h_i = M.
    # To minimize sum h_i^2, we should distribute the remaining sum (1-M) among the other N-1 elements as equally as possible, but they can be at most M.
    # If M >= 1/(N-1), we can distribute equally: h_1=M, h_2...h_N = (1-M)/(N-1).
    # Then ||h||_2^2 = M^2 + (N-1) * ((1-M)/(N-1))^2 = M^2 + (1-M)^2/(N-1).
    # Score(M) = [M^2 + (1-M)^2/(N-1)] / M = M + (1-M)^2 / ((N-1)M).
    # We want to minimize this for M in [1/N, 1].
    # Let g(M) = M + (1-M)^2 / ((N-1)M).
    # g'(M) = 1 + [ 2(1-M)(-1)M - (1-M)^2 ] / ((N-1)M^2)
    # = 1 - (1-M)[2M + 1-M] / ((N-1)M^2)
    # = 1 - (1-M)(M+1) / ((N-1)M^2)
    # = 1 - (1-M^2) / ((N-1)M^2)
    # Set g'(M) = 0:
    # (N-1)M^2 = 1 - M^2
    # N M^2 = 1
    # M = 1/sqrt(N).
    # This is the optimal M.
    # The corresponding h is: one element M, N-1 elements (1-M)/(N-1).
    # Let's check if (1-M)/(N-1) <= M.
    # 1-M <= M(N-1) => 1 <= MN => 1 <= sqrt(N) => N >= 1. True.
    # So the optimal solution has one peak of height 1/sqrt(N) and N-1 valleys of height (1-1/sqrt(N))/(N-1).
    # The score is g(1/sqrt(N)).
    # g(1/sqrt(N)) = 1/sqrt(N) + (1-1/sqrt(N))^2 / ((N-1)/sqrt(N))
    # = 1/sqrt(N) + (1 - 2/sqrt(N) + 1/N) * sqrt(N) / (N-1)
    # = 1/sqrt(N) + (sqrt(N) - 2 + 1/sqrt(N)) / (N-1)
    # = 1/sqrt(N) + (sqrt(N) + 1/sqrt(N) - 2) / (N-1)
    # = 1/sqrt(N) + ( (N+1)/sqrt(N) - 2 ) / (N-1)
    # = 1/sqrt(N) + (N+1)/(sqrt(N)(N-1)) - 2/(N-1)
    # = 1/sqrt(N) + (N+1)/(sqrt(N)(N-1)) - 2/(N-1)
    # For large N, this is approx 1/sqrt(N) + 1/sqrt(N) = 2/sqrt(N).
    # For N=600, 2/sqrt(600) ≈ 2/24.5 ≈ 0.0816.
    # This is much better than 1.
    
    # So the optimal structure is a single spike and a flat background.
    # We can construct this directly.
    
    M = 1.0 / np.sqrt(N)
    h = np.full(N, (1.0 - M) / (N - 1))
    h[0] = M
    
    # Let's verify the score
    score = compute_score(h)
    
    # We can try to optimize this further?
    # The derivation assumed we can choose any h.
    # The problem allows any non-negative step function.
    # So this should be the global optimum for the continuous relaxation?
    # Wait, the "step function with 600 equal intervals" means h is a vector of 600 values.
    # My derivation holds for any vector.
    # Is there a better configuration?
    # What if we have k spikes of height M?
    # Sum h = kM + (N-k)L = 1.
    # Minimize sum h^2 = kM^2 + (N-k)L^2.
    # Score = (kM^2 + (N-k)L^2) / (1*M) = kM + (N-k)L^2/M.
    # L = (1-kM)/(N-k).
    # Let's check k=1 vs k=2.
    # For k=1, M=1/sqrt(N). Score ≈ 2/sqrt(N).
    # For k=2, if we set M such that L=M (uniform), Score=1.
    # If we make L small?
    # Let's optimize M for k=2.
    # f(M) = 2M + (N-2) ((1-2M)/(N-2))^2 / M
    # = 2M + (1-2M)^2 / ((N-2)M)
    # f'(M) = 2 + [ 2(1-2M)(-2)M - (1-2M)^2 ] / ((N-2)M^2)
    # = 2 - (1-2M)[4M + 1-2M] / ((N-2)M^2)
    # = 2 - (1-2M)(1+2M) / ((N-2)M^2)
    # = 2 - (1-4M^2) / ((N-2)M^2)
    # Set to 0: 2(N-2)M^2 = 1 - 4M^2
    # 2NM - 4M^2 = 1 - 4M^2
    # 2NM = 1 => M = 1/(2N).
    # Then L = (1 - 2/(2N))/(N-2) = (1-1/N)/(N-2) ≈ 1/N.
    # Score = 2(1/2N) + (1-1/N)^2 / ((N-2)(1/2N))
    # = 1/N + 2N(1-1/N)^2 / (N-2)
    # ≈ 1/N + 2N / N = 1/N + 2 ≈ 2.
    # This is worse than 2/sqrt(N) for large N.
    
    # So k=1 is better.
    
    # Let's just return the k=1 solution.
    # h[0] = 1/sqrt(N), rest = (1-1/sqrt(N))/(N-1).
    
    return list(h)
