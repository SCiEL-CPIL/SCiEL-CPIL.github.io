import numpy as np
import time

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit

    def compute_value(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))

    best_h = np.ones(N)
    best_val = compute_value(best_h)  # Should be 2.0

    # Strategy: The key insight is that we want to minimize max(z) / s^2.
    # By making h non-negative and using a "top-hat" like structure, we can approach 2.0.
    # But to beat 2.0, we need to exploit the "signed max" property.
    # If we can make the largest entry in the autocorrelation negative, abs() flips it.
    # However, for a sum-positive signal, the zero-lag term h*h is positive and large.
    # Unless... we scale things such that the zero-lag term is not the max?
    # No, zero-lag is sum(h_i^2).
    # For a top-hat of height 1, width W, s=W. max(z)=W. val = 2*N*W/W^2 = 2*N/W.
    # Minimized when W is max. W<=N. So val >= 2.
    
    # What if we use a "double-pulse" structure?
    # h = [1]*W + [0]*(N-2W) + [1]*W
    # s = 2W.
    # z(0) = 2W.
    # z(W) = W (overlap of first pulse with second? No, they are separated).
    # If separation is large enough, z(W) is 0.
    # Max(z) is likely z(0) = 2W.
    # val = 2*N*2W / (2W)^2 = 4NW / 4W^2 = N/W.
    # Same bound.

    # What if we use negative heights?
    # h = [1]*W - [1]*W
    # s = 0. Invalid.
    # h = [1]*W - [0.5]*W
    # s = 0.5W.
    # This is just a scaled version of [2]*W - [1]*W.
    # Let's try to find a structure where the autocorrelation has a large negative peak
    # that exceeds the positive peaks in magnitude? No, max() takes the largest value.
    # If max(z) is negative, then abs(max(z)) = -max(z).
    # Can we make max(z) negative?
    # z(0) = sum(h^2) is always positive.
    # So max(z) is always positive (at least z(0)).
    # Therefore, we are always minimizing 2*N*max(z)/s^2.
    # And since z(0) is always in z, max(z) >= z(0).
    # So val >= 2*N*z(0)/s^2 = 2*N*sum(h^2)/sum(h)^2.
    # By Cauchy-Schwarz, sum(h^2)*N >= sum(h)^2 => sum(h^2)/sum(h)^2 >= 1/N.
    # So val >= 2*N*(1/N) = 2.
    # The minimum possible value is 2.0, achieved when h is constant.
    # Wait, is that true?
    # If h is constant, z(0) = N*c^2. s = N*c. s^2 = N^2*c^2.
    # z(0) = N*c^2.
    # max(z) = N*c^2 (since all lags are N*c^2 * (1 - k/N) ... wait).
    # For h = c * ones(N), z(k) = c^2 * (N - k).
    # Max is at k=0: c^2 * N.
    # val = 2*N * (c^2*N) / (N*c)^2 = 2*N^2*c^2 / (N^2*c^2) = 2.
    
    # So the theoretical lower bound is 2.0?
    # Let's check the seed. Seed is np.ones(400). Value is 2.0.
    # The task says "seed np.ones(400) gives 2.0".
    # And "lower is better".
    # If 2.0 is the lower bound, how can we improve?
    # Is my bound tight?
    # val = 2 * N * max(z) / s^2.
    # max(z) >= z(0) = sum(h^2).
    # val >= 2 * N * sum(h^2) / s^2.
    # Let h_i = 1 for i in S, 0 otherwise. |S|=k.
    # s=k. sum(h^2)=k.
    # val >= 2*N*k/k^2 = 2*N/k.
    # To minimize, maximize k. k<=N. min val = 2.
    
    # What if h has negative values?
    # Let h = [1, -1, 1, -1, ...]
    # s = 0 (if N even). Invalid.
    # Let h = [1, -0.5, 1, -0.5, ...]
    # s = 0.5 * N/2 = N/4.
    # sum(h^2) = N/2 * 1 + N/2 * 0.25 = 0.625 N.
    # Lower bound from C-S: 2*N*0.625N / (N/4)^2 = 2*N*0.625N / (N^2/16) = 1.25 * 16 = 20.
    # Much worse.
    
    # It seems 2.0 is the global minimum?
    # If so, the current program scoring 1.1758 (which means value ~1.7?) is impossible?
    # Wait. The score is seed/value.
    # Seed value is 2.0.
    # If program value is V, score = 2.0 / V.
    # If score is 1.1758, then V = 2.0 / 1.1758 ≈ 1.701.
    # This implies V < 2.0.
    # My derivation says V >= 2.0.
    # Where is the error?
    # "The evaluator computes exactly this value... value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # My C-S inequality: sum(h^2) / sum(h)^2 >= 1/N.
    # This holds for real h.
    # So val >= 2.
    # How can V be 1.7?
    # Maybe I am misinterpreting "np.max(np.convolve(h, h))".
    # Convolve is full length 2N-1.
    # z(k) = sum_{i} h[i] h[i-k].
    # z(0) = sum h[i]^2.
    # Is it possible that max(z) is NOT z(0)?
    # If h has negative values, z(0) is still sum squares, so positive.
    # Could some other lag be larger?
    # z(k) = sum h[i] h[i-k].
    # If h is positive, z(k) <= z(0) is not necessarily true?
    # For h >= 0, z(0) - z(k) = sum (h[i]^2 - h[i]h[i-k])? No.
    # z(0) = sum h[i]^2.
    # z(k) = sum h[i] h[i-k].
    # By Cauchy-Schwarz, z(k)^2 <= z(0) z(2k)? No.
    # Actually, for positive h, z(0) is the maximum of the autocorrelation.
    # Proof: z(0) - z(k) = sum_{i} (h[i]^2 - h[i]h[i-k]) ... this doesn't look obviously positive.
    # Let's test with a small example.
    # h = [1, 0, 1]. N=3.
    # z = [1, 0, 2, 0, 1]. max=2. z(0)=2.
    # h = [1, 1, 0]. N=3.
    # z = [1, 2, 1, 0, 0]. max=2. z(0)=2.
    # h = [1, 2, 1]. N=3.
    # z = [1, 2, 6, 2, 1]. max=6. z(0)=1+4+1=6.
    # h = [1, -1, 1]. N=3.
    # z = [1, -2, 3, -2, 1]. max=3. z(0)=3.
    # h = [1, -2, 1]. N=3.
    # z = [1, -4, 5, -4, 1]. max=5. z(0)=1+4+1=6.
    # Here max(z) = 5 < z(0) = 6.
    # So for signed h, max(z) CAN be less than z(0).
    
    # Let's re-evaluate the bound.
    # val = 2 * N * max(z) / s^2.
    # We want to minimize max(z) / s^2.
    # We know z(0) = sum h^2.
    # If we can make max(z) < z(0), we might beat 2.0.
    # In the example h=[1, -2, 1], N=3.
    # s = 0. Invalid.
    # Let's try to make s non-zero.
    # h = [1, -2, 1, eps].
    # This is getting complicated.
    
    # Let's look at the structure of h=[1, -2, 1].
    # It's a second difference.
    # General idea: use a high-pass filter shape to cancel out the low-frequency content (which contributes to large lags in autocorrelation) while maintaining a non-zero sum.
    
    # Let's try optimizing a parametric family that allows signed values and has a "dip".
    # Family: h(x) = A * exp(-((x-c)^2)/w^2) - B * exp(-((x-c)^2)/(w/2)^2)
    # Or polynomial: h(x) = 1 - 2*(2x-1)^2 ? (Parabola opening down)
    # h = [1, 0, -1] (discrete).
    # s=0.
    # h = [1, 0, -1, eps].
    
    # Let's use scipy.optimize to directly minimize the value over a few parameters of a signed bump.
    # Since direct optimization over 400 dims is hard, we parameterize.
    
    from scipy.optimize import minimize

    def generate_h(params):
        # params: [amp_pos, width_pos, amp_neg, width_neg, shift]
        amp_p, w_p, amp_n, w_n, shift = params
        x = np.linspace(-0.5, 0.5, N)
        h = amp_p * np.exp(-((x-shift)**2) / (2*w_p**2))
        h -= amp_n * np.exp(-((x-shift)**2) / (2*w_n**2))
        # Ensure sum is not zero. If it is close to zero, we have a problem.
        # But the optimizer will avoid s=0 if it checks.
        return h

    def objective(params):
        if time.time() > deadline - 5:
            return float('inf')
        try:
            h = generate_h(params)
            s = np.sum(h)
            if abs(s) < 1e-10:
                return float('inf')
            return compute_value(h)
        except:
            return float('inf')

    # Initial guesses
    # We want a dip in the middle.
    # Positive broad peak, negative narrow peak.
    inits = [
        [1.0, 0.1, 2.0, 0.05, 0.0], # Broad pos, narrow neg
        [1.0, 0.2, 3.0, 0.05, 0.0],
        [2.0, 0.1, 4.0, 0.05, 0.0],
        [1.0, 0.1, 2.0, 0.1, 0.1], # Shifted
        [1.0, 0.1, 2.0, 0.1, -0.1],
        [1.0, 0.2, 1.5, 0.1, 0.0],
    ]
    
    for p0 in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, p0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_h = generate_h(res.x)
        except:
            pass
            
    return best_h
