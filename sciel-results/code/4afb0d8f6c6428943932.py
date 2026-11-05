import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    def value(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))
    
    # The strategy is to minimize the L2 norm of the autoconvolution c = h*h.
    # This is a Quadratic Program (QP) because the objective is quadratic in h.
    # We fix sum(h) = 1 to handle scale invariance.
    # The objective is ||c||_2^2 = sum_k c[k]^2.
    # c[k] = sum_i h[i] h[i+k-1] (with appropriate index shifts for full convolution).
    # Note: c is a Toeplitz matrix applied to h? No, c is the convolution of h with h.
    # c = h * h.
    # ||c||_2^2 = ||h * h||_2^2.
    # This is a quartic function of h, not quadratic.
    # Wait, the directive says: "Formulate the problem as a convex Quadratic Program (QP) that minimizes the squared L2 norm of the autoconvolution c = h * h".
    # Minimizing ||c||_2^2 is minimizing a quartic. This is not a QP.
    # However, if we fix the spectrum, or if we approximate?
    # Or maybe the directive means minimizing the L2 norm of c, i.e., ||c||_2?
    # Minimizing ||c||_2 is minimizing sqrt(sum c[k]^2), which is the same as minimizing sum c[k]^2.
    # This is still quartic.
    
    # Let's re-read carefully: "minimizes the squared L2 norm of the autoconvolution c = h * h".
    # This is definitely quartic.
    # However, there is a known result: minimizing the energy of the autocorrelation is related to minimizing the L2 norm of the signal in the frequency domain?
    # By Parseval's theorem: ||c||_2^2 = (1/N) ||C||_2^2, where C is the DFT of c.
    # C(f) = H(f) * H(f) = H(f)^2 (since c = h*h, in freq domain it's product).
    # So ||c||_2^2 = (1/N) sum_f |H(f)|^4.
    # We want to minimize sum |H(f)|^4 subject to sum h[i] = 1 (which means H(0) = 1).
    # This is a convex problem in the frequency domain magnitudes?
    # Let x_f = |H(f)|^2. Then we minimize sum x_f^2 subject to x_0 = 1 and x_f >= 0.
    # The minimum is achieved when x_f is as equal as possible?
    # No, x_0 is fixed at 1. The other x_f should be 0 to minimize the sum of squares?
    # If x_f = 0 for f != 0, then H(f) is a delta at 0, which means h is constant.
    # h = 1/N.
    # Then c is a triangle. ||c||_2^2 is large.
    # Wait, if H(f) is a delta at 0, then h is constant.
    # If we make H(f) flat (white noise), then x_f is constant.
    # Let's check the objective again.
    # We want to minimize the peak of c, not the L2 norm.
    # The directive suggests minimizing the L2 norm as a proxy for the peak.
    # This is a valid heuristic: if the total energy is low, the peak is likely lower.
    # But minimizing sum |H(f)|^4 with H(0)=1 leads to H(f)=0 for f!=0, which is the constant signal.
    # The constant signal has value 2.0.
    # So this direction is bad.
    
    # Let's try a different interpretation.
    # Maybe minimize the L2 norm of c subject to sum(h)=1 AND some other constraint?
    # Or maybe the QP is on a different variable?
    
    # Let's look at the inspirations.
    # Inspiration 2: "exploit scale invariance + nonnegativity".
    # Inspiration 3: "phase-locked loop structure to enforce spectral flatness".
    
    # Let's try to enforce spectral flatness directly.
    # We want |H(f)| to be constant for f != 0, and H(0) = 1.
    # This is the "flat spectrum" idea.
    # Let H(f) = A for f != 0, and H(0) = 1.
    # By Parseval: sum |H(f)|^2 = N * sum h[i]^2.
    # 1 + (N-1) A^2 = N * sum h[i]^2.
    # This doesn't fix A.
    # But if we want the time-domain autocorrelation to be flat (except at lag 0),
    # we want c[k] to be constant for k != 0.
    # c[k] = sum_i h[i] h[i+k].
    # If c[k] = B for k != 0 and c[0] = A,
    # then the DFT of c is A + B * (N-1) at freq 0, and 0 elsewhere?
    # No, DFT of c[k] (k=-N+1..N-1) is |H(f)|^2.
    # If c[k] is a constant B for k!=0 and A at k=0,
    # |H(f)|^2 = A + B * sum_{k!=0} e^{-i 2 pi f k / N} = A + B * (-1) = A - B.
    # So |H(f)|^2 is constant for all f.
    # This means h has a flat power spectrum.
    # Such sequences are called "Barker sequences" (for binary) or more generally, sequences with flat autocorrelation.
    # For real-valued sequences, we can construct them using inverse DFT of a flat spectrum.
    # Let |H(f)|^2 = C for all f.
    # Then H(0)^2 = C => H(0) = sqrt(C) => sum h[i] = sqrt(C).
    # We can normalize sum h[i] = 1 => C = 1.
    # So we want |H(f)| = 1 for all f.
    # This is a "unimodular" sequence in frequency domain.
    # We can choose random phases for H(f), enforce Hermitian symmetry, and IFFT.
    # This gives a real signal h with |H(f)| = 1 for all f.
    # Then c[k] = IFFT(|H(f)|^2) = IFFT(1) = delta[k].
    # Wait, if |H(f)|^2 = 1, then c[k] = delta[k].
    # c[0] = 1, c[k] = 0 for k != 0.
    # Then max(c) = 1.
    # value = 2 * N * 1 / 1^2 = 2N = 800.
    # This is terrible!
    
    # Why? Because the peak is at lag 0, which is c[0] = sum h[i]^2.
    # If |H(f)| = 1, then sum h[i]^2 = (1/N) sum |H(f)|^2 = (1/N) * N = 1.
    # So c[0] = 1.
    # But for the constant signal h=1/N, c[0] = N*(1/N)^2 = 1/N.
    # And sum h[i] = 1.
    # value = 2N * (1/N) / 1 = 2.
    # So the constant signal is much better.
    # The problem is that the peak is always at lag 0 for positive signals.
    # To reduce the peak, we need to make c[0] small.
    # c[0] = sum h[i]^2.
    # By Cauchy-Schwarz, (sum h[i])^2 <= N * sum h[i]^2.
    # So sum h[i]^2 >= (sum h[i])^2 / N.
    # The minimum is achieved when h is constant.
    # So the constant signal minimizes c[0].
    # But the peak might not be at lag 0 if h has negative parts.
    # If h has negative parts, c[0] can be smaller? No, c[0] is sum of squares, always positive.
    # But the maximum of c[k] might be at k != 0 if c[0] is small and some c[k] is large?
    # No, c[0] is usually the largest because it's the sum of squares.
    # Unless the signal is oscillatory.
    # If h is a high-frequency sine wave, c[0] is large, but c[k] for k near the period might be large too.
    # Actually, for a sine wave, c[k] is a cosine wave with the same frequency.
    # The peak is at k=0 and k=period.
    # So the peak is still c[0].
    
    # So it seems hard to beat the constant signal if we only look at the peak.
    # But the seed is 2.0, and we have 1.04.
    # How did we get 1.04?
    # The GA found a signal with value 1.12, and now 1.04.
    # This implies that the peak is NOT at lag 0 for these signals.
    # How can the peak be at k != 0?
    # If h has a "hole" in the middle, c[0] might be reduced?
    # No, c[0] is sum of squares.
    # Wait, the convolution is defined as np.convolve(h, h).
    # This is the full convolution.
    # c[k] = sum_i h[i] h[i+k-1].
    # The index k goes from 0 to 2N-2.
    # The center is at k = N-1.
    # c[N-1] = sum_i h[i] h[i] = sum h[i]^2.
    # So the peak is usually at the center.
    # But if the signal is anti-symmetric?
    # If h[i] = -h[N-1-i], then c[N-1] = sum h[i]^2 > 0.
    # But c[0] = h[0]^2.
    # If h is mostly zero in the middle, c[N-1] might be small?
    # No, c[N-1] is the sum of squares of all elements.
    # It cannot be small if sum h[i] = 1.
    # Minimum c[N-1] is 1/N.
    
    # So the peak is always at least 1/N.
    # value >= 2N * (1/N) / 1 = 2.
    # So the lower bound is 2.0?
    # But we have 1.04.
    # This implies that my understanding of the problem is wrong.
    # Let's re-read the problem statement.
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # "The maximum is taken over its signed entries, with no absolute value inside np.max."
    # Ah! "signed entries".
    # So we take the maximum of the signed values.
    # If all c[k] are negative, the maximum is the least negative one.
    # If c[k] can be negative, the maximum might be much smaller than c[0] (which is positive).
    # c[0] = h[0]^2 >= 0.
    # c[N-1] = sum h[i]^2 > 0.
    # So there are always positive entries.
    # The maximum must be positive.
    # So the lower bound is still 2.0?
    
    # Let's check the seed: np.ones(400).
    # h = [1, 1, ..., 1].
    # sum(h) = 400.
    # c = np.convolve(h, h) = [1, 2, ..., 400, ..., 2, 1].
    # max(c) = 400.
    # value = 2 * 400 * 400 / 400^2 = 80000 / 160000 = 0.5?
    # Wait.
    # 2 * N * max(c) / sum(h)^2
    # = 2 * 400 * 400 / 160000
    # = 320000 / 160000 = 2.0.
    # Yes, it is 2.0.
    
    # So how can we get 1.04?
    # We need max(c) / sum(h)^2 to be smaller.
    # Since sum(h) is in the denominator, we want a large sum(h).
    # But we can scale h. The value is scale-invariant.
    # So we fix sum(h) = 1.
    # Then value = 2N * max(c).
    # We want to minimize max(c).
    # We know c[N-1] = sum h[i]^2 >= 1/N = 0.0025.
    # So max(c) >= 0.0025.
    # value >= 2 * 400 * 0.0025 = 2.0.
    
    # So it is impossible to get a value less than 2.0?
    # But the score is 1.04, which corresponds to value = 2.0 / 1.04 = 1.92.
    # This is less than 2.0.
    # So my lower bound analysis must be wrong.
    # Where is the mistake?
    # "c[N-1] = sum h[i]^2" is correct for the center of the full convolution.
    # Is it possible that max(c) is not at the center?
    # Yes, if c[center] is small and some other c[k] is larger?
    # No, c[center] is the sum of squares, which is the largest possible value for the autocorrelation of a real signal?
    # Actually, for a real signal, c[k] <= c[0] (if we define c[0] as sum h[i]^2).
    # In np.convolve, the center is the sum of squares.
    # So max(c) >= c[center] = sum h[i]^2.
    # And sum h[i]^2 >= (sum h[i])^2 / N.
    # So max(c) >= (sum h[i])^2 / N.
    # value = 2N * max(c) / sum(h)^2 >= 2N * (sum h[i])^2 / (N * sum(h)^2) = 2.
    
    # So the lower bound is 2.0.
    # How is it possible to have a score of 1.04?
    # Score = seed / value = 2.0 / value.
    # If score = 1.04, value = 2.0 / 1.04 = 1.923.
    # This is < 2.0.
    # Contradiction.
    
    # Let's re-read the problem statement very carefully.
    # "Choose real, possibly negative step heights h[0], ..., h[N-1]"
    # "value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)"
    # "The maximum is taken over its signed entries, with no absolute value inside np.max."
    
    # Is it possible that np.max returns a negative number?
    # Only if all entries are negative.
    # But c[center] = sum h[i]^2 is always non-negative.
    # So max(c) is always non-negative.
    
    # Is it possible that sum(h) is negative?
    # sum(h)^2 is positive.
    
    # Is it possible that I am misunderstanding the convolution?
    # np.convolve(h, h)
    # If h = [1, 0, 0], c = [1, 0, 0]. max = 1.
    # If h = [1, 1, 1], c = [1, 2, 3, 2, 1]. max = 3.
    # If h = [1, -1, 1], c = [1, 0, 1, 0, 1]. max = 1.
    # sum(h) = 1.
    # value = 2 * 3 * 1 / 1 = 6.
    # If h = [1, -0.5, 1], sum = 1.5.
    # c = [1, 0.5, 1.25, 0.5, 1]. max = 1.25.
    # value = 2 * 3 * 1.25 / 2.25 = 7.5 / 2.25 = 3.33.
    
    # Let's try a signal with a large negative peak.
    # h = [1, 1, -1, 1, 1].
    # sum = 3.
    # c = [1, 2, 0, 2, 5, 2, 0, 2, 1]. max = 5.
    # value = 2 * 5 * 5 / 9 = 50/9 = 5.5.
    
    # It seems hard to get below 2.
    
    # Wait, the seed is 2.0.
    # The score is 1.04.
    # This means the value is 1.92.
    # How is this possible?
    # Maybe the lower bound is not 2.0?
    # Let's check the inequality: sum h[i]^2 >= (sum h[i])^2 / N.
    # This is Cauchy-Schwarz. It is always true.
    # So max(c) >= sum h[i]^2 >= (sum h[i])^2 / N.
    # So value >= 2.
    
    # Unless... the problem allows complex numbers? No, "real, possibly negative".
    # Unless... the convolution is circular? No, "full autoconvolution".
    
    # Is it possible that the "seed" value 2.0 is not the lower bound?
    # Yes, the lower bound is 2.0, and the seed achieves it.
    # So the best possible score is 1.0.
    # But the current score is 1.04, which is > 1.0.
    # This implies that the value is < 2.0.
    # This is a contradiction.
    
    # Let's re-read the score definition.
    # "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # So if score > 1.0, value < seed_value.
    # seed_value = 2.0.
    # So value < 2.0.
    
    # How can value < 2.0?
    # My proof that value >= 2.0 must be flawed.
    # Where is the flaw?
    # "max(c) >= c[center]"
    # Is c[center] the sum of squares?
    # For np.convolve(h, h), the center element is indeed sum h[i]^2.
    # So max(c) >= sum h[i]^2.
    # And sum h[i]^2 >= (sum h[i])^2 / N.
    # So value = 2N * max(c) / sum(h)^2 >= 2N * (sum h[i])^2 / (N * sum(h)^2) = 2.
    
    # Is it possible that sum(h) is complex? No.
    # Is it possible that N is not 400? No.
    
    # Let's test with a small N.
    # N=2. h=[1, 0]. sum=1. c=[1, 0, 0]. max=1. value = 2*2*1/1 = 4.
    # h=[1, 1]. sum=2. c=[1, 2, 1]. max=2. value = 2*2*2/4 = 2.
    # h=[1, -1]. sum=0. Invalid.
    # h=[1, 0.5]. sum=1.5. c=[1, 1, 0.25]. max=1. value = 2*2*1/2.25 = 4/2.25 = 1.77.
    # Ah! 1.77 < 2.
    # So the lower bound is NOT 2.
    # My Cauchy-Schwarz application was correct, but I assumed max(c) = c[center].
    # In the example h=[1, 0.5], c[center]=c[1]=1.
    # sum h[i]^2 = 1 + 0.25 = 1.25.
    # Wait, c[1] = h[0]h[1] + h[1]h[0] = 2*0.5 = 1.
    # But sum h[i]^2 = 1.25.
    # So c[center] != sum h[i]^2?
    # Let's check np.convolve([1, 0.5], [1, 0.5]).
    # [1*1, 1*0.5 + 0.5*1, 0.5*0.5] = [1, 1, 0.25].
    # The center is index 1. Value is 1.
    # But sum h[i]^2 = 1.25.
    # Why is c[center] not sum h[i]^2?
    # Because the full convolution of two vectors of length N has length 2N-1.
    # The center is at index N-1.
    # c[N-1] = sum_{i=0}^{N-1} h[i] h[i+N-1-(N-1)]? No.
    # c[k] = sum_i h[i] h[k-i].
    # For k=N-1: c[N-1] = sum_i h[i] h[N-1-i].
    # This is NOT sum h[i]^2 unless h is symmetric.
    # Ah! I made a mistake.
    # The autocorrelation is usually defined as sum h[i] h[i+k].
    # np.convolve(h, h) is the convolution, which is sum h[i] h[k-i].
    # These are different!
    # Convolution reverses one of the signals.
    # So c[k] = sum h[i] h[k-i].
    # The peak is not necessarily at the center.
    # And c[center] is not sum h[i]^2.
    # sum h[i]^2 is c[0] if we define autocorrelation as sum h[i] h[i+k] with k=0?
    # No, for convolution, c[0] = h[0]^2.
    # The sum of squares is sum h[i]^2.
    # Where is this in the convolution?
    # It is not a single element.
    
    # So the lower bound analysis is completely wrong.
    # We can have max(c) < (sum h[i])^2 / N.
    # In the example, max(c) = 1, (sum h[i])^2 / N = 2.25 / 2 = 1.125.
    # 1 < 1.125.
    # So value can be less than 2.
    
    # Now, back to the strategy.
    # We want to minimize max(c).
    # c = h * h (convolution).
    # This is equivalent to minimizing the peak of the convolution of h with itself.
    # This is related to "peak-to-sidelobe ratio" or "minimum peak autocorrelation".
    # But it's convolution, not autocorrelation.
    # However, for real h, h*h is symmetric.
    # And if we reverse h, the convolution is the same.
    # So we can assume h is symmetric?
    # If h is symmetric, h[i] = h[N-1-i].
    # Then c[k] = sum h[i] h[k-i].
    # This is the autocorrelation (with a shift).
    # So for symmetric h, the problem is to minimize the peak of the autocorrelation.
    # And we know that the peak of the autocorrelation is at lag 0, which is sum h[i]^2.
    # So for symmetric h, max(c) = sum h[i]^2.
    # And we are back to the lower bound of 2.0.
    
    # So to beat 2.0, h must be ASYMMETRIC.
    # The current program (score 1.04) likely uses an asymmetric h.
    # The GA found a "center bump, negative sides" strategy.
    # This is asymmetric?
    # If it's a bump in the center, it might be symmetric.
    # But if the sides are negative, it might be asymmetric.
    
    # Let's try to construct an asymmetric h that minimizes max(c).
    # The directive suggests a QP.
    # Let's try to minimize the L2 norm of c.
    # This is a quartic problem.
    # But we can use gradient descent.
    # The gradient of ||c||_2^2 with respect to h is 4 * (c * h) ?
    # Let's derive it.
    # c = h * h.
    # ||c||_2^2 = sum_k c[k]^2.
    # d/dc_k (c[k]^2) = 2 c[k].
    # dc[k]/dh[i] = h[k-i].
    # So d/dh[i] ||c||_2^2 = sum_k 2 c[k] h[k-i] = 2 (c * h)[i].
    # So the gradient is 2 * (c * h).
    # We can use this for gradient descent.
    
    # We want to minimize ||c||_2^2 subject to sum(h) = 1.
    # We can use the method of Lagrange multipliers.
    # L = ||c||_2^2 - lambda (sum(h) - 1).
    # dL/dh[i] = 2 (c * h)[i] - lambda = 0.
    # So (c * h)[i] = lambda / 2 for all i.
    # This means c * h is constant.
    # This is a fixed-point equation.
    # We can solve it by iteration.
    # h_{new} = h_old - eta * (2 (c * h) - lambda).
    # And adjust lambda to maintain sum(h) = 1.
    
    # Let's implement this.
    
    def score(h):
        s = np.sum(h)
        if s == 0 or not np.isfinite(s):
            return float('inf')
        z = np.convolve(h, h)
        return abs(2 * N * np.max(z) / (s * s))
    
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Initialize h
    h = np.ones(N) / N  # sum = 1
    h += np.random.randn(N) * 0.1
    h /= np.sum(h)
    
    eta = 0.1
    for i in range(5000):
        if time.time() > deadline:
            break
        c = np.convolve(h, h)
        grad = 2 * np.convolve(c, h)  # This is not quite right, c*h is convolution of length 3N-2
        # We need the gradient with respect to h of length N.
        # The gradient is the central part of c*h?
        # Let's check the dimensions.
        # c has length 2N-1.
        # h has length N.
        # c*h has length 3N-2.
        # The gradient should have length N.
        # The term (c * h)[i] = sum_k c[k] h[k-i].
        # This is valid for i=0..N-1.
        # So we take the first N elements of the convolution?
        # No, the convolution is full.
        # We need to align the indices correctly.
        # Actually, the gradient of sum_k c[k]^2 with respect to h[i] is 2 * sum_k c[k] h[k-i].
        # This is the cross-correlation of c and h, or convolution?
        # It is (c * h)[i] if we define convolution as sum c[k] h[k-i].
        # So we can compute g = np.convolve(c, h) and take the appropriate slice.
        # The slice should be such that the indices match.
        # For i=0, g[0] = c[0]h[0].
        # For i=N-1, g[N-1] = sum c[k] h[k-(N-1)].
        # This seems correct.
        
        g = np.convolve(c, h)
        # The gradient is the central part?
        # No, the indices i go from 0 to N-1.
        # The convolution g has indices 0 to 3N-2.
        # The term for h[i] is g[i + (N-1)]?
        # Let's test with N=2.
        # h = [h0, h1]. c = [h0^2, 2h0h1, h1^2].
        # grad[0] = 2 * (c[0]h[0] + c[1]h[1]) = 2 * (h0^3 + 2h0^2 h1).
        # g = c * h = [h0^3, 2h0^2 h1, 2h0 h1^2, h1^3].
        # g[0] = h0^3.
        # g[1] = 2h0^2 h1.
        # So grad[0] = 2 * (g[0] + g[1]).
        # This is not a simple slice.
        
        # Let's use a more robust method: finite differences or automatic differentiation?
        # Or just use the FFT-based gradient.
        # The gradient of ||c||_2^2 is 4 * IFFT(FFT(c) * FFT(h))?
        # No, c = h*h => C = H^2.
        # ||c||_2^2 = (1/N) ||C||_2^2 = (1/N) sum |H(f)|^4.
        # d/dh (1/N sum |H|^4) = (1/N) sum 4 |H|^2 H' .
        # This is getting complicated.
        
        # Let's just use a simple numerical gradient or a library.
        # Or, let's try a different approach: minimize the peak directly using a smooth surrogate.
        # max(c) ~ log(sum exp(beta * c)).
        # This is a smooth function.
        # We can minimize this using gradient descent.
        
        beta = 10
        def smooth_max(c):
            return np.log(np.sum(np.exp(beta * c))) / beta
        
        # We want to minimize smooth_max(c) subject to sum(h)=1.
        # We can use scipy.optimize.minimize with SLSQP.
        
        def objective(h_flat):
            h = h_flat.reshape(N)
            s = np.sum(h)
            if s == 0:
                return 1e10
            c = np.convolve(h, h)
            return smooth_max(c)
        
        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1}]
        
        h0 = h.copy()
        h0 /= np.sum(h0)
        
        try:
            res = minimize(objective, h0, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                h_new = res.x
                h_new /= np.sum(h_new)
                s_new = score(h_new)
                if s_new < best_score:
                    best_score = s_new
                    best_h = h_new.copy()
                    h = h_new.copy()
        except Exception as e:
            pass
        
    return best_h
