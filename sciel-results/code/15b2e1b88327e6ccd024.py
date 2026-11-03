"""
Signed autocorrelation maximum absolute-value variant.

We seek a signed step function f (length N) that minimizes:
    max_k | (f * f)[k] / (sum f)^2 |
where * is circular autoconvolution (or linear, depending on interpretation).

The current seed returns np.ones(N), which gives:
    autoconvolution peak = N (at lag 0)
    sum(f)^2 = N^2
    ratio = N / N^2 = 1/N

But we can do better by using signs to cancel the peak.

Key insight: If we use a Rademacher-like sequence (±1), the autoconvolution at lag 0 is always N (since f[i]^2=1).
But at other lags, the autoconvolution can be small if the sequence has low autocorrelation.

However, the denominator is (sum f)^2. If we use balanced ±1, sum f ≈ 0, making the denominator tiny and the ratio huge.

So we need to balance:
- Keep sum(f) large (to make denominator large)
- Keep max |autoconv| small (to make numerator small)

One approach: Use a sequence that is mostly +1 but with some -1 placed to reduce peak autoconvolution.

Another approach: Use a sequence based on a low-autocorrelation binary sequence (like m-sequences or Barker codes) but scaled to have large sum.

Actually, let's think differently. The objective is:
    max_k |C[k]| / S^2
where C[k] = sum_i f[i] f[i-k] (circular) and S = sum f[i].

If we set f[i] = a + b * g[i] where g[i] ∈ {-1, 1} is a low-autocorrelation sequence, and a > |b|, then:
- S = N*a + b*sum(g)
- C[k] = sum_i (a + b*g[i])(a + b*g[i-k])
       = a^2*N + a*b*sum(g[i] + g[i-k]) + b^2*sum(g[i]g[i-k])
       = a^2*N + a*b*(sum(g) + sum(g[i-k])) + b^2*C_g[k]
       = a^2*N + 2*a*b*sum(g) + b^2*C_g[k]   [since circular sum of g[i-k] = sum(g)]

If sum(g) = 0 (balanced), then:
    C[k] = a^2*N + b^2*C_g[k]
    S = N*a

So ratio = max_k |a^2*N + b^2*C_g[k]| / (N*a)^2

If we choose b small relative to a, then C[k] ≈ a^2*N, and ratio ≈ a^2*N / (a^2*N^2) = 1/N.

But if we can make C_g[k] negative enough to cancel a^2*N at some k, we could reduce the max.

However, the peak at k=0 is C[0] = a^2*N + b^2*N = N*(a^2 + b^2).
At other k, C[k] = a^2*N + b^2*C_g[k].

If C_g[k] can be as low as -N (in the best case for a random sequence, C_g[k] ~ O(sqrt(N))), then:
    C[k] ≈ a^2*N - b^2*sqrt(N) for k≠0.

The max will likely be at k=0: N*(a^2 + b^2).

So ratio ≈ N*(a^2 + b^2) / (N^2 * a^2) = (a^2 + b^2) / (N * a^2) = (1 + (b/a)^2) / N.

This is worse than 1/N when b≠0. So adding a low-autocorrelation component with zero mean actually increases the peak.

What if we don't require sum(g)=0? Let's reconsider.

Alternative approach: Use a sequence that is constant except for a few sign flips to create destructive interference in the autoconvolution.

Actually, let's try a different family: use a sequence that is +1 for most entries, but -1 in a pattern that reduces the autoconvolution peak.

For N=400, the seed gives score = 1/400 = 0.0025.

Can we do better? The theoretical lower bound for max |autoconv| / S^2 is 1/N (achieved by constant sequence). So we cannot do better than 1/N if we require f[i] ≥ 0. But we allow signed values.

Wait, the problem says "signed step heights", so f[i] can be negative.

If we allow negative values, can we make the autoconvolution peak smaller relative to S^2?

Consider f[i] = 1 for i even, -1 for i odd. Then S = 0, denominator is 0. Bad.

Consider f[i] = 1 for i < N/2, -1 for i ≥ N/2. Then S = 0. Bad.

We need S ≠ 0.

Let f[i] = 1 + ε*h[i] where h[i] ∈ {-1, 1} and ε < 1.

If h has low autocorrelation, maybe we can reduce the peak.

But as shown above, if sum(h)=0, the k=0 peak increases.

What if we use a different structure? Let's try using a sequence based on a chirp or linear congruential generator to get low peak-to-average ratio.

Actually, let's just try a simple heuristic: start with all +1, and iteratively flip the sign of the element that most reduces the current max autoconvolution / S^2.

Given the time constraint, let's implement a simple local search.

But N=400, and computing autoconvolution is O(N^2) per evaluation, which is 160,000 operations. With 150s deadline, we can do many iterations.

Let's implement:
1. Start with f = np.ones(N)
2. Compute initial score
3. For each i, try flipping f[i], compute new score, keep the best flip
4. Repeat until no improvement or time runs out

This is a simple coordinate descent.

To speed up, we can use FFT for autoconvolution.

Let's implement this.
"""

import numpy as np
import time

N = 400

def compute_score(f):
    """Compute max |autoconv| / (sum f)^2"""
    S = np.sum(f)
    if abs(S) < 1e-10:
        return np.inf
    # Circular autoconvolution via FFT
    F = np.fft.fft(f)
    C = np.fft.ifft(F * np.conj(F)).real
    max_abs = np.max(np.abs(C))
    return max_abs / (S * S)

def construct():
    deadline = time.time() + 145  # 145s to be safe
    
    # Initialize with all ones
    f = np.ones(N)
    best_score = compute_score(f)
    
    # Local search: coordinate descent
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() >= deadline:
                break
            # Try flipping sign of f[i]
            old_val = f[i]
            f[i] = -f[i]
            new_score = compute_score(f)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                f[i] = old_val
    
    return f

# EVOLVE-BLOCK-END
