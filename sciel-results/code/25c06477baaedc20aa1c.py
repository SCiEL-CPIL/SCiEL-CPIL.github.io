"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Key structural insight from inspirations:
- The DFT parameterization h_k = |P(omega^k)|^2 (or similar spectral representations)
  naturally enforces non-negativity and has been shown to score ~1.22.
- The 'spike' solution (narrow support, high amplitude) yields ratio ~100 in some
  formulations, but that may be an artifact of continuous vs discrete.
- For a rectangular window of length L with height A:
  - ||h||_2^2 = L * A^2
  - ||h||_1 = L * A
  - ||h*h||_inf = L * A^2 (peak of triangular autoconvolution)
  - Ratio = (L*A^2) / (L*A * L*A^2) = 1/L
  So shorter rectangles give higher ratio. The minimum L=1 gives ratio=1.

But wait - let me reconsider. For a single spike h = [0,...,A,...,0]:
  - ||h||_2^2 = A^2
  - ||h||_1 = A
  - autoconv: only one nonzero term = A^2
  - Ratio = A^2 / (A * A^2) = 1/A

So as A -> infinity, ratio -> 0. As A -> 1, ratio -> 1. That's not great.

Let me reconsider the discrete convolution more carefully.
If h has N entries, conv = np.convolve(h, h, 'full') has 2N-1 entries.
For h = [A] (N=1): conv = [A^2], peak = A^2, ratio = A^2/(A*A^2) = 1/A.

For h = [A, 0, 0, ..., 0] (N=50): conv has 2N-1=99 entries, only conv[0]=A^2 is nonzero.
peak = A^2, ratio = A^2/(A*A^2) = 1/A.

Hmm, so a single spike gives 1/A which is maximized at A=1, giving ratio=1.

What about two spikes? h = [A, 0, ..., 0, A] (spikes at positions 0 and N-1).
conv[0] = A^2, conv[N-1] = 2*A^2 (cross terms), conv[2N-2] = A^2.
peak = 2*A^2, ||h||_2^2 = 2*A^2, ||h||_1 = 2*A.
ratio = 2*A^2 / (2*A * 2*A^2) = 1/(2*A). Worse.

What about a block of L consecutive A's?
conv is triangular: peak = L*A^2.
||h||_2^2 = L*A^2, ||h||_1 = L*A.
ratio = L*A^2 / (L*A * L*A^2) = 1/L.

So for any constant-amplitude block, ratio = 1/L, maximized at L=1, ratio=1.

The key insight from the inspirations is that the DFT parameterization allows
non-constant shapes that can do better. The ratio can exceed 1 when the shape
is not a rectangle.

Let me think about what shapes maximize this. We want to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

By Cauchy-Schwarz, ||h||_2^2 <= ||h||_1 * max(h_i) for nonnegative h... no that's not right.
Actually ||h||_2^2 = sum h_i^2 <= max(h_i) * sum h_i = max(h_i) * ||h||_1.
So ||h||_2^2 / ||h||_1 <= max(h_i).

And ||h*h||_inf >= h_0^2 (since (h*h)(0) = sum h_i * h_i = ||h||_2^2... wait no.
(h*h)(k) = sum_i h_i * h_{i+k} (with appropriate indexing for full convolution).
(h*h)(0) = sum_i h_i^2 = ||h||_2^2.

So ||h*h||_inf >= ||h||_2^2.

Therefore ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf) <= ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.

So the ratio is bounded by 1/||h||_1. To maximize it, we want ||h||_1 small.
But we also need ||h*h||_inf to not be too large relative to ||h||_2^2.

Actually, let me re-examine. The ratio is scale-invariant: if we scale h by c,
||ch||_2^2 = c^2 ||h||_2^2, ||ch||_1 = c ||h||_1, ||ch*ch||_inf = c^2 ||h*h||_inf.
Ratio = c^2 ||h||_2^2 / (c ||h||_1 * c^2 ||h*h||_inf) = ||h||_2^2 / (c ||h||_1 * ||h*h||_inf).
Wait, that's not scale invariant. Let me redo.

ratio(c*h) = ||c h||_2^2 / (||c h||_1 * ||c h * c h||_inf)
= c^2 ||h||_2^2 / (c ||h||_1 * c^2 ||h*h||_inf)
= ||h||_2^2 / (c ||h||_1 * ||h*h||_inf)
= ratio(h) / c.

So the ratio scales as 1/c. To maximize, we want c as small as possible.
But c can be arbitrarily small, making the ratio arbitrarily large!

Wait, that can't be right for a meaningful optimization problem. Let me re-read the task.

"Choose 50 nonnegative heights on equal intervals of [-1/4,1/4] to maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution."

Hmm, maybe the norms are actual integrals (continuous), not discrete sums?
"on equal intervals" suggests the heights are constant on each interval, so:
||h||_2^2 = delta * sum h_i^2 where delta = interval width = 1/50.
||h||_1 = delta * sum h_i.
||h*h||_inf = delta * max(conv) where conv is the discrete convolution (since (h*h)(t) = delta * sum h_i h_{i+k} approximately).

Wait, actually for step functions on intervals of width delta:
(h*h)(t) = integral h(s) h(t-s) ds.
If h is piecewise constant with h_i on interval [i*delta, (i+1)*delta), then
(h*h)(t) = sum_{i,j} h_i h_j * measure({s: s in I_i, t-s in I_j}).

This is more complex than simple discrete convolution. But for the purpose of this problem,
the discrete convolution approximation is standard.

Given the scale invariance issue, I think the problem must have an implicit constraint,
or the "score" is defined differently. Looking at the seed program, it just computes
l2sq / (l1 * peak) without any normalization. And the score is "value/seed" where seed
program gives 1.0.

So the seed program achieves ratio ~1.0, and we need to beat that. The scale invariance
means we can always scale up to increase the ratio... unless there's a constraint I'm missing.

Wait, re-reading: "score is the lower bound on C2". Maybe C2 is a constant in an inequality,
and the ratio gives a lower bound. The actual value might be bounded.

Let me just focus on maximizing the ratio as computed by the seed program's _ratio function,
and use the DFT parameterization from the inspirations which scored 1.22.

The DFT parameterization: h_k = |sum_j P_j * omega^{jk}|^2 for some complex vector P.
This is equivalent to h = |FFT(P)|^2 (up to scaling). This ensures h >= 0.

Actually, a simpler approach: since the ratio is scale-invariant (scales as 1/c),
we should normalize somehow. Looking at the seed, it doesn't normalize. But the
optimization will naturally find a scale.

Let me use the DFT-based parameterization and optimize the complex coefficients P.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0  # safety margin before 150s kill

def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 == 0 or l2sq == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if peak == 0:
        return 0.0
    return l2sq / (l1 * peak)

def _h_from_P(P_complex):
    """Convert complex coefficients to nonnegative heights via DFT magnitude squared."""
    # P_complex has N components
    H = np.fft.fft(P_complex)
    h = np.abs(H)**2
    return h

def _objective_from_P(P_flat):
    """Objective to minimize: negative of ratio, parameterized by real/imag of P."""
    P_complex = P_flat[0:N] + 1j * P_flat[N:2*N]
    h = _h_from_P(P_complex)
    l2sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)
    if peak < 1e-15:
        return 0.0
    return -l2sq / (l1 * peak)

def _optimize_complex(P0, time_budget):
    """Optimize complex parameters using L-BFGS-B."""
    start = time.time()
    P0_flat = np.concatenate([P0.real, P0.imag])
    bounds = [(None, None)] * (2 * N)
    
    def obj(x):
        if time.time() - start > time_budget:
            return np.inf
        return _objective_from_P(x)
    
    result = minimize(
        obj,
        P0_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1000}
    )
    
    P_opt = result.x[0:N] + 1j * result.x[N:2*N]
    return P_opt

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    t0 = time.time()
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    rng = np.random.RandomState(12345)
    
    # Generate diverse initial P vectors
    P_inits = []
    
    # 1. Uniform P (gives flat h)
    P_inits.append(np.ones(N, dtype=complex))
    
    # 2. Random P vectors
    for seed in range(30):
        r = np.random.RandomState(seed)
        P_inits.append(r.randn(N) + 1j * r.randn(N))
    
    # 3. Structured: low-frequency dominant
    for freq_scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
        k = np.arange(N)
        P_struct = np.exp(-k**2 / (N * freq_scale)) * (1 + 0.5j)
        P_inits.append(P_struct)
    
    # 4. Spike in P (gives chirp-like h)
    for pos in [0, N//4, N//2, 3*N//4, N-1]:
        P_spike = np.zeros(N, dtype=complex)
        P_spike[pos] = 1.0
        P_inits.append(P_spike)
    
    # 5. Two spikes in P
    for p1, p2 in [(0, N//2), (N//4, 3*N//4), (0, N-1)]:
        P_two = np.zeros(N, dtype=complex)
        P_two[p1] = 1.0
        P_two[p2] = 0.5 + 0.5j
        P_inits.append(P_two)
    
    # 6. Gaussian in frequency
    for sigma_p in [1, 2, 5, 10]:
        k = np.arange(N) - N//2
        P_gauss = np.exp(-k**2 / (2 * sigma_p**2))
        P_inits.append(P_gauss.astype(complex))
    
    # Optimize from each initialization
    time_per_opt = 2.5  # seconds per attempt
    
    for i, P0 in enumerate(P_inits):
        elapsed = time.time() - t0
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        P_opt = _optimize_complex(P0, time_per_opt)
        h_opt = _h_from_P(P_opt)
        
        # Ensure nonnegativity (should be automatic from |FFT|^2)
        h_opt = np.maximum(h_opt, 0.0)
        
        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Also try direct optimization on h with nonnegativity constraints
    # using a reparameterization h_i = x_i^2
    t_direct = time.time()
    if t_direct - t0 < _DEADLINE - 20:
        h_inits_direct = [
            np.ones(N),
            np.exp(-np.linspace(-1, 1, N)**2 / 0.5),
            np.exp(-np.linspace(-1, 1, N)**2 / 2.0),
            np.abs(np.sin(np.linspace(0, 2*np.pi, N))),
        ]
        for seed in [100, 200, 300]:
            r = np.random.RandomState(seed)
            h_inits_direct.append(np.abs(r.randn(N)))
        
        for h0 in h_inits_direct:
            elapsed = time.time() - t0
            if elapsed + 3 > _DEADLINE:
                break
            
            # Use x_i = sqrt(h_i) parameterization
            x0 = np.sqrt(np.maximum(h0, 0))
            
            def obj_x(x):
                if time.time() - t0 > _DEADLINE:
                    return np.inf
                h = x**2
                l2sq = np.sum(h**2)
                l1 = np.sum(h)
                if l1 < 1e-15:
                    return 0.0
                conv = np.convolve(h, h, mode='full')
                peak = np.max(conv)
                if peak < 1e-15:
                    return 0.0
                return -l2sq / (l1 * peak)
            
            result = minimize(
                obj_x,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 500}
            )
            h_opt = result.x**2
            h_opt = np.maximum(h_opt, 0.0)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
    
    # Final: try scaling the best solution
    # Since ratio scales as 1/c, we want small c. But the problem might have
    # an implicit normalization. Let's just return the best found.
    
    elapsed = time.time() - t0
    if elapsed > _DEADLINE:
        # Shouldn't happen due to guards, but just in case
        pass
    
    return list(best_h)
