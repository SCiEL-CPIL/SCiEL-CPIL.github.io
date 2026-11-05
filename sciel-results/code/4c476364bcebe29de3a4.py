"""
Strategy: Full 400-dim Optimization with Analytic Gradients (L-BFGS-B)
The current strategy limits h to a 4-parameter sinusoidal family. 
This strategy optimizes the full vector h directly.
We derive the analytic gradient of V(h) = |2*N*max(conv(h,h)) / (sum(h))^2|.
Let C = conv(h,h), M = max(C), S = sum(h).
V = 2*N*M / S^2 (assuming M>0 and S!=0, and we handle signs carefully).
Actually, the evaluator computes:
val = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)

Let f(h) = 2 * N * max_k (conv(h,h)_k) / (sum(h))^2.
We want to minimize |f(h)|.
Since we can scale h, the value is scale-invariant? 
If we scale h by c, conv scales by c^2, sum scales by c.
val = |2 * N * c^2 * M / (c * S)^2| = |2 * N * M / S^2|. 
Yes, it is scale invariant. 
However, the sign of S matters for the square, but S^2 is positive.
The sign of M matters for the absolute value.
If M is negative, val = -2*N*M/S^2. If M is positive, val = 2*N*M/S^2.
We want to minimize this.
Ideally, we want M to be as close to 0 as possible, or negative if we can flip the sign?
Wait, if M < 0, then val = -2*N*M/S^2 > 0.
If M > 0, val = 2*N*M/S^2 > 0.
So we want M/S^2 to be as small in magnitude as possible.
Since we can flip the sign of h (h -> -h), M -> -M, S -> -S, S^2 -> S^2.
So val(-h) = |2*N*(-M)/S^2| = |2*N*M/S^2| = val(h).
So the sign of h doesn't change the value.
We can assume without loss of generality that S > 0.
Then we want to minimize 2*N*|M|/S^2.
If M >= 0, we minimize M/S^2.
If M < 0, we minimize -M/S^2.

The function g(h) = max_k conv_k(h) is convex, but not differentiable everywhere.
However, for a specific h, the set of indices k where conv_k(h) = max(conv(h)) is the "active set".
Let K_max be the set of indices where conv_k(h) = M.
The gradient of M with respect to h is the average of the gradients of the active terms?
No, the subgradient is the convex hull. For L-BFGS-B, we can pick one active index, or the average.
Actually, if there's a unique maximum, the gradient is well-defined.
If there are multiple, the function is non-smooth.
But in practice, we can pick the index of the maximum (or average over ties) to get a valid subgradient direction that often works well for descent.

Let conv_k(h) = sum_j h[j] h[j+k] (for full convolution, indices need care).
np.convolve(h, h) produces a vector of length 2N-1.
Let's say C = np.convolve(h, h).
C_k = sum_{i} h[i] h[i+k] ? No, standard convolution:
C[k] = sum_{i} h[i] h[k-i] ?
NumPy convolve: np.convolve(a, v) is the discrete, linear convolution of 1-dimensional sequences.
C[k] = sum_{i=0}^{N-1} a[i] * v[k-i] where v is zero-padded.
For a=v=h, C[k] = sum_{i} h[i] h[k-i].
Indices k range from 0 to 2N-2.
Let's define the gradient of C[k] with respect to h.
C[k] = sum_{i} h[i] h[k-i].
dC[k]/dh[j] = h[k-j] + h[j] (if k-j is in [0, N-1] and j is in [0, N-1]).
Actually, it's the cross-correlation.
If we fix k, the gradient vector G_k has components:
G_k[j] = h[k-j] if 0 <= k-j < N, plus h[j] if 0 <= k < N?
Let's check.
C[k] = sum_{i=0}^{N-1} h[i] h[k-i].
Term i contributes h[i] h[k-i].
Derivative w.r.t h[j]:
1. If i=j, derivative is h[k-j] (provided 0 <= k-j < N).
2. If k-i=j => i=k-j, derivative is h[k-j] (provided 0 <= k-j < N).
So if j != k-j, we have two terms?
Wait, if i=j and i=k-j, then j = k-j => k=2j.
So generally, dC[k]/dh[j] = h[k-j] * I(0<=k-j<N) + h[j] * I(0<=k-j<N)?
No, the second term comes from the case where the index in the second factor is j.
Let u = i, v = k-i.
C[k] = sum_u h[u] h[k-u].
dC[k]/dh[j] = sum_u ( delta_{u,j} h[k-u] + h[u] delta_{k-u,j} )
= h[k-j] (if 0<=k-j<N) + h[k-j] (if 0<=k-j<N) ?
Wait, if u=j, term is h[j]h[k-j]. Deriv w.r.t h[j] is h[k-j].
If k-u=j => u=k-j. Term is h[k-j]h[j]. Deriv w.r.t h[j] is h[k-j].
So if j != k-j, we have h[k-j] + h[k-j] = 2 h[k-j].
If j = k-j (i.e., k=2j), we have h[j] * h[j] = h[j]^2. Deriv is 2 h[j].
So in all cases, dC[k]/dh[j] = 2 h[k-j] if 0 <= k-j < N, else 0?
Let's test with N=2, h=[a,b].
conv = [a^2, ab, b^2].
C[0] = a^2. dC[0]/da = 2a. Formula: k=0, j=0. 2h[0]=2a. j=1. k-j=-1 (out). 0. Correct.
C[1] = ab. dC[1]/da = b. Formula: k=1, j=0. 2h[1]=2b. Wait.
My derivation:
u=0, v=1. Term a b. Deriv w.r.t a is b.
u=1, v=0. Term b a. Deriv w.r.t a is 0 (since a is not in h[1]).
So dC[1]/da = b.
Formula gave 2b. Where is the error?
The sum is over i.
C[k] = sum_{i=0}^{N-1} h[i] h[k-i].
For C[1] with N=2:
i=0: h[0]h[1] = ab.
i=1: h[1]h[0] = ba.
Sum = 2ab.
Ah, np.convolve([a,b],[a,b]) = [a^2, 2ab, b^2].
So C[1] = 2ab.
dC[1]/da = 2b.
Formula: j=0, k=1. k-j=1. 2h[1]=2b. Correct.
dC[1]/db = 2a.
Formula: j=1, k=1. k-j=0. 2h[0]=2a. Correct.

So, Gradient of C[k] w.r.t h is a vector G_k where:
G_k[j] = 2 * h[k-j] if 0 <= k-j < N, else 0.
Note that this is essentially 2 * reverse(h) shifted?
Let h_rev = h[::-1].
G_k is 2 * (h_rev shifted by k)?
Let's check.
h = [h0, h1, ..., h_{N-1}].
h_rev = [h_{N-1}, ..., h1, h0].
G_k[j] = 2 h[k-j].
If we define a vector v such that v[j] = h[k-j], this is a shifted and reversed version of h.
Specifically, if we create a zero-padded array H of length 2N-1, with H[N-1 + i] = h[i]? No.
Let's just compute G_k explicitly or via slicing.
G_k = np.zeros(N)
if k < N:
    # j from 0 to N-1, k-j in [0, N-1] => j in [k-(N-1), k]
    # j >= 0 => j >= max(0, k-N+1)
    # j <= N-1 => j <= min(N-1, k)
    start = max(0, k - N + 1)
    end = min(N - 1, k)
    # G_k[j] = 2 h[k-j]
    # Let m = k-j. As j goes from start to end, m goes from k-start to k-end.
    # G_k[start:end+1] = 2 * h[k-start : k-end+1][::-1] ?
    # Let's check indices.
    # j = start => m = k-start.
    # j = end => m = k-end.
    # So we need h[m] for m in [k-end, k-start].
    # Slice h[k-end : k-start+1] and reverse it.
    G_k[start:end+1] = 2.0 * h[k-end : k-start+1][::-1]
elif k >= N:
    # k-j in [0, N-1] => j in [k-(N-1), k]
    # j <= N-1 => k-j >= k-(N-1)
    # j >= 0 => k-j <= k
    # Also j <= N-1.
    # Let m = k-j. m in [0, N-1].
    # j = k-m.
    # j in [0, N-1] => k-m in [0, N-1] => m in [k-(N-1), k].
    # Also m in [0, N-1].
    # So m in [max(0, k-N+1), min(N-1, k)].
    # G_k[j] = 2 h[m] where j = k-m.
    # So for each m in range, G_k[k-m] = 2 h[m].
    start_m = max(0, k - N + 1)
    end_m = min(N - 1, k)
    # j = k - m.
    # m goes start_m to end_m.
    # j goes k-start_m to k-end_m.
    # G_k[k-start_m : k-end_m+1] = 2 * h[start_m : end_m+1][::-1]
    G_k[k-start_m : k-end_m+1] = 2.0 * h[start_m : end_m+1][::-1]

This is O(N) per k. Since we need gradient of max, we only need this for the active k (or few active k).
So gradient computation is O(N).
Evaluation is O(N log N) via FFT or O(N^2) via direct. N=400, N^2=160k, very fast.
L-BFGS-B with analytic gradient should be very fast.

We will use a randomized restart strategy.
1. Generate random h (e.g., Gaussian).
2. Normalize h so sum(h) = 1 (to fix scale, though not strictly necessary, helps conditioning).
   Actually, if we normalize, we must account for the constraint in the gradient?
   No, L-BFGS-B doesn't handle equality constraints well.
   Better to let it optimize freely, or use a reparametrization.
   Or just optimize h directly. The function is scale invariant, so the optimizer might wander in scale.
   To prevent this, we can add a small regularization term ||h||^2 to the objective?
   Or just normalize h after each step? No, that breaks the gradient.
   Simplest: Optimize h directly. If it drifts, it's fine.
   However, if S -> 0, val -> inf. We should avoid S=0.
   We can add a penalty if |S| is small.
   
   Let's try optimizing h directly with L-BFGS-B.
   Objective: val(h) + lambda * (1/|S| - 1) if |S| < 1?
   Or just val(h).
   
   We will do multiple restarts.
   
   To speed up convergence, we can use the fact that the objective is smooth except at max ties.
   
   Let's implement the gradient carefully.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 145.0

def _evaluate(h):
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _objective_and_grad(h):
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return 1e10, np.zeros(N)
    
    conv = np.convolve(h, h)
    max_val = np.max(conv)
    val = abs(2.0 * N * max_val / (s * s))
    
    # Gradient
    # d(val)/dh = d/dh ( | 2N M / S^2 | )
    # Let A = 2N M / S^2. val = |A|.
    # d(val)/dh = sign(A) * dA/dh (if A != 0).
    # A = 2N M S^{-2}.
    # dA/dh = 2N ( dM/dh * S^{-2} + M * d(S^{-2})/dh )
    # d(S^{-2})/dh = -2 S^{-3} * 1 = -2/S^3.
    # dA/dh = 2N ( G_M / S^2 - 2 M / S^3 )
    # where G_M is the gradient of M = max(conv).
    
    # Find active indices
    # Due to floating point, we might have multiple close maxima.
    # We pick the index of the maximum.
    # If there are ties, the gradient is undefined, but picking one is a subgradient.
    idx_max = np.argmax(conv)
    
    # Compute G_M (gradient of conv[idx_max] w.r.t h)
    k = idx_max
    G_M = np.zeros(N)
    if k < N:
        start = max(0, k - N + 1)
        end = min(N - 1, k)
        G_M[start:end+1] = 2.0 * h[k-end : k-start+1][::-1]
    else:
        start_m = max(0, k - N + 1)
        end_m = min(N - 1, k)
        G_M[k-start_m : k-end_m+1] = 2.0 * h[start_m : end_m+1][::-1]
        
    # Calculate gradient of val
    # sign_A = 1 if A > 0 else -1
    # A = 2N M / S^2. Since S^2 > 0, sign(A) = sign(M).
    sign_M = 1.0 if max_val > 0 else -1.0
    if max_val == 0:
        return val, np.zeros(N)
        
    grad_A = 2.0 * N * (G_M / (s * s) - 2.0 * max_val / (s * s * s))
    grad_val = sign_M * grad_A
    
    return val, grad_val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Number of restarts to try
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize h
        # Mix of random and structured initializations
        if i % 2 == 0:
            h0 = rng.standard_normal(N)
        else:
            # Low frequency sinusoid
            freq = rng.uniform(1, 10)
            phi = rng.uniform(0, 2*np.pi)
            h0 = np.cos(2*np.pi*freq*np.arange(N)/N + phi)
            h0 += rng.standard_normal(N) * 0.1
            
        # Normalize to avoid extreme scales
        s0 = np.sum(h0)
        if abs(s0) > 1e-6:
            h0 = h0 / s0 * N # Make sum N? Or just 1?
            # Scale invariance means we can pick any.
            # Let's keep sum around N to avoid tiny numbers.
            
        try:
            res = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if res.fun < best_val:
                best_h = res.x
                best_val = res.fun
                # Check if we are close to optimal?
                # The theoretical minimum is likely related to the roots of unity or similar.
                # For N=400, maybe we can do much better than 2.0.
                
        except Exception:
            continue
            
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
