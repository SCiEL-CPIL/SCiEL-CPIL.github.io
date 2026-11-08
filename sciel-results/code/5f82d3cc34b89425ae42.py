"""
Strategy: Continuous Gradient Descent on the signal values.

The task is to minimize the maximum absolute value of the autoconvolution 
divided by the squared integral. 
Let x be a vector of length N.
Autoconvolution C = x * x (linear convolution).
Integral I = sum(x).
Objective J = max(|C|) / I^2.

We can rewrite this as:
minimize max(|C|) subject to I^2 = 1 (or I = 1).
This is equivalent to minimizing max(|C|) / (sum(x))^2.

Since the objective is piecewise linear (max of absolute values), 
we can use subgradient descent. However, standard gradient descent 
on the max function is unstable. A better approach for "minimize max" 
problems is to use a smooth approximation (e.g., soft-max) or 
alternating minimization.

However, a simpler and robust structural change is to use 
**Simulated Annealing** on the continuous values, or 
**Gradient Descent with a Log-Sum-Exp approximation** of the max.

Let's use the Log-Sum-Exp (LSE) approximation for the max absolute value:
LSE(C, beta) ≈ max(|C|) for large beta.
Objective to minimize: LSE(|C|, beta) / (sum(x))^2.

We will perform gradient descent on x.
To handle the division by I^2, we can normalize x at each step to have sum=1, 
or include the scaling in the gradient.
Let f(x) = LSE(|C(x)|, beta).
We want to minimize f(x) / (sum(x))^2.
Let g(x) = f(x) / (sum(x))^2.
Gradient descent step: x_new = x - lr * grad(g).

Actually, it is often better to fix the norm or sum.
If we fix sum(x) = 1, we just minimize max(|C|).
Let's stick to fixing sum(x)=1 to simplify the gradient calculation.
We project x onto the hyperplane sum(x)=1 after each step.

The autoconvolution C can be computed efficiently using FFT.
C = ifft(fft(x, n=2N) ** 2)
We only need the first N elements (or 2N-1 depending on definition, but 
autoconvolution of length N usually implies linear conv of length 2N-1).
The problem states "signed equal-width step functions". This implies x_i are 
the heights.
"minimizing the maximum absolute autoconvolution".
Usually autoconvolution is defined as sum x_i x_{i+k}.
Let's assume standard linear autoconvolution.

Algorithm:
1. Initialize x with random noise, normalize sum(x)=1.
2. Iterate:
   a. Compute C = linear_autoconv(x).
   b. Compute gradient of LSE(|C|, beta) w.r.t x.
      d/dx LSE(|C|) = (1/beta) * sum_j (exp(|C_j|/beta) * sign(C_j) * dC_j/dx) / sum_j exp(|C_j|/beta)
      dC_j/dx is sparse. C_j = sum_k x_k x_{j-k}.
      This derivative is expensive to compute explicitly if done naively.
      
Alternative: Use the fact that C = ifft(|fft(x)|^2).
Gradient of C w.r.t x can be computed via FFT as well.
Let X = fft(x, n=2N).
C = ifft(X^2).
dC/dx is a linear operator. 
Actually, it is easier to use the chain rule with FFTs.
Let's use a simpler heuristic: **Coordinate Descent** or **Random Perturbation** 
with acceptance based on the objective. 
Given the 120s limit, a heavy gradient descent might be slow if N=400.
N=400 is small. FFT is O(N log N). 2N=800. 
We can afford ~1000-10000 steps.

Let's try **Gradient Descent with LSE approximation**.
To compute grad efficiently:
L = log(sum(exp(|C_j|/beta)))
dL/dC_j = exp(|C_j|/beta) / sum(exp(|C_i|/beta)) * sign(C_j) / beta
dC_j/dx is the "autoconvolution gradient".
Note that C = x * x.
The gradient of C w.r.t x is a matrix G where G_{i,j} = 2 x_{i-j} (with appropriate shifts).
Actually, d/dx_k (sum_i x_i x_{j-i}) = x_{j-k} + x_{j-k} = 2 x_{j-k} (if symmetric) or similar.
Let's use the FFT trick for the gradient of the convolution.
If y = x * x, then in frequency domain Y = X^2.
The gradient of y w.r.t x is not simply 2X in frequency domain because 
the mapping from x to y is quadratic.
However, we can compute the gradient of the objective with respect to x 
using the adjoint of the convolution operator.
Let A be the linear operator that computes the autoconvolution? No, it's quadratic.
Let's stick to a simpler, robust method: **Simulated Annealing** or **Random Search with Memory**.

Actually, a very effective method for "minimize max of quadratic forms" 
is **Alternating Minimization** or **Subgradient Descent**.

Let's try **Subgradient Descent** on the LSE approximation.
We need dL/dx.
L = log( sum_j exp(|C_j|/beta) )
dL/dx_k = (1/beta) * sum_j [ exp(|C_j|/beta) * sign(C_j) * (dC_j/dx_k) ] / Z
where Z = sum_j exp(|C_j|/beta).
dC_j/dx_k = x_{j-k} + x_{k-j} ? 
Linear autoconvolution: C_j = sum_{i} x_i x_{j-i}.
dC_j/dx_k = x_{j-k} + x_{k-j} (if we consider x_{n} for all n).
For k in [0, N-1] and j in [0, 2N-2]:
Term 1: x_{j-k} is valid if 0 <= j-k < N.
Term 2: x_{k-j} is valid if 0 <= k-j < N.
So dC_j/dx_k = (x_{j-k} if 0<=j-k<N else 0) + (x_{k-j} if 0<=k-j<N else 0).

This allows O(N^2) gradient computation per step if done naively.
N=400, N^2=160,000. 1000 steps = 160M ops. This is fast in Python with NumPy if vectorized.
We can vectorize the gradient computation.

Steps:
1. Define construct().
2. Init x = random normal, normalize sum=1.
3. Loop T times:
   a. Compute C = linear autoconvolution of x.
   b. Compute weights w_j = exp(|C_j|/beta) / Z.
   c. Compute grad_x. 
      grad_x[k] = sum_j w_j * sign(C_j) * (dC_j/dx_k) / beta
      We can compute this using correlations.
      dC_j/dx_k depends on x_{j-k} and x_{k-j}.
      Let's define a matrix M where M_{j,k} = dC_j/dx_k.
      This matrix is a banded matrix (Toeplitz-like).
      Actually, we can compute the gradient vector g using FFTs?
      Let's try to vectorize the sum over j.
      g_k = (1/beta) * sum_j w_j sign(C_j) [ x_{j-k} + x_{k-j} ]
      Let u_j = w_j sign(C_j).
      g_k = (1/beta) * ( sum_j u_j x_{j-k} + sum_j u_j x_{k-j} )
      The first term is a cross-correlation of u and x.
      The second term is a cross-correlation of u and x (reversed).
      Specifically:
      term1_k = sum_j u_j x_{j-k} = (u * x)_k (convolution)
      term2_k = sum_j u_j x_{k-j} = (u * x_rev)_k ?
      Let's check indices.
      x_{k-j} is x at index k-j.
      If we define x_rev[i] = x[-i], then x_{k-j} = x_rev[j-k].
      So sum_j u_j x_rev[j-k] is also a convolution/correlation.
      
      We can compute these using FFT.
      u is length 2N-1. x is length N.
      We need to be careful with padding.
      
      Let's implement this.
4. Normalize x to sum=1.
5. Return x.

We need to choose beta. Start large, decrease? 
Or just use a fixed beta that is large enough to be smooth but small enough to approximate max.
max(|C|) is around 1 (since sum x=1, C_0 = sum x^2 <= 1). 
Actually C_0 = sum x_i^2. If x is uniform, C_0 = 1/N.
If x is a spike, C_0 = 1.
We want to minimize the max.
Let's use beta = 0.1 or 0.5.

Deadline guard:
Time limit 120s.
We will run until time is up or convergence.
"""

import numpy as np
import time

N = 400

def _linear_autoconv(x):
    """Compute linear autoconvolution of x."""
    # Use FFT for speed
    n = len(x)
    size = 2 * n
    fx = np.fft.fft(x, n=size)
    fy = fx * fx
    return np.fft.ifft(fy).real[:2*n-1]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Initialize
    np.random.seed(42)
    x = np.random.randn(N)
    # Normalize to sum 1
    x = x / np.sum(x)
    
    best_x = x.copy()
    best_score = np.inf
    
    # Parameters
    beta = 0.5
    lr = 0.01
    
    # Pre-compute indices or use FFT for gradient
    # g_k = (1/beta) * (conv(u, x)_k + conv(u, x_rev)_k)
    # Note: conv(u, x) at index k is sum_j u_j x_{k-j}
    # Our formula: 
    # term1_k = sum_j u_j x_{j-k}
    # Let i = j-k => j = i+k. sum_i u_{i+k} x_i. This is correlation.
    # term2_k = sum_j u_j x_{k-j}
    # Let i = k-j => j = k-i. sum_i u_{k-i} x_i. This is convolution.
    
    # So g_k = (1/beta) * ( corr(u, x)[k] + conv(u, x)[k] ) 
    # where corr(u, x)[k] = sum_i u_{i+k} x_i
    # and conv(u, x)[k] = sum_i u_{k-i} x_i
    
    # We can compute both using FFT.
    # conv(u, x) = ifft(fft(u) * fft(x))
    # corr(u, x) is related to conv(u_rev, x) or similar.
    # Actually, let's just use np.convolve for simplicity if N is small?
    # N=400. np.convolve is O(N^2) in Python? No, it uses C loops, but 
    # FFT is O(N log N).
    # Let's use FFT.
    
    size = 2 * N
    fx = np.fft.fft(x, n=size)
    
    def compute_grad(u):
        # u is length 2N-1
        # Pad u to size
        u_pad = np.zeros(size)
        u_pad[:len(u)] = u
        fu = np.fft.fft(u_pad)
        
        # conv(u, x)
        conv_ux = np.fft.ifft(fu * fx).real[:N]
        
        # corr(u, x) = sum_i u_{i+k} x_i
        # This is equivalent to conv(u_rev, x) shifted?
        # u_rev[j] = u[-j-1] ?
        # Let's check: conv(u_rev, x)_k = sum_i u_rev[i] x_{k-i} = sum_i u[-i-1] x_{k-i}
        # Let j = -i-1 => i = -j-1.
        # sum_j u[j] x_{k+j+1}.
        # This is close to sum_j u_j x_{j-k} ?
        # My term1 was sum_j u_j x_{j-k}.
        # Let's re-derive carefully.
        
        # C_j = sum_i x_i x_{j-i}
        # dC_j/dx_k = x_{j-k} + x_{k-j}
        
        # g_k = sum_j u_j (x_{j-k} + x_{k-j})
        
        # Part A: sum_j u_j x_{j-k}
        # Let m = j-k => j = m+k.
        # sum_m u_{m+k} x_m
        # This is the cross-correlation of u and x, evaluated at lag k.
        # corr_ux[k] = sum_m u_{m+k} x_m
        
        # Part B: sum_j u_j x_{k-j}
        # Let m = k-j => j = k-m.
        # sum_m u_{k-m} x_m
        # This is the convolution of u and x, evaluated at k.
        # conv_ux[k] = sum_m u_{k-m} x_m
        
        # So g_k = corr_ux[k] + conv_ux[k]
        
        # How to compute corr_ux using FFT?
        # corr_ux[k] = sum_m u_{m+k} x_m
        # This is the same as conv(u, x_rev) ?
        # Let x_rev[m] = x[-m].
        # conv(u, x_rev)_k = sum_m u_{k-m} x_rev[m] = sum_m u_{k-m} x[-m]
        # Let n = -m => m = -n.
        # sum_n u_{k+n} x[n].
        # This is sum_n u_{k+n} x_n.
        # My term A is sum_m u_{m+k} x_m. Same thing!
        # So Part A = conv(u, x_rev)[k]
        
        x_rev = x[::-1]
        fx_rev = np.fft.fft(x_rev, n=size)
        conv_u_xrev = np.fft.ifft(fu * fx_rev).real[:N]
        
        # Part B is conv(u, x)
        # We already computed conv_ux above.
        
        return (conv_ux + conv_u_xrev) / beta

    for step in range(10000):
        if time.time() > deadline:
            break
            
        C = _linear_autoconv(x)
        
        # Compute u_j = w_j sign(C_j)
        # w_j = exp(|C_j|/beta) / Z
        abs_C = np.abs(C)
        max_abs = np.max(abs_C)
        # Shift to prevent overflow
        exp_vals = np.exp((abs_C - max_abs) / beta)
        Z = np.sum(exp_vals)
        w = exp_vals / Z
        u = w * np.sign(C)
        
        # Compute gradient
        grad = compute_grad(u)
        
        # Gradient descent
        x = x - lr * grad
        
        # Normalize to sum 1
        s = np.sum(x)
        if s != 0:
            x = x / s
            
        # Calculate score
        C_new = _linear_autoconv(x)
        score = np.max(np.abs(C_new))
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
        # Adaptive learning rate?
        # If score increases, reduce lr?
        # For now, keep simple.
        
    return best_x

# EVOLVE-BLOCK-END
