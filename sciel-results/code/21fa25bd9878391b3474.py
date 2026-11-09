import numpy as np
import time

N = 600
# Domain is [-1/4, 1/4], length L = 1/2
# Interval width dx = L / N
L = 0.5
dx = L / N

def compute_autoconv_max(h):
    """
    Compute the maximum value of the autoconvolution of the step function defined by heights h.
    h is an array of length N.
    The step function is f(x) = h[i] for x in [x_i, x_{i+1}].
    """
    # For efficiency, we can use FFT or direct convolution.
    # Since N=600, direct convolution is O(N^2) = 360,000 ops, which is fast.
    # However, the autoconvolution of a step function is piecewise quadratic.
    # The maximum is likely at the center if symmetric, or at a peak of the discrete correlation.
    # A robust way is to compute the discrete correlation of the "masses" m_i = h_i * dx.
    # g(x) = sum_{i,j} m_i m_j * overlap_length(x_i + x_j - t) ... this is complex.
    
    # Simpler approximation:
    # The peak of the autoconvolution of a non-negative function is bounded by and often close to 
    # the sum of squares of the masses if concentrated, or the square of the integral if spread.
    # Actually, for step functions, the continuous autoconvolution g(t) is a quadratic spline.
    # The maximum can be found by evaluating g(t) at the grid points and midpoints.
    
    # Let's use the property that g(t) is the convolution of the step function with itself.
    # We can approximate the max by computing the discrete auto-correlation of the heights,
    # scaled appropriately.
    
    # Masses
    m = h * dx
    # Discrete auto-correlation c[k] = sum_i m[i] m[i+k]
    # This approximates the integral of f(t)f(t-k dx) ?
    # Actually, the continuous autoconvolution g(t) at t = k * dx is 
    # approximately dx * sum_i m[i] m[i+k] ? No.
    
    # Let's just use scipy.signal.fftconvolve or np.convolve on the masses, 
    # but we need the continuous max.
    # For a step function, the autoconvolution is a quadratic function between grid points.
    # The maximum of a quadratic function on an interval is at the endpoints or the vertex.
    
    # To be safe and fast, we can compute the discrete correlation of m.
    # c = np.correlate(m, m, mode='full')
    # The peak of the continuous convolution is related to the peak of this discrete correlation.
    # Specifically, max(g) is approximately max(c) * dx?
    # Let's test with uniform h=1.
    # m = dx * ones(N).
    # c[0] = sum m^2 = N * dx^2.
    # c[k] = (N-k) dx^2.
    # Max c is N dx^2 = (L/dx) dx^2 = L dx.
    # Actual max(g) for uniform is L (since f=1/L, g(0)=1/L * L = 1? No).
    # If f=1, g(0)=L.
    # If h=1, f=1. g(0)=L.
    # c[0] = N dx^2 = (L/dx) dx^2 = L dx.
    # So max(g) is NOT c[0].
    
    # Let's use a direct numerical integration for the few candidates we test.
    # We only need to test a small number of candidates.
    
    # Create the function values at midpoints? No, step function.
    # We can compute g(t) for t in a fine grid and take the max.
    # But we want to be fast.
    
    # Alternative: The maximum of the autoconvolution of a non-negative step function 
    # is achieved at t=0 if the function is symmetric and unimodal, or at the boundaries of the support.
    # For our candidates (Uniform, Spike, Tri), they are symmetric and unimodal.
    # So max is at t=0.
    # g(0) = sum_i (h_i * dx) * h_i * dx ? No.
    # g(0) = int f(t)^2 dt = sum_i h_i^2 dx.
    
    # Let's verify if max is always at 0 for these shapes.
    # Uniform: Yes.
    # Spike: Yes.
    # Triangle: Yes.
    
    # So we can just compute g(0) = sum(h**2) * dx.
    # And integral I = sum(h) * dx.
    # Score = g(0) / I**2.
    
    g0 = np.sum(h**2) * dx
    I = np.sum(h) * dx
    if I == 0:
        return float('inf')
    return g0 / (I**2)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Candidate 1: Uniform
    h_uniform = np.ones(N)
    
    # Candidate 2: Single Spike
    h_spike = np.zeros(N)
    h_spike[N // 2] = 1.0
    
    # Candidate 3: Triangle
    h_tri = np.zeros(N)
    center = N // 2
    width = N // 2
    for i in range(N):
        dist = abs(i - center)
        if dist < width:
            h_tri[i] = 1.0 - (dist / width)
        else:
            h_tri[i] = 0.0
            
    candidates = [
        ("uniform", h_uniform),
        ("spike", h_spike),
        ("tri", h_tri)
    ]
    
    best_h = h_uniform
    best_score = float('inf')
    
    for name, h in candidates:
        s = compute_autoconv_max(h)
        if s < best_score:
            best_score = s
            best_h = h
            
        if time.time() > deadline:
            break
            
    return list(best_h)
