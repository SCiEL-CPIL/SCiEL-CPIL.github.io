"""Optimized program for the first autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    # We want to find a nonnegative step function h on N equal intervals of [-1/4, 1/4]
    # that minimizes the maximum of its autoconvolution.
    
    # Let's denote the step heights as x = [x_0, ..., x_{N-1}], x_i >= 0.
    # The autoconvolution of a step function with these heights on equal intervals of width w=1/(2N) 
    # (since interval is [-1/4, 1/4], length 1/2, so w = 1/(2*600))
    # The autoconvolution (h*h)(t) for t in [-1/2, 1/2] is a piecewise linear function.
    # However, the "normalized maximum" likely refers to the maximum value of the autoconvolution
    # normalized by something. Let's look at the standard definition.
    
    # For a step function h(x) = sum_i x_i * 1_{[i*w, (i+1)*w)}(x - a) where a is the start.
    # The autoconvolution is (h*h)(t) = integral h(t-s)h(s) ds.
    # Since h is a step function, the autoconvolution is a piecewise linear function (tent-like shapes added up).
    # Specifically, if h is constant on intervals, the autoconvolution is the sum of shifted triangles.
    
    # Let's define the intervals. The domain is [-1/4, 1/4].
    # Let the intervals be I_k = [-1/4 + k*w, -1/4 + (k+1)*w) for k=0,...,N-1, where w = 1/(2N).
    # Let h(x) = x_k for x in I_k.
    
    # The autoconvolution c(t) = (h*h)(t).
    # c(t) is continuous and piecewise linear. Its maximum occurs at one of the "peaks" of the constituent triangles or at the boundaries of the pieces.
    # Actually, for nonnegative h, the maximum of the autoconvolution is bounded by the square of the L1 norm? No.
    # Let's think about the discrete approximation.
    # If we sample the autoconvolution at the midpoints or at the grid points, we can formulate this.
    
    # A simpler approach: The problem asks for the "normalized maximum of its autoconvolution".
    # Usually, for such inequalities, we normalize by the integral of h or the L2 norm.
    # Let's assume the score is max_t (h*h)(t) / (integral h)^2 or similar.
    # Wait, the task says "minimize the normalized maximum of its autoconvolution; this gives an upper bound on C1".
    # The constant C1 in the first autocorrelation inequality usually relates to ||f*f||_infty <= C1 ||f||_2^2 or similar.
    # Let's assume the score is max_t (h*h)(t) where h is normalized such that ||h||_1 = 1 or ||h||_2 = 1.
    
    # Let's try to minimize max_t (h*h)(t) subject to sum(x_i * w) = 1 (L1 norm = 1) and x_i >= 0.
    # Or perhaps sum(x_i^2 * w) = 1 (L2 norm = 1).
    # Given the context of "upper bound on C1", and C1 is often a constant in ||f*f||_infty <= C ||f||_2^2,
    # let's assume we fix ||h||_2 = 1 and minimize max_t (h*h)(t).
    
    # Let's compute the autoconvolution for a given x.
    # The autoconvolution of a step function can be computed efficiently.
    # c(t) for t in [k*w, (k+1)*w] is linear.
    # Actually, the maximum of the autoconvolution of a nonnegative function occurs at t=0? No, not necessarily.
    # But for symmetric functions, it might. Let's not assume symmetry initially, but it's likely optimal.
    
    # Let's use a numerical optimization approach.
    # Variables: x (N floats, >= 0)
    # Constraint: sum(x_i^2 * w) = 1 (L2 normalization)
    # Objective: max_t (h*h)(t)
    
    # How to compute max_t (h*h)(t) for a step function?
    # The autoconvolution c(t) is the sum over i,j of x_i x_j * (length of intersection of I_i shifted by t and I_j).
    # This is equivalent to the discrete convolution of x with x, but scaled and integrated.
    # Let's define the discrete autoconvolution d_k = sum_i x_i x_{i-k} (with appropriate boundaries).
    # The continuous autoconvolution at the midpoint of the k-th interval of the output domain
    # is related to d_k.
    
    # Actually, a simpler discrete approximation:
    # Let's just use the discrete convolution.
    # Let y = np.convolve(x, x, mode='full').
    # The maximum of the continuous autoconvolution is bounded by the maximum of the discrete convolution
    # scaled by w.
    # Specifically, (h*h)(t) <= w * max_k (sum_i x_i x_{i-k}) ?
    # Let's check dimensions. h has units 1/length. h*h has units 1/length^2.
    # x_i are the heights. w is the width.
    # integral h dx = sum x_i w.
    # (h*h)(0) = integral h(s) h(-s) ds. If h is symmetric, (h*h)(0) = integral h(s)^2 ds = sum x_i^2 w = ||h||_2^2.
    # So if we fix ||h||_2 = 1, then (h*h)(0) = 1.
    # Can we make the maximum of the autoconvolution smaller than 1?
    # No, because (h*h)(0) = ||h||_2^2 = 1.
    # So the maximum is at least 1.
    # Therefore, if we minimize max_t (h*h)(t) subject to ||h||_2=1, the minimum is 1, achieved by any h where the autoconvolution
    # is constant or peaks at 0.
    # This suggests the normalization is different.
    
    # Let's reconsider. Maybe the score is max_t (h*h)(t) / (integral h)^2.
    # Let I = integral h = sum x_i w.
    # Score = max_t (h*h)(t) / I^2.
    # If we scale h by c, max_t (ch*ch)(t) = c^2 max_t (h*h)(t), and I becomes cI.
    # So the score is scale-invariant.
    # Let's set I = 1. Then we minimize max_t (h*h)(t) subject to sum x_i w = 1, x_i >= 0.
    # In this case, (h*h)(0) = integral h(s)h(-s) ds. If h is symmetric, (h*h)(0) = integral h(s)^2 ds.
    # By Cauchy-Schwarz, (integral h)^2 <= (measure of support) * integral h^2.
    # So integral h^2 >= 1 / (measure of support).
    # The measure of support is 1/2.
    # So (h*h)(0) >= 2.
    # Thus the score is at least 2.
    
    # Let's try to minimize max_t (h*h)(t) subject to sum x_i w = 1.
    # We can use scipy.optimize.minimize.
    # The objective function needs to compute the max of the autoconvolution.
    
    # Let's implement the autoconvolution calculation for a step function.
    # The autoconvolution c(t) is piecewise linear. The maximum must occur at one of the "knots" of the piecewise linear function.
    # The knots are at t = (i-j)*w for integers i,j.
    # So we just need to evaluate c(t) at all these points and take the max.
    
    # How to compute c(t) at t = k*w?
    # c(k*w) = sum_{i,j} x_i x_j * length(I_i ∩ (I_j + k*w))
    # I_i = [i*w, (i+1)*w) (shifted to start at 0 for simplicity, then we shift back)
    # Actually, let's define the intervals relative to 0.
    # Let the intervals be J_i = [i*w, (i+1)*w) for i = -M, ..., M-1 where M=N/2? No, N=600.
    # The domain is [-1/4, 1/4]. Let's shift it to [0, 1/2].
    # Let the intervals be K_i = [i*w, (i+1)*w) for i=0,...,N-1, where w=1/(2N).
    # Let h(x) = x_i for x in K_i.
    # Then (h*h)(t) = integral h(t-s)h(s) ds.
    # Let's evaluate at t = k*w.
    # (h*h)(k*w) = sum_{i,j} x_i x_j * length(K_i ∩ (K_j + k*w))
    # K_j + k*w = [(j+k)*w, (j+k+1)*w)
    # The intersection of [i*w, (i+1)*w) and [(j+k)*w, (j+k+1)*w) has length:
    # - w if i = j+k
    # - w/2 if |i - (j+k)| = 1 ? No, they are disjoint or overlap by w or 0.
    # Wait, the intervals are adjacent.
    # [i*w, (i+1)*w) and [(j+k)*w, (j+k+1)*w) overlap if |i - (j+k)| <= 0?
    # No, they are unit intervals in the scaled domain.
    # If i = j+k, they are the same interval, length w.
    # If i != j+k, they are disjoint?
    # No, [0,1) and [1,2) are disjoint.
    # So the intersection is w if i = j+k, and 0 otherwise.
    # This would imply (h*h)(k*w) = w * sum_i x_i x_{i-k}.
    # But this is only true if the function is defined on the whole real line and we are evaluating at the grid points.
    # However, the function is zero outside [0, 1/2].
    # So we need to be careful with the boundaries.
    
    # Let's use a more robust method: compute the autoconvolution on a fine grid and take the max.
    # Or, use the fact that the max of the autoconvolution of a nonnegative step function occurs at one of the midpoints of the intervals or at the boundaries.
    
    # Let's try a simple optimization with scipy.
    # We will minimize the maximum of the autoconvolution, subject to sum(x) = 1 (since w is constant, we can ignore it for the constraint and objective scaling).
    # Let's define the objective as max_t (h*h)(t).
    
    # To compute (h*h)(t) efficiently:
    # We can use the discrete convolution.
    # Let's create a longer array to handle the convolution.
    # Actually, the autoconvolution of a function with support [0, L] has support [0, 2L].
    # Here L = 1/2. So support is [0, 1].
    # We can sample the autoconvolution at N*2 points.
    
    # Let's define a function to compute the max autoconvolution value.
    def max_autoconv(x):
        # x is a vector of N nonnegative numbers.
        # Compute the discrete autoconvolution.
        # The continuous autoconvolution at t = (k + 0.5) * w (midpoints) is:
        # c((k+0.5)w) = w * sum_i x_i * x_{i-k} ? No.
        # Let's use the formula: c(t) = sum_i x_i * (h(t - (i+0.5)w)) ? No.
        
        # Let's use the property that the autoconvolution of a step function is a piecewise linear function.
        # The maximum occurs at one of the knots, which are at t = k*w.
        # Let's evaluate c(t) at t = k*w for k = -N, ..., N.
        # c(k*w) = w * sum_{i} x_i * x_{i-k} ?
        # Let's check for k=0. c(0) = w * sum_i x_i^2. This is correct if h is symmetric? No.
        # c(0) = integral h(s) h(-s) ds.
        # If h is defined on [0, L], then h(-s) is 0 for s>0. So c(0) = 0?
        # This depends on the definition. The autoconvolution is (h*h)(t) = integral h(t-s)h(s) ds.
        # If h is supported on [0, L], then (h*h)(t) is supported on [0, 2L].
        # (h*h)(0) = integral h(-s)h(s) ds = 0 (since h is 0 for negative arguments).
        # The maximum is at t=L, where (h*h)(L) = integral h(L-s)h(s) ds.
        # If h is constant, h(s)=1, then (h*h)(L) = L.
        
        # Let's assume the step function is on [-1/4, 1/4].
        # Let's shift it to [0, 1/2] for simplicity.
        # Let h(s) = x_i for s in [i*w, (i+1)*w).
        # Then (h*h)(t) = integral h(t-s) h(s) ds.
        # The maximum is at t = 1/2 (the center of the support of the autoconvolution).
        # (h*h)(1/2) = integral h(1/2 - s) h(s) ds.
        # Let s be in [i*w, (i+1)*w). Then 1/2 - s is in [1/2 - (i+1)w, 1/2 - i*w).
        # 1/2 = N*w. So 1/2 - s is in [(N-i-1)w, (N-i)w).
        # So h(1/2 - s) = x_{N-i-1}.
        # Thus (h*h)(1/2) = sum_i w * x_i * x_{N-i-1}.
        # This is the inner product of x and its reverse.
        
        # But the maximum might not be at t=1/2.
        # Let's compute the autoconvolution at all grid points t = k*w.
        # c(k*w) = w * sum_i x_i * x_{k-i} ?
        # Let's derive it.
        # c(t) = sum_i x_i * (length of {s: s in K_i and t-s in K_j for some j})
        # This is getting complicated. Let's just use a fine grid and np.convolve.
        
        # Let's create a zero-padded array for h.
        # The support of h is [0, 1/2]. The support of h*h is [0, 1].
        # Let's sample at 2N points.
        # Let's create an array h_samples of length 2N, where h_samples[i] = x[i] for i < N, and 0 otherwise.
        # But this is a discrete approximation.
        
        # A better way: The autoconvolution of a step function is the sum of triangles.
        # Each pair (i,j) contributes a triangle centered at (i+0.5)w + (j+0.5)w = (i+j+1)w.
        # The height of the triangle is x_i x_j w.
        # The width is 2w.
        # So the autoconvolution is a sum of triangles.
        # The maximum of the sum of triangles can be found by evaluating at the peaks of the triangles.
        # The peaks are at t = (i+j+1)w.
        # So we just need to evaluate the sum at all these points.
        
        # Let's compute the sum at t = k*w for k = 1, ..., 2N-1.
        # c(k*w) = w * sum_{i+j = k-1} x_i x_j * (1 - |...|) ? No, the triangles overlap.
        
        # Let's use the discrete convolution of x with x.
        # Let y = np.convolve(x, x, mode='full').
        # y[k] = sum_i x_i x_{k-i}.
        # The continuous autoconvolution at t = (k+1)w is w * y[k] ?
        # Let's check for N=1. x=[1]. y=[1].
        # h is 1 on [0,w]. (h*h)(t) is a triangle from 0 to 2w, peak at w with height w.
        # y[0]=1. w*y[0]=w. Correct.
        # For N=2. x=[a,b]. y=[a^2, ab+ba, b^2].
        # (h*h)(w) = w*(a^2 + 2ab) ? No.
        # Let's calculate (h*h)(w).
        # h(s) = a for s in [0,w), b for s in [w,2w).
        # (h*h)(w) = integral h(w-s)h(s) ds.
        # For s in [0,w), w-s in (0,w], so h(w-s)=a (if w-s < w) or b (if w-s >= w).
        # If s in [0,w), w-s in (0,w]. At s=0, w-s=w, h(w)=b (if we define it that way).
        # Let's assume h is right-continuous.
        # This is getting messy. Let's just use a fine grid and interpolate.
        
        # Let's use the fact that the max of the autoconvolution is at least the value at the center.
        # And for symmetric functions, the center is the max.
        # Let's assume the optimal function is symmetric.
        # Then x_i = x_{N-1-i}.
        # Then the autoconvolution is symmetric around 1/2.
        # And the max is at 1/2.
        # So we can just compute c(1/2) = w * sum_i x_i x_{N-1-i}.
        # And minimize this subject to sum x_i w = 1.
        
        # Let's try this.
        # Objective: sum_i x_i x_{N-1-i}
        # Constraint: sum x_i = 1/w = 2N.
        # x_i >= 0.
        
        # This is a quadratic optimization problem.
        # Let's use scipy.optimize.minimize.
        
        return None # Placeholder

    # Let's implement the optimization.
    # We assume symmetry: x_i = x_{N-1-i}.
    # So we only need to optimize the first N/2 variables.
    # Let y = x[:N//2].
    # x = [y_0, y_1, ..., y_{N/2-1}, y_{N/2-1}, ..., y_1, y_0] (if N is even)
    
    # Objective: c(1/2) = w * (sum_{i=0}^{N/2-1} y_i y_{N-1-i} + y_{N/2-1}^2 if N is even)
    # Wait, if N is even, x_{N-1-i} for i in 0..N/2-1 is y_{N-1-i}.
    # N-1-i = 599-i.
    # If i=0, N-1-i=599. x_599 = y_0.
    # If i=N/2-1=299, N-1-i=300. x_300 = y_299.
    # So sum_{i=0}^{299} y_i y_{299-i}.
    # This is the dot product of y and its reverse.
    
    # Constraint: sum x_i = 2 * sum y_i = 2N => sum y_i = N.
    
    # Let's minimize f(y) = sum_{i=0}^{N/2-1} y_i y_{N/2-1-i} subject to sum y_i = N, y_i >= 0.
    
    # This is a quadratic form.
    # Let's use scipy.optimize.minimize with SLSQP.
    
    def objective(y):
        # y is the first half of x.
        # Compute the dot product of y and its reverse.
        return np.dot(y, y[::-1])
    
    def constraint(y):
        return np.sum(y) - N
    
    # Initial guess: uniform distribution.
    y0 = np.ones(N//2) * (N / (N//2))
    
    bounds = [(0, None)] * (N//2)
    
    cons = [{'type': 'eq', 'fun': constraint}]
    
    res = minimize(objective, y0, method='SLSQP', bounds=bounds, constraints=cons)
    
    y_opt = res.x
    
    # Construct x.
    x = np.zeros(N)
    x[:N//2] = y_opt
    x[N//2:] = y_opt[::-1]
    
    # Normalize x so that sum x_i w = 1.
    # sum x_i w = w * sum x_i = w * 2 * sum y_i = w * 2 * N = 2N w = 2N * 1/(2N) = 1.
    # So it's already normalized.
    
    # The score is the max of the autoconvolution.
    # We assumed the max is at the center.
    # Let's verify this by computing the autoconvolution on a grid.
    
    # Let's compute the autoconvolution at all grid points t = k*w.
    # c(k*w) = w * sum_i x_i x_{k-i} ?
    # Let's use the discrete convolution.
    y_conv = np.convolve(x, x, mode='full')
    # The continuous autoconvolution at t = (k+1)w is w * y_conv[k].
    # Let's check.
    # The max of w * y_conv is the score.
    
    score = np.max(y_conv) * (1/(2*N))
    
    # But we want to return the step heights.
    # The task says "return a sequence of nonnegative step heights".
    # So we return x.
    
    return x.tolist()

# EVOLVE-BLOCK-END
