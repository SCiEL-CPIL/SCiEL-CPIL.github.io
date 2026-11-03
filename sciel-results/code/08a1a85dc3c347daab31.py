"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _soft_max_abs(x, beta):
    """
    Smooth approximation of max|c_k| where c is autoconvolution of x.
    Uses log-sum-exp trick on |c|^2 to approximate max|c|^2, then sqrt.
    This function and its gradient are computed efficiently.
    """
    fx = np.fft.rfft(x, _FFT_SIZE)
    c = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    
    # To avoid numerical issues with log(0), shift by small epsilon or use max
    # We want to approximate max(|c|). 
    # Let M = max(|c|). 
    # log-sum-exp of |c|^2 approximates M^2.
    # sqrt(LSE) approximates M.
    
    c2 = c * c
    # Softmax trick: log(sum(exp(beta * c2))) = beta * max(c2) + log(sum(exp(beta * (c2 - max(c2)))))
    # We want M = sqrt(max(c2)).
    # Approximation: M_approx = sqrt( (1/beta) * log(sum(exp(beta * c2))) )
    
    # For stability:
    max_c2 = np.max(c2)
    if max_c2 < 1e-12:
        return 0.0, np.zeros_like(x)
        
    # exp(beta * (c2 - max_c2))
    # beta is chosen such that beta * max_c2 is not too large to cause overflow, but large enough for sharpness.
    # Actually, the standard LSE is log(sum(exp(y_i))).
    # Here y_i = beta * c2_i.
    # log(sum(exp(beta * c2))) = beta * max(c2) + log(sum(exp(beta * (c2 - max(c2)))))
    
    # Let's compute gradient of J = (1/beta) * log(sum(exp(beta * c2)))
    # dJ/dx = (1/beta) * (1/sum(exp(beta*c2))) * sum( exp(beta*c2) * d(beta*c2)/dx )
    # dJ/dx = (1/sum(exp(beta*c2))) * sum( exp(beta*c2) * 2*c * dc/dx )
    # Note: dc/dx is the convolution matrix.
    # c_k = sum_i x_i x_{i+k-N+1} (roughly, for linear conv)
    # dc_k/dx_j = x_{j+k-N+1} + x_{j-k+N-1} (if indices valid)
    
    # This is getting complex for analytic gradient.
    # Let's use a simpler convex surrogate: sum of squares of the largest few peaks? No, that's not convex in max.
    
    # Alternative: Minimize the sum of squared autoconvolution coefficients weighted by a Gaussian around the peak?
    # Or simply minimize the L2 norm of the autoconvolution?
    # min ||c||_2^2 / (sum x)^2.
    # This is a Rayleigh quotient problem!
    # ||c||_2^2 = ||x * x||_2^2.
    # In frequency domain: ||c||_2^2 = (1/FFT_SIZE) * sum |X(k)|^4.
    # This is not convex in x.
    
    # Let's stick to the soft-max approach but use finite differences for gradient if analytic is too hard?
    # No, 400 dims, finite diff is 401 evaluations. 150s is plenty for ~1000 evaluations if each is fast.
    # FFT of 1024 is very fast.
    
    # Let's try minimizing the "peakiness" using a different convex formulation.
    # Consider the problem: min max_k |c_k| subject to sum x = 1.
    # This is equivalent to: min t subject to |c_k| <= t for all k, sum x = 1.
    # This is a convex optimization problem (SOCP or LP if we linearize |c_k|, but c_k is quadratic).
    # Wait, c_k is quadratic in x. So |c_k| <= t is not convex.
    # However, if we fix the sign of x, or if we look at the structure...
    
    # Actually, there is a known result that for x with sum 1, the minimum of max |c_k| is achieved by specific structures.
    # But let's try a different angle:
    # The metric is max|c| / S^2.
    # If we scale x by alpha, max|c| scales by alpha^2, S scales by alpha.
    # Metric is invariant to scaling?
    # max|alpha^2 c| / (alpha S)^2 = alpha^2 max|c| / (alpha^2 S^2) = max|c|/S^2.
    # Yes, scale invariant.
    # So we can fix sum(x) = 1.
    
    # Then we want to minimize max_k |c_k|.
    
    # Let's use scipy.optimize.minimize with BFGS and a smooth approximation of max|c|.
    # f(x) = log( sum_k exp(beta * c_k^2) ) / beta
    # This approximates max(c_k^2).
    # We want to minimize sqrt(f(x)).
    
    # Gradient of f(x):
    # f(x) = (1/beta) log( sum exp(beta c_k^2) )
    # df/dx = (1/beta) * (1/S) * sum_k exp(beta c_k^2) * d(beta c_k^2)/dx
    # df/dx = (1/S) * sum_k exp(beta c_k^2) * 2 c_k * (dc_k/dx)
    # where S = sum_k exp(beta c_k^2).
    
    # dc_k/dx is a matrix. c = A x^2? No, c is quadratic form x^T M_k x.
    # dc_k/dx = 2 M_k x.
    # So df/dx = (2/S) * sum_k exp(beta c_k^2) c_k M_k x.
    
    # M_k is the matrix for the k-th autoconvolution coefficient.
    # c_k = sum_i x_i x_{i+k} (with appropriate bounds).
    # M_k has 1s on the k-th diagonal and -k-th diagonal.
    
    # Computing this gradient explicitly requires constructing or applying these matrices.
    # Applying M_k x is just shifting x.
    # c_k = <x, shift_k(x)>.
    # dc_k/dx = shift_k(x) + shift_{-k}(x).
    
    # So we can compute the gradient efficiently!
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N)
        fx = np.fft.rfft(x, _FFT_SIZE)
        c = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
        
        c2 = c * c
        max_c2 = np.max(c2)
        if max_c2 < 1e-15:
            return 0.0, np.zeros(N)
        
        # Normalize c2 for numerical stability in exp
        # We want LSE = log(sum(exp(beta * c2)))
        # Let y = beta * c2.
        # LSE = max(y) + log(sum(exp(y - max(y))))
        
        y = beta * c2
        max_y = np.max(y)
        
        # exp(y - max_y)
        exp_y = np.exp(y - max_y)
        sum_exp_y = np.sum(exp_y)
        
        if sum_exp_y < 1e-300:
             return 0.0, np.zeros(N)
             
        # Objective value: (1/beta) * (max_y + log(sum_exp_y))
        # Note: max_y = beta * max_c2
        # So obj = max_c2 + (1/beta) log(sum_exp_y)
        obj = max_c2 + (1.0 / beta) * np.log(sum_exp_y)
        
        # Gradient
        # weights w_k = exp(beta c_k^2) / sum(exp(beta c^2)) = exp(y_k) / sum_exp_y
        # Wait, the formula for gradient of LSE(y) is softmax(y).
        # d/dx LSE(y(x)) = softmax(y) @ dy/dx
        # Here y_k = beta c_k^2.
        # dy_k/dx = 2 beta c_k (dc_k/dx)
        
        # softmax_weights = exp(y - max_y) / sum_exp_y
        w = exp_y / sum_exp_y
        
        # grad_obj = (1/beta) * sum_k w_k * dy_k/dx
        # = (1/beta) * sum_k w_k * 2 beta c_k (dc_k/dx)
        # = 2 * sum_k w_k c_k (dc_k/dx)
        
        grad = np.zeros(N)
        
        # We need to compute sum_k w_k c_k (dc_k/dx)
        # dc_k/dx = shift_k(x) + shift_{-k}(x)
        # This sum is effectively a convolution-like operation weighted by w_k c_k.
        
        # Let a_k = w_k c_k.
        # grad = 2 * sum_k a_k (shift_k(x) + shift_{-k}(x))
        
        # We can compute this by iterating over k?
        # There are 2N-1 = 799 values of k.
        # For each k, we add a_k * (x shifted by k) to grad.
        # This is O(N * 2N) = O(N^2) = 160,000 operations. Very fast.
        
        a = w * c
        
        # k ranges from -(N-1) to N-1.
        # In our c array, index 0 corresponds to k = -(N-1)?
        # Let's check np.fft.irfft output ordering.
        # Linear convolution of x with x (length N) has length 2N-1.
        # c[0] = x[0]x[0]
        # c[1] = x[0]x[1] + x[1]x[0]
        # ...
        # c[N-1] = sum x_i x_{i+N-1-N+1}? No.
        # c[k] = sum_{i} x_i x_{i+k-N+1}?
        
        # Let's just use the definition:
        # c is the linear autoconvolution.
        # c[j] = sum_{i} x[i] x[i + j - (N-1)] ?
        # Actually, standard linear conv c = x * x.
        # c[k] = sum_{i=0}^{N-1} x[i] x[k-i] for k=0..2N-2.
        
        # dc_k/dx_j = x[k-j] + x[j-k] (if indices valid).
        # So term for a given k is a_k * (x[k-j] + x[j-k]).
        
        # We can compute this by:
        # For each k, we add a_k * x[k-j] to grad[j]. This is x reversed/shifted.
        # Let's use FFT for this gradient step too?
        # The operation is: grad = 2 * (a * x) + 2 * (a_rev * x_rev)?
        
        # Let's do it explicitly for clarity and speed (N=400 is small).
        # a has length 2N-1.
        # k index in a corresponds to shift k.
        # c[k] = sum_i x_i x_{k-i}.
        # dc_k/dx_j = x_{k-j} + x_{j-k}.
        
        # Let's compute S = sum_k a_k (x_{k-j} + x_{j-k})
        # Note that sum_k a_k x_{k-j} is a convolution of a and x.
        # Let v = a * x (linear conv).
        # Then v[m] = sum_k a_k x_{m-k}.
        # We want x_{k-j}. Let m = k-j => k = m+j.
        # sum_k a_k x_{k-j} = sum_m a_{m+j} x_m.
        # This is a cross-correlation or shifted conv.
        
        # Simpler:
        # term1 = sum_k a_k x_{k-j}
        # term2 = sum_k a_k x_{j-k}
        
        # Note term2 is the same as term1 if a is symmetric?
        # a_k = w_k c_k. c_k is symmetric (c_k = c_{2N-2-k}).
        # w_k is symmetric.
        # So a_k is symmetric.
        # Then term1 and term2 are identical.
        # grad = 4 * sum_k a_k x_{k-j}
        
        # How to compute sum_k a_k x_{k-j} for all j?
        # This is the convolution of a and x, evaluated at specific indices?
        # Let's just loop. 799 iterations, each O(N). 320k ops. Negligible.
        
        for k in range(2 * N - 1):
            if a[k] == 0:
                continue
            # x_{k-j}
            # Valid j: 0 <= j < N, 0 <= k-j < N
            # j >= k - (N-1) and j <= k
            j_start = max(0, k - (N - 1))
            j_end = min(N, k + 1)
            
            # For these j, x_{k-j} is x[k-j]
            # indices in x: k - j_end + 1 ... k - j_start
            idx_start = k - (j_end - 1)
            idx_end = k - j_start + 1
            
            # grad[j] += a[k] * x[k-j]
            # j ranges j_start to j_end-1
            # x index ranges idx_start to idx_end-1
            # Note: as j increases, x index decreases.
            # So we need to reverse the slice of x.
            x_slice = x[idx_start:idx_end][::-1]
            grad[j_start:j_end] += a[k] * x_slice
            
        grad *= 4.0
        
        # Constraint: sum(x) = 1.
        # We minimize f(x) subject to g(x) = sum(x) - 1 = 0.
        # We can use Lagrange multiplier or just project.
        # Since scipy minimize doesn't handle equality constraints well in BFGS without SLSQP,
        # and SLSQP is slower, let's use a trick.
        # Or use SLSQP with bounds?
        
        # Actually, let's just use SLSQP with the constraint.
        # It might be slow but N=400 is small.
        
        return obj, grad

    # We need to define the function for scipy.
    # But scipy calls fun(x) and jac(x) separately.
    
    # To avoid recomputing c and w, we could cache, but scipy might call them in different order.
    # Given N=400, recomputation is cheap.
    
    def fun(x):
        val, _ = objective_and_grad(x)
        return val

    def jac(x):
        _, g = objective_and_grad(x)
        return g

    # Initial guess
    # Random +1/-1? Or ones?
    # Let's try a few starts.
    
    best_x = None
    best_val = float('inf')
    
    rng = np.random.default_rng(42)
    
    starts = []
    # 1. All ones
    starts.append(np.ones(N) / N) # sum = 1
    # 2. Random +/- 1
    for _ in range(5):
        x_rand = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(x_rand)
        if abs(s) > 1e-6:
            starts.append(x_rand / s)
    # 3. Sinusoidal
    for f in [1, 2, 3]:
        x_sin = np.sin(2 * np.pi * f * np.arange(N) / N)
        s = np.sum(x_sin)
        if abs(s) > 1e-6:
            starts.append(x_sin / s)
        else:
            # Add small constant to make sum non-zero
            x_sin += 0.1
            s = np.sum(x_sin)
            starts.append(x_sin / s)

    deadline = time.time() + 140.0
    
    for x0 in starts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(fun, x0, method='L-BFGS-B', jac=jac, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success or res.status == 1:
                x_opt = res.x
                # Project to sum=1? L-BFGS-B doesn't enforce equality constraints.
                # We need to handle the constraint sum(x)=1.
                # L-BFGS-B only handles bounds.
                # So this will drift away from sum=1.
                
                # Let's switch to SLSQP which handles equality constraints.
                # Or use a penalty method.
                
                # Let's use SLSQP.
                # SLSQP is slower but robust for small N.
                
        except Exception:
            continue

    # Restart with SLSQP
    for x0 in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)}]
            res = minimize(fun, x0, method='SLSQP', jac=jac, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.success:
                x_opt = res.x
                # Ensure sum is exactly 1
                x_opt = x_opt / np.sum(x_opt)
                val = _metric(x_opt)
                if val < best_val:
                    best_val = val
                    best_x = x_opt.copy()
                    
        except Exception:
            continue

    if best_x is None:
        # Fallback
        best_x = np.ones(N)
        
    return best_x

# EVOLVE-BLOCK-END
