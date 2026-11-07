import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import sys

N = 100
DEADLINE = 145.0
START = time.time()

def remaining():
    return DEADLINE - (time.time() - START)

def compute_value(h):
    """Compute the exact value to minimize."""
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0

def project(h):
    """Project h to [0,1]^N with sum = N/2."""
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(100):
        s = h.sum()
        if abs(s - target) < 1e-12:
            break
        h = h + (target - s) / N
        h = np.clip(h, 0.0, 1.0)
    # Final fine-tuning
    s = h.sum()
    if abs(s - target) > 1e-10:
        diff = target - s
        free = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    i = np.arange(N)
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h) # 0.5
    
    # Soft-max parameter
    beta = 20.0
    
    def objective_and_grad(x):
        """
        x is the variable, h = x + 0.5.
        Objective: soft_max(beta, c)
        Gradient: sum_m softmax_weights_m * grad_c_m
        """
        h = x + 0.5
        # Compute correlations c
        c = np.correlate(h, 1.0 - h, mode="full")
        
        # Soft max
        c_max = np.max(c)
        # Shift for numerical stability
        c_shifted = c - c_max
        exp_c = np.exp(beta * c_shifted)
        sum_exp = np.sum(exp_c)
        obj = c_max + np.log(sum_exp) / beta
        
        # Gradient of soft max
        weights = exp_c / sum_exp # Shape (2N-1,)
        
        # Compute gradient of c w.r.t h
        # c_m = sum_i h_{i+m} (1-h_i)
        # grad_c_m[k] = (1 - h_{k-m}) - h_{k+m} (with boundaries)
        # We need sum_m weights_m * grad_c_m
        
        # Efficient computation:
        # grad_obj[k] = sum_m weights_m * ( (1 - h_{k-m})_m - (h_{k+m})_m )
        # Note: h is defined on [0, N-1]. Outside is 0.
        
        # Let's compute the two terms separately using convolution/correlation
        # Term 1: sum_m weights_m * (1 - h_{k-m})
        # h_{k-m} is h shifted.
        # If we define h_pad with zeros, we can use correlate.
        # But simpler: just loop or use vectorized ops if possible. N=100 is small.
        
        # Let's use a direct vectorized approach for the gradient
        # grad_h = np.zeros(N)
        
        # For each k in 0..N-1:
        # grad_h[k] = sum_m weights_m * ( (1 - h[k-m] if 0<=k-m<N else 1) - (h[k+m] if 0<=k+m<N else 0) )
        
        # Let's compute the "shifted h" arrays
        # A_m[k] = h[k-m] if valid else 0 (for the -h part) -> Wait, formula is 1 - h[k-m]. If invalid, h=0, so term is 1.
        # B_m[k] = h[k+m] if valid else 0.
        
        # Let's just compute it. 100*199 ops is trivial.
        grad_h = np.zeros(N)
        for m in range(2 * N - 1):
            w = weights[m]
            if w < 1e-9: continue
            # m corresponds to shift in correlate.
            # c_m = sum_i h[i+m](1-h[i])
            # Indices: i from max(0, -m) to min(N-1, N-1-m) ?
            # Actually np.correlate(h, 1-h, 'full') index m corresponds to:
            # c[m] = sum_{i} h[i+m] (1-h[i]) where valid i are such that 0<=i<N and 0<=i+m<N.
            
            # Gradient contribution to h[k]:
            # From term h[i+m]: k = i+m => i = k-m. Valid if 0<=k-m<N.
            #   Contrib: w * (1 - h[k-m])
            # From term (1-h[i]): k = i. Valid if 0<=k<N.
            #   Contrib: w * (-h[k+m])
            
            # Vectorized for this m:
            # valid_i_for_h_shifted: 0 <= k-m < N => m <= k < N+m
            # valid_i_for_h: 0 <= k < N
            
            # Let's just do it with arrays
            # Term 1: 1 - h[k-m]
            # Create an array for h_shifted_down (k-m)
            # h[k-m] is h shifted right by m.
            # If m >= 0: h[k-m] is h[0..N-1-m] at indices m..N-1, 0 elsewhere?
            # No, if k-m < 0, h is 0.
            
            # Let's use np.roll or slicing
            # Term 1: sum_m w_m * (1 - h_{k-m})
            # This is 1 * sum(w) - sum_m w_m h_{k-m}
            # sum(w) is 1.
            # So Term 1 = 1 - conv(h, weights)[k] ?
            # h_{k-m} is correlation of h with delta_m.
            # sum_m w_m h_{k-m} is conv(h, w)[k] where w is reversed?
            # Correlation c = h * w_reversed.
            # Let's just compute the sum directly for safety. N is small.
            
            # Actually, we can compute the gradient of the soft-max objective more efficiently.
            # grad_obj = sum_m weights_m * grad_c_m
            # grad_c_m[k] = delta_{k-m, valid} * (1-h_{k-m}) - delta_{k, valid} * h_{k+m}
            
            # Let's accumulate
            # Part A: sum_m weights_m * (1 - h_{k-m})
            # Part B: - sum_m weights_m * h_{k+m}
            
            # Part A is constant 1 * sum(weights) - sum_m weights_m h_{k-m}
            # sum(weights) = 1.
            # So Part A[k] = 1 - (h * weights_reversed)[k] ?
            # h[k-m] is h shifted by m.
            # sum_m w_m h[k-m] is conv(h, w)[k] if w is indexed by m.
            # Let's use np.convolve(h, weights, mode='same')? No, weights length is 2N-1.
            
            # Let's stick to the loop for clarity and correctness, it's fast enough.
            # But we can optimize the inner loop.
            pass
            
        # Optimized Gradient Calculation:
        # grad_h = np.ones(N) * np.sum(weights) # Which is 1
        # Then subtract contributions.
        
        # Contribution 1: - sum_m weights_m h[k-m]
        # This is conv(h, weights)[k] with appropriate alignment.
        # weights[m] corresponds to shift m.
        # h[k-m] means h shifted right by m.
        # Let h_pad be h with zeros.
        # We can use correlate(h, weights, mode='full')?
        # c_full[m] = sum_i h[i+m] weights[i] ? No.
        
        # Let's use the property:
        # sum_m w_m h[k-m] = (h * w)[k] where * is convolution.
        # If w is defined on -N..N, and h on 0..N.
        
        # Let's just use the loop, it's robust.
        for m in range(2 * N - 1):
            w = weights[m]
            if w < 1e-9: continue
            
            # Part A: (1 - h[k-m])
            # k-m is valid index for h if 0 <= k-m < N
            # This is h shifted right by m.
            # h_shifted[k] = h[k-m] if 0<=k-m<N else 0
            # We want sum_m w_m (1 - h_shifted_m[k])
            # = sum_m w_m - sum_m w_m h_shifted_m[k]
            # = 1 - sum_m w_m h_shifted_m[k]
            
            # Part B: - h[k+m]
            # h_shifted2[k] = h[k+m] if 0<=k+m<N else 0
            # = - sum_m w_m h_shifted2_m[k]
            
            # Let's compute the sums
            # sum_m w_m h[k-m]
            # k-m in [0, N-1] => m in [k-N+1, k]
            # sum_m w_m h[k+m]
            # k+m in [0, N-1] => m in [-k, N-1-k]
            
            # This is getting complicated to vectorize without errors.
            # Given N=100, the loop over 199 m's is fine.
            
            # Inside the loop, we can update grad_h vectorized for this m?
            # No, m is fixed, k varies.
            
            # Let's just compute the whole gradient in one go using np.correlate
            # grad_c_m[k] = (1 - h[k-m]) - h[k+m]
            # sum_m w_m grad_c_m[k]
            
            # Term 1: sum_m w_m (1 - h[k-m]) = 1 - sum_m w_m h[k-m]
            # sum_m w_m h[k-m] is conv(h, w)[k] ?
            # Let's check indices.
            # conv(h, w)[k] = sum_i h[i] w[k-i].
            # Let j = k-i => i = k-j. sum_j h[k-j] w[j].
            # This matches sum_m w_m h[k-m].
            # So Term 1 = 1 - conv(h, w)[k].
            
            # Term 2: - sum_m w_m h[k+m]
            # sum_m w_m h[k+m] = sum_j w_j h[k+j].
            # Let j' = -j. sum_j' w_{-j'} h[k-j'].
            # This is conv(h, w_reversed)[k].
            # So Term 2 = - conv(h, w_reversed)[k].
            
            # So grad_h[k] = 1 - conv(h, w)[k] - conv(h, w_reversed)[k].
            
            # We need conv(h, w) and conv(h, w_reversed) for the full range of k (0..N-1).
            # w has length 2N-1.
            # conv(h, w) has length N + 2N-1 - 1 = 3N-1.
            # We need the part corresponding to k in 0..N-1.
            
            # Let's compute these convolutions once.
            break # Break out to do it outside the loop
            
        # Compute convolutions
        # w is weights (length 2N-1)
        # w_reversed is w[::-1]
        
        # conv1 = np.convolve(h, weights, mode='full')
        # conv2 = np.convolve(h, weights[::-1], mode='full')
        
        # We need to align them correctly.
        # conv1[k] = sum_i h[i] weights[k-i].
        # We want sum_m weights[m] h[k-m].
        # Let i = k-m => m = k-i.
        # sum_i h[i] weights[k-i].
        # This is exactly conv1[k] IF weights is indexed 0..2N-2 and h 0..N-1.
        # But weights[m] is defined for m in 0..2N-2.
        # In the correlation formula, m is the shift.
        # c_m = sum_i h[i+m](1-h[i]).
        # The index m in np.correlate(h, 1-h, 'full') goes from 0 to 2N-2.
        # The shift is effectively (m - (N-1))?
        # No, np.correlate(h, v, 'full')[m] = sum_i h[i+m] v[i] ?
        # Actually, np.correlate(h, v, 'full')[k] = sum_i h[i+k] v[i].
        # So m in our formula corresponds to index k in the array.
        
        # So weights[m] is the weight for the term h[i+m]v[i].
        
        # Term 1: sum_m weights[m] h[k-m].
        # This is not a standard convolution with weights[m] because of the negative shift in h.
        # h[k-m] = h[i] with i=k-m.
        # sum_m weights[m] h[k-m] = sum_i h[i] weights[k-i].
        # This IS conv(h, weights)[k].
        
        # Term 2: sum_m weights[m] h[k+m].
        # h[k+m] = h[i] with i=k+m => m=i-k.
        # sum_m weights[m] h[k+m] = sum_i h[i] weights[i-k].
        # This is conv(h, weights_reversed)[k] ?
        # conv(h, w_rev)[k] = sum_i h[i] w_rev[k-i] = sum_i h[i] w_rev[(k-i)].
        # w_rev[j] = w[-j]? No, w_rev is just reversed array.
        # w_rev[j] = w[L-1-j].
        # We want weights[i-k].
        # If we define w_neg[m] = weights[-m]?
        # Let's just use the loop for the gradient to avoid alignment bugs. N=100 is small.
        
        grad_h = np.zeros(N)
        for m in range(2 * N - 1):
            w = weights[m]
            if w < 1e-9: continue
            
            # Part A: 1 - h[k-m]
            # Valid k: 0 <= k-m < N => m <= k < N+m
            # h[k-m] is h shifted right by m.
            # Let's construct the vector h_shifted = np.zeros(N)
            # h_shifted[k] = h[k-m] if 0<=k-m<N else 0
            
            # Part B: -h[k+m]
            # Valid k: 0 <= k+m < N => -m <= k < N-m
            # h[k+m] is h shifted left by m.
            
            # We can use np.roll or slicing.
            
            # Part A contribution: w * (1 - h_shifted_right)
            # Part B contribution: w * (-h_shifted_left)
            
            # h_shifted_right[k] = h[k-m]
            # If m >= 0:
            #   h_shifted_right[m:N] = h[0:N-m]
            #   h_shifted_right[0:m] = 0
            
            # h_shifted_left[k] = h[k+m]
            # If m >= 0:
            #   h_shifted_left[0:N-m] = h[m:N]
            #   h_shifted_left[N-m:N] = 0
            
            # Note: m in np.correlate 'full' goes 0 to 2N-2.
            # The shift in c_m = sum h[i+m](1-h[i]) is effectively m.
            # But wait, np.correlate(h, v, 'full')[m] uses h[i+m].
            # So m is the shift.
            
            # However, the range of m is 0..2N-2.
            # For m >= N, h[i+m] is always 0?
            # If m >= N, i+m >= N for all i>=0. So c_m = 0.
            # So weights[m] will be ~0 for m >= N (unless h is all 0, but sum is N/2).
            # Actually, if m is large, c_m is small.
            
            # Let's handle m < N and m >= N separately or just generally.
            
            # General case for h[k-m]:
            # idx = k - m
            # valid if 0 <= idx < N
            
            # Let's just compute the arrays
            h_right = np.zeros(N)
            h_left = np.zeros(N)
            
            if m < N:
                # h[k-m] non-zero for k >= m
                h_right[m:] = h[:N-m]
                
                # h[k+m] non-zero for k < N-m
                h_left[:N-m] = h[m:]
            else:
                # m >= N
                # h[k-m]: k-m < N => k < N+m. k >= 0.
                # k-m >= 0 => k >= m.
                # If m >= N, k >= m >= N. But k < N. So no valid k.
                # So h_right is all 0.
                
                # h[k+m]: k+m < N => k < N-m.
                # If m >= N, N-m <= 0. So no valid k.
                # So h_left is all 0.
                
                # Wait, if m >= N, c_m = 0. So weight should be 0.
                # So this case won't contribute.
                pass
            
            grad_h += w * (1.0 - h_right) # Part A
            grad_h += w * (-h_left) # Part B
            
        return obj, grad_h

    # Bounds for x
    bounds = [(-0.5, 0.5)] * N
    
    # Constraints
    # Sum x = 0
    # We use trust-constr
    
    constraint_sum = LinearConstraint(np.ones(N), 0, 0)
    constraints = [constraint_sum]
    
    starts = [np.zeros(N)]
    for k in range(1, 8):
        for A in [0.2, 0.4]:
            starts.append(A * np.sin(2 * np.pi * k * i / N))
            starts.append(A * np.cos(2 * np.pi * k * i / N))
            
    np.random.seed(42)
    for _ in range(5):
        x_rand = np.random.randn(N)
        x_rand -= np.mean(x_rand)
        x_rand /= np.max(np.abs(x_rand)) * 2
        starts.append(x_rand)
        
    for x0 in starts:
        if remaining() < 10:
            break
            
        # Project initial guess
        x0 = np.clip(x0, -0.5, 0.5)
        s = np.sum(x0)
        if abs(s) > 1e-10:
            x0 = x0 - s / N
            x0 = np.clip(x0, -0.5, 0.5)
            for _ in range(10):
                s = np.sum(x0)
                if abs(s) < 1e-10: break
                x0 = x0 - s / N
                x0 = np.clip(x0, -0.5, 0.5)
                
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='trust-constr',
                jac=True, # We return grad
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'gtol': 1e-8}
            )
            
            x_opt = res.x
            h_opt = x_opt + 0.5
            h_opt = project(h_opt)
            val = compute_value(h_opt)
            
            if val < best_val:
                best_val = val
                best_h = h_opt
                
        except Exception as e:
            pass
            
    return best_h
