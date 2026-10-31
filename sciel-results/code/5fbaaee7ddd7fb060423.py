import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 400
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv(f):
    """Compute the linear (full) autoconvolution of f."""
    return np.convolve(f, f, mode='full')


def _objective_vectorized(f):
    """
    Compute max(|autoconv(f)|) / (sum(f))^2.
    If sum(f) == 0, return a large penalty.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _objective_smooth(f, t=0.01):
    """
    Smooth surrogate using log-sum-exp for the max.
    Objective: (exp(t * logsumexp(|C|)) / (sum(f))^2)^(1/t)
    This is differentiable and approximates the original as t -> 0.
    """
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    absC = np.abs(C)
    # log-sum-exp trick
    max_val = np.max(absC)
    logsumexp = max_val + np.log(np.sum(np.exp(absC - max_val)))
    # The objective is (exp(t * logsumexp) / S^2)^(1/t)
    # = exp(logsumexp) / S^2 = max_val * exp(log(sum(exp(absC - max_val)))) / S^2
    # Actually, let's just use: max_val * (1 + t * variance_term) as a smooth approx?
    # Or simply: (1/t) * log( (1/n) * sum(exp(t * absC)) ) / S^2
    # The log-sum-exp function is LSE(t, x) = log(sum(exp(t*x))).
    # If we want max(x), as t->inf, LSE/t -> max(x).
    # So let's minimize (LSE(t, absC) / t) / S^2.
    lse = max_val + np.log(np.sum(np.exp(t * (absC - max_val))))
    return (lse / t) / (S * S)


def _gradient_smooth(f, t=0.01):
    """Gradient of the smooth objective."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros_like(f)
    
    C = _autoconv(f)
    absC = np.abs(C)
    signC = np.sign(C)
    max_val = np.max(absC)
    
    # LSE = max_val + log(sum(exp(t*(absC - max_val))))
    exp_term = np.exp(t * (absC - max_val))
    sum_exp = np.sum(exp_term)
    
    # d/d(absC_i) [ LSE ] = t * exp(t*(absC_i - max_val)) / sum_exp
    # But LSE itself depends on max_val if max_val is not unique, 
    # however the standard log-sum-exp gradient is stable.
    # Let's use the standard formulation:
    # LSE = log(sum(exp(t * absC)))
    # d LSE / d absC_i = t * exp(t * absC_i) / sum(exp(t * absC))
    
    # To avoid overflow, we subtract max_val.
    # LSE = max_val + log(sum(exp(t * (absC - max_val))))
    # d LSE / d absC_i = t * exp(t * (absC_i - max_val)) / sum(exp(t * (absC - max_val)))
    
    weights = exp_term / sum_exp
    
    # d/d absC_i (LSE / t) = weights_i
    # d/d f ( LSE/t / S^2 ) = (1/S^2) * d/d f (LSE/t) - 2*S_inv2 * S_inv * d/d f (LSE/t) * S?
    # Let J = LSE(t, absC).
    # Obj = (J/t) / S^2 = J / (t * S^2)
    # d Obj / d f = (1/(t*S^2)) * dJ/df - (J/(t*S^4)) * d(S^2)/df
    # d(S^2)/df = 2 * S * ones
    # dJ/df = sum_k (dJ/d absC_k) * (d absC_k / df)
    # d absC_k / df_i = sign(C_k) * d C_k / df_i
    # C_k = sum_{i+j=k} f_i f_j
    # d C_k / df_i = sum_{j: i+j=k} f_j  (for linear convolution, this is a convolution with reversed f)
    
    # Let's compute dC/df.
    # C = conv(f, f)
    # dC/df_i is the vector of partials.
    # Since C is symmetric (autoconv), dC_k/df_i = f_{k-i} + f_{i-k} ? No.
    # C_k = sum_j f_j f_{k-j}.
    # d C_k / d f_i = f_{k-i} + f_{k-i} ? 
    # If we differentiate w.r.t f_i:
    # Term where j=i: f_{k-i}
    # Term where k-j=i => j=k-i: f_i f_{k-i} -> derivative is f_{k-i}
    # So d C_k / d f_i = f_{k-i} + f_{k-i} = 2 f_{k-i} if both terms exist?
    # Wait, if i != k-i, they are distinct terms.
    # If i = k-i, it's one term f_i^2, derivative 2 f_i.
    # Generally, d C_k / d f_i = f_{k-i} + f_{k-i} is not quite right for boundaries.
    # It is the convolution of f with a vector that has 1 at position i?
    # Actually, d C / d f = conv(f, delta_i) + conv(f, delta_i) = 2 * (f shifted)
    # Specifically, the gradient of C_k w.r.t f is a vector v_k such that v_k[i] = d C_k / d f_i.
    # v_k[i] = f_{k-i} (if 0<=k-i<N) + f_{k-i} (if 0<=k-i<N)?
    # Let's just use the fact that C = conv(f,f).
    # The Jacobian J_{k,i} = d C_k / d f_i.
    # This is symmetric in i and k-i.
    # We can compute dJ/df by computing the gradient of each C_k.
    # For efficiency, note that dJ/df is a 2N-1 x N matrix.
    # J_{k,i} = f_{k-i} + f_{k-i} ?? No.
    # C_k = \sum_{j=0}^{N-1} f_j f_{k-j}
    # \frac{\partial C_k}{\partial f_i} = \sum_{j} \delta_{j,i} f_{k-j} + f_j \delta_{k-j,i}
    # = f_{k-i} + \sum_j f_j \delta_{j, k-i}
    # = f_{k-i} + f_{k-i}
    # So yes, it is 2 f_{k-i} (with appropriate boundary handling where indices are out of range, the term is 0).
    # Wait, if k-i is out of [0, N-1], then f_{k-i} is 0.
    # So J_{k,i} = f_{k-i} + f_{k-i} = 2 f_{k-i} (if 0 <= k-i < N) else 0?
    # Let's check:
    # C_k = \sum_{j} f_j f_{k-j}
    # If we change f_i, only terms where j=i or k-j=i are affected.
    # j=i -> term is f_i f_{k-i}. Deriv w.r.t f_i is f_{k-i}.
    # k-j=i -> j=k-i -> term is f_{k-i} f_i. Deriv w.r.t f_i is f_{k-i}.
    # So total derivative is 2 f_{k-i}.
    # This holds for all i, j in range. If k-i is out of range, f_{k-i}=0.
    
    # So dJ/df can be computed efficiently.
    # We need sum_k (weights_k * sign(C_k)) * J_k
    # where J_k[i] = 2 f_{k-i} (if valid)
    
    # Let A = weights * signC
    # grad_f_i = sum_k A_k * 2 f_{k-i}
    # This is a convolution of A with f (reversed)?
    # sum_k A_k f_{k-i} is (A * f)[i] ?
    # Convolution (A * f)[i] = sum_k A_k f[i-k].
    # Here we have sum_k A_k f_{k-i}.
    # Let m = k-i => k = m+i.
    # sum_m A_{m+i} f_m.
    # This is cross-correlation.
    # np.correlate(A, f, mode='valid')? No, A is length 2N-1, f is N.
    
    # Let's just do it with loops or vectorized shifts for clarity and correctness, 
    # or use scipy.signal.
    # Given N=400, a loop over i might be slow if done naively in Python, 
    # but we can vectorize the convolution.
    
    # Let's construct the gradient term by term.
    # grad_smooth_part = sum_k (A_k * 2 f_{k-i})
    # This is equivalent to 2 * (A_reversed * f)[i]?
    
    # Let's use a simple loop for the gradient computation to ensure correctness, 
    # as N=400 is small. 400 iterations of vector ops is fast.
    
    grad = np.zeros_like(f)
    for i in range(N):
        # J_k[i] is non-zero when 0 <= k-i < N => i <= k < i+N
        k_start = i
        k_end = i + N
        A_slice = A[k_start:k_end]
        f_slice = f[0:N] # f_{k-i} for k in [i, i+N-1] is f[0], f[1]... f[N-1]
        # Wait, k-i ranges from 0 to N-1.
        # So f_{k-i} is just f[0:N]
        # So sum_k A_k f_{k-i} = sum_{m=0}^{N-1} A_{i+m} f_m
        # This is dot(A[i:i+N], f)
        grad[i] = 2.0 * np.dot(A[i:i+N], f)
        
    dJ_df = grad
    
    dObj_df = (1.0 / (t * S * S)) * dJ_df - (J_val / (t * S * S * S)) * 2.0 * S
    # Wait, J_val = t * LSE? No, I defined Obj = (LSE/t) / S^2.
    # Let G = LSE/t.
    # Obj = G / S^2.
    # dObj/df = (1/S^2) dG/df - (G/S^4) d(S^2)/df
    # dG/df = d(LSE/t)/df = (1/t) dLSE/df.
    # dLSE/df = sum_k (dLSE/d absC_k) * sign(C_k) * dC_k/df
    # dLSE/d absC_k = weights_k (as derived above, which is exp(t(absC-max))/sum)
    # Wait, earlier I said dLSE/df = sum_k weights_k * sign(C_k) * J_k.
    # So dG/df = (1/t) * sum_k weights_k * sign(C_k) * J_k.
    
    # Let's re-evaluate.
    # LSE = log(sum(exp(t absC)))
    # dLSE/d absC_k = t exp(t absC_k) / sum(exp(t absC)) = t * weights_k
    # So dLSE/df = sum_k (t weights_k) * sign(C_k) * J_k
    # dG/df = (1/t) * dLSE/df = sum_k weights_k * sign(C_k) * J_k
    # This matches my `grad` variable above (which was 2 * sum ...).
    
    G_val = LSE_val / t
    
    return (1.0 / (S * S)) * grad - (G_val / (S * S * S)) * 2.0 * S

N = 400

def construct():
    start_time = time.time()
    best_f = np.ones(N)
    best_score = _objective_vectorized(best_f)
    
    # Try a few random initializations
    rng = np.random.default_rng(42)
    
    for restart in range(5):
        if time.time() > DEADLINE:
            break
            
        f0 = rng.uniform(-1, 1, N)
        # Normalize sum to 1 to avoid scaling issues
        if abs(np.sum(f0)) > 1e-6:
            f0 = f0 / np.sum(f0)
        else:
            f0 = np.ones(N)
            
        # Use smooth objective with decreasing t
        ts = [0.1, 0.05, 0.01, 0.005]
        current_f = f0
        
        for t in ts:
            if time.time() > DEADLINE:
                break
                
            res = minimize(
                _objective_smooth, 
                current_f, 
                args=(t,),
                method='L-BFGS-B',
                jac=None, # Let it approximate or use our custom grad if available
                options={'maxiter': 20, 'maxls': 20}
            )
            
            if res.success and res.fun < best_score:
                # Verify with true objective
                true_score = _objective_vectorized(res.x)
                if true_score < best_score:
                    best_score = true_score
                    best_f = res.x
                current_f = res.x
                
    # Final check
    return best_f

# EVOLVE-BLOCK-END
