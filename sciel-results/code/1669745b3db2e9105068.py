"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct step heights h[0..N-1] in [0,1] with sum N/2 to minimize
    the maximum overlap of h with shifted 1-h.
    
    The objective is min_h max_k sum_i h[i] * (1 - h[i+k mod N])
    which is equivalent to min_h max_k (sum_i h[i] - sum_i h[i]*h[i+k])
    Since sum_i h[i] = N/2 is fixed, this is min_h max_k (N/2 - C_k)
    where C_k = sum_i h[i]*h[i+k mod N] is the circular autocorrelation at lag k.
    So we want to maximize min_k C_k.
    
    This is a non-convex problem. We use a smooth surrogate:
    maximize the soft-min of C_k using log-sum-exp style.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Initial guess: alternating 0.5 (which gives C_k = N/4 for all k, overlap = N/4)
    # Actually, let's think more carefully.
    # If h[i] = 0.5 for all i, then C_k = sum 0.5*0.5 = N*0.25 = N/4
    # Overlap at lag k = N/2 - N/4 = N/4 = 25 for N=100
    # Can we do better?
    
    # Let's try to use gradient-based optimization with a smooth max surrogate.
    # Objective: f(h) = logsumexp(C_k / tau) / N  (minimize this, or equivalently maximize min C_k)
    # Actually we want to minimize max_k (N/2 - C_k) = N/2 - min_k C_k
    # So minimize: N/2 - softmin(C_k) where softmin(x) = -logsumexp(-x/tau)
    
    def compute_C(h):
        """Compute circular autocorrelation C_k = sum_i h[i]*h[(i+k)%N] for all k."""
        # Use FFT for efficiency
        H = np.fft.fft(h)
        C = np.fft.ifft(H * np.conj(H)).real
        # C[k] = sum_i h[i] * h[i+k mod N] ... need to verify convention
        # Actually np.fft.ifft(|FFT(h)|^2) gives circular autocorrelation
        # C[k] = sum_i h[i] * h[(i+k) % N]
        return C
    
    def objective_and_grad(h):
        """Compute objective (softmax of overlaps) and gradient."""
        C = compute_C(h)
        # Overlaps: O_k = N/2 - C_k
        # We want to minimize max O_k = N/2 - min C_k
        # Soft surrogate: f = N/2 + logsumexp(-C/tau) / N ... let's use:
        # f = -softmin(C) where softmin(C) = -tau * logsumexp(-C/tau)
        # So f = tau * logsumexp(-C/tau)
        
        tau = 5.0  # Temperature parameter
        
        neg_C_over_tau = -C / tau
        # logsumexp with numerical stability
        m = np.max(neg_C_over_tau)
        lse = m + np.log(np.sum(np.exp(neg_C_over_tau - m)))
        f = tau * lse
        
        # Gradient of f w.r.t. h
        # f = tau * logsumexp(-C/tau)
        # df/dC_k = -exp(-C_k/tau) / sum_j exp(-C_j/tau)
        weights = np.exp(neg_C_over_tau - m)
        weights /= np.sum(weights)
        # weights[k] = exp(-C_k/tau) / sum exp(-C_j/tau)
        # df/dC_k = -weights[k]
        
        # Now C_k = sum_i h[i] * h[(i+k)%N]
        # dC_k/dh[j] = h[(j+k)%N] + h[(j-k)%N] (for k != 0)
        # dC_0/dh[j] = 2*h[j]
        # 
        # So df/dh[j] = sum_k (df/dC_k) * (dC_k/dh[j])
        #             = sum_k (-weights[k]) * (h[(j+k)%N] + h[(j-k)%N])  for k>0
        #             + (-weights[0]) * 2*h[j]
        
        # This is a convolution-like operation. Let's use FFT.
        # df/dh[j] = -sum_k weights[k] * h[(j+k)%N] - sum_k weights[k] * h[(j-k)%N] + weights[0]*2*h[j] ... wait
        # Let me redo:
        # dC_k/dh[j] = h[(j+k)%N] + h[(j-k)%N] if k != 0, and 2*h[j] if k=0
        # Actually for k=0: C_0 = sum_i h[i]^2, dC_0/dh[j] = 2*h[j]
        # For k!=0: C_k = sum_i h[i]*h[(i+k)%N], dC_k/dh[j] = h[(j+k)%N] + h[(j-k)%N]
        
        # df/dh[j] = sum_k (df/dC_k) * dC_k/dh[j]
        #           = sum_k (-weights[k]) * dC_k/dh[j]
        
        # Let W = weights (length N), W[k] as above
        # df/dh = -convolution of W with h, but careful with the k=0 term
        # Actually: sum_k W[k] * h[(j+k)%N] is circular convolution (W * h)[j]
        # sum_k W[k] * h[(j-k)%N] is circular correlation, which is (W * h[::-1])[j] reversed... 
        # Actually sum_k W[k] * h[(j-k)%N] = (W * h)[j] where * is circular convolution if we define it properly.
        # Let's just use FFT:
        # A[j] = sum_k W[k] * h[(j+k)%N] = circular cross-correlation
        # This is ifft(fft(W) * conj(fft(h)))... let me just do it directly with FFT
        # sum_k W[k] * h[(j+k)%N]: let l = j+k, k = l-j, sum_l W[l-j] * h[l]
        # This is circular convolution of W and h with W reversed? 
        # Actually: (W * h)[j] = sum_k W[k] * h[j-k]
        # We want sum_k W[k] * h[j+k] = sum_k W[k] * h[j-(-k)] = sum_m W[-m] * h[j-m] = (W_rev * h)[j]
        # where W_rev[k] = W[-k mod N]
        
        W_rev = np.roll(W, 1)  # W_rev[k] = W[(k-1)%N]... let me think again
        # W_rev[k] should be W[(-k)%N] = W[N-k] for k>0, W[0] for k=0
        W_rev = np.concatenate([W[0], W[:0:-1]])
        
        WH = np.fft.fft(W_rev) * np.fft.fft(h)
        A = np.fft.ifft(WH).real  # A[j] = sum_k W_rev[k] * h[j-k] = sum_k W[-k] * h[j-k] = sum_m W[m] * h[j+m]
        
        # Similarly, sum_k W[k] * h[j-k] = (W * h)[j]
        WH2 = np.fft.fft(W) * np.fft.fft(h)
        B = np.fft.ifft(WH2).real  # B[j] = sum_k W[k] * h[j-k]
        
        # df/dh[j] = -A[j] - B[j] + 2*W[0]*h[j]
        # Wait, the k=0 term: for k=0, dC_0/dh[j] = 2*h[j], and df/dC_0 = -W[0]
        # For k!=0: dC_k/dh[j] = h[j+k] + h[j-k]
        # So df/dh[j] = -W[0]*2*h[j] - sum_{k!=0} W[k]*(h[j+k] + h[j-k])
        #             = -2*W[0]*h[j] - (A[j] - W[0]*h[j]) - (B[j] - W[0]*h[j])
        #             = -2*W[0]*h[j] - A[j] + W[0]*h[j] - B[j] + W[0]*h[j]
        #             = -A[j] - B[j]
        
        grad = -A - B
        
        # Project gradient onto feasible set (sum h = N/2, 0<=h<=1)
        # For now, just return objective and gradient, use projected gradient
        return f, grad
    
    # Projected gradient descent
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_val = np.inf
    
    lr = 0.01
    tau_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.25]
    
    for tau_idx, tau in enumerate(tau_values):
        if time.time() > deadline:
            break
            
        # Reset learning rate for each tau
        lr = 0.01 / (1 + tau_idx * 0.5)
        
        for it in range(2000):
            if time.time() > deadline:
                break
                
            f, grad = objective_and_grad(h)
            
            # Projected gradient step
            h_new = h - lr * grad
            # Project onto {sum = N/2, 0 <= h <= 1}
            # Simple projection: clip and adjust mean
            h_new = np.clip(h_new, 0, 1)
            # Adjust to make sum = N/2
            diff = N / 2 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            # Second projection pass
            diff = N / 2 - np.sum(h_new)
            h_new += diff / N
            h_new = np.clip(h_new, 0, 1)
            
            h = h_new
            
            # Evaluate actual objective (max overlap)
            C = compute_C(h)
            overlaps = N / 2 - C
            max_overlap = np.max(overlaps)
            
            if max_overlap < best_val:
                best_val = max_overlap
                best_h = h.copy()
    
    # Final polish: use scipy minimize with SLSQP for a few iterations
    if time.time() < deadline - 20:
        try:
            def obj_func(h_flat):
                h = h_flat.reshape(N)
                C = compute_C(h)
                overlaps = N / 2 - C
                return np.max(overlaps)
            
            def obj_grad(h_flat):
                h = h_flat.reshape(N)
                C = compute_C(h)
                # Gradient of max overlap is tricky, use subgradient
                k_max = np.argmax(N / 2 - C)
                # d/dh[j] of (N/2 - C_kmax)
                # dC_kmax/dh[j] = h[(j+kmax)%N] + h[(j-kmax)%N] if kmax != 0, else 2*h[j]
                if k_max == 0:
                    grad_C = 2 * h
                else:
                    grad_C = np.roll(h, -k_max) + np.roll(h, k_max)
                return -grad_C.reshape(-1)
            
            result = minimize(
                obj_func, best_h, jac=obj_grad,
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}],
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.success:
                h_candidate = np.clip(result.x, 0, 1)
                # Ensure sum constraint
                h_candidate += (N/2 - np.sum(h_candidate)) / N
                h_candidate = np.clip(h_candidate, 0, 1)
                C = compute_C(h_candidate)
                overlaps = N / 2 - C
                max_overlap = np.max(overlaps)
                if max_overlap < best_val:
                    best_h = h_candidate.copy()
                    best_val = max_overlap
        except Exception:
            pass
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    best_h += (N / 2 - np.sum(best_h)) / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
