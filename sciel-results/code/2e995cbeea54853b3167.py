import numpy as np
import time
import sys

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_value(h):
        """Compute the objective value (lower is better)."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    def compute_conv(h):
        return np.convolve(h, h)

    # --- Phase 1: Energy Minimization via Frequency Domain ---
    # The L2 norm of the autocorrelation is related to the L4 norm of the signal.
    # However, a simpler smooth proxy is minimizing the energy of the autocorrelation.
    # E = sum(conv^2).
    # In frequency domain, autocorrelation corresponds to |H|^2.
    # Let H_k = FFT(h).
    # conv = IFFT(|H|^2) ? No, conv = IFFT(H * H_rev) where H_rev is reversed.
    # Actually, circular autocorrelation r_c[k] = IFFT(|FFT(h)|^2).
    # Linear autocorrelation is slightly different, but for large N, circular is a good approximation
    # if we treat the signal as periodic or if we use the "zero-padded" FFT trick.
    # Let's use the standard property:
    # Linear autocorrelation c = convolve(h, h[::-1])
    # FFT(c) = FFT(h) * conj(FFT(h)) = |FFT(h)|^2
    # Wait, np.convolve(h, h) is not exactly autocorrelation if h is not symmetric.
    # Autocorrelation is usually convolve(h, h[::-1]).
    # The task defines conv = np.convolve(h, h).
    # If we set h to be symmetric (h[i] = h[N-1-i]), then conv(h, h) is the autocorrelation.
    # Let's enforce symmetry to simplify the math and reduce degrees of freedom.
    # If h is symmetric, h = h[::-1].
    # Then conv(h, h) is the autocorrelation.
    # Let's try optimizing the L2 norm of this autocorrelation.
    
    # We parameterize h by its first N/2 + 1 elements to enforce symmetry.
    # h_full[0:N//2+1] = x
    # h_full[N//2:N] = x[1:N//2][::-1] (for even N)
    # N=400.
    # h[0], h[1], ..., h[199], h[200], h[199], ..., h[1]
    # Wait, symmetric means h[i] = h[N-1-i].
    # h[0] = h[399], h[1] = h[398], ..., h[199] = h[200].
    # So we need 200 unique values.
    
    M = N // 2
    x = np.ones(M) # Initial guess: constant (symmetric)
    
    def expand_symmetric(x):
        h = np.empty(N)
        h[:M] = x
        h[M:] = x[:M-1][::-1] # Wait, h[M] corresponds to h[N-1-M] = h[M-1]?
        # Let's check indices.
        # N=4, M=2. h[0], h[1], h[2], h[3].
        # Symmetric: h[0]=h[3], h[1]=h[2].
        # x = [x0, x1].
        # h[0]=x0, h[1]=x1.
        # h[2]=h[1]=x1, h[3]=h[0]=x0.
        # So h[M:] should be x[1:M][::-1]?
        # x[1:2] is [x1]. reversed is [x1].
        # h[2] = x1. Correct.
        # h[3] = x0.
        # My formula: h[M:] = x[:M-1][::-1]
        # x[:1] is [x0]. reversed is [x0].
        # So h[2] = x0. Incorrect.
        
        # Correct construction for even N:
        # h[0:M] = x[0:M]
        # h[M:N] = x[1:M][::-1]
        # Let's re-verify.
        # h[M] should equal h[N-1-M] = h[M-1] = x[M-1].
        # x[1:M][::-1] last element is x[1]. First element is x[M-1].
        # So h[M] gets x[M-1]. Correct.
        # h[N-1] gets x[1]. But h[N-1] should equal h[0] = x[0].
        # So h[M:N] = x[:M-1][::-1] ?
        # x[:M-1] is x[0:M-1]. Reversed is x[M-2]...x[0].
        # Length is M-1.
        # h[M:N] has length N-M = M.
        # So we need M elements.
        # h[M] = x[M-1]
        # h[M+1] = x[M-2]
        # ...
        # h[N-1] = x[0]
        # So h[M:N] = x[::-1]
        # Let's check:
        # h[M] = x[M-1]. Correct.
        # h[N-1] = x[0]. Correct.
        # So h[M:] = x[::-1]
        
        h[:M] = x
        h[M:] = x[::-1]
        return h

    # Objective: Minimize sum(conv^2)
    # conv = np.convolve(h, h)
    # Since h is symmetric, conv is symmetric.
    # We can use FFT to compute conv efficiently if we zero-pad.
    # But N=400 is small, np.convolve is fast enough.
    
    # Let's use gradient descent on x (200 dims) to minimize Energy = sum(conv^2)
    # Gradient of E w.r.t h is 2 * convolve(conv, h[::-1]) ?
    # d/dh_j sum_k (sum_i h_i h_{k-i})^2
    # This is getting messy. Let's just use numerical gradient or a simple optimizer.
    # Better: Use the fact that min sum(conv^2) is equivalent to min ||h||_4^4 * N?
    # Parseval: sum |conv|^2 = N * sum |FFT(conv)|^2 ? No.
    # Let's stick to direct numerical optimization on x using L-BFGS-B from SciPy.
    
    try:
        from scipy.optimize import minimize
        
        def energy_obj(x):
            if time.time() > deadline - 5.0:
                return 0.0
            h = expand_symmetric(x)
            s = np.sum(h)
            if abs(s) < 1e-6:
                return 1e10
            conv = np.convolve(h, h)
            return np.sum(conv**2)
        
        # Initial guess: ones
        x0 = np.ones(M)
        
        res = minimize(energy_obj, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        if res.success and time.time() < deadline - 20.0:
            best_x = res.x
        else:
            best_x = x0
            # Fallback to simple gradient descent if L-BFGS fails or times out
            lr = 0.01
            for i in range(1000):
                if time.time() > deadline - 20.0: break
                h = expand_symmetric(best_x)
                conv = np.convolve(h, h)
                # Numerical gradient
                grad_x = np.zeros(M)
                eps = 1e-5
                for j in range(M):
                    xp = best_x.copy()
                    xp[j] += eps
                    hp = expand_symmetric(xp)
                    conv_p = np.convolve(hp, hp)
                    grad_x[j] = (np.sum(conv_p**2) - energy_obj(best_x)) / eps
                best_x -= lr * grad_x
                # Project to keep sum non-zero?
                if abs(np.sum(expand_symmetric(best_x))) < 1e-6:
                    best_x = x0
                    break
                    
    except Exception as e:
        best_x = np.ones(M)

    h = expand_symmetric(best_x)
    # Normalize h to have sum 1 to simplify scaling?
    # The objective is scale invariant: value = abs(2*N*max(conv)/sum(h)^2)
    # If we scale h by k, conv scales by k^2, sum(h)^2 scales by k^2. Value is invariant.
    # So we can fix sum(h) = 1.
    s = np.sum(h)
    if abs(s) > 1e-6:
        h = h / s
    else:
        h = np.ones(N) / N

    current_val = compute_value(h)
    best_h = h.copy()
    best_val = current_val
    
    # --- Phase 2: Sub-gradient Descent on Max Objective ---
    # Objective: f(h) = max(conv) / sum(h)^2
    # We want to minimize this.
    # Let C = conv(h, h).
    # Let M = max(C).
    # Let S = sum(h).
    # f = M / S^2
    # Gradient of f w.r.t h:
    # df/dh = (1/S^2) * dM/dh - (2M/S^3) * dS/dh
    # dS/dh = 1
    # dM/dh: If M is at index k_max, dM/dh_j = h[k_max-j] (assuming valid indices)
    # Actually, C[k] = sum_i h[i] h[k-i].
    # dC[k]/dh[j] = h[k-j] + (if k-j != j) ... no, derivative of h[i]h[k-i] w.r.t h[j]
    # is h[k-j] if i=j, plus h[j] if k-i=j => i=k-j.
    # So dC[k]/dh[j] = h[k-j] + h[j] (if k-j != j, else 2*h[j] if k=2j? No, chain rule)
    # C[k] = sum_i h[i] h[k-i].
    # d/dh[j] sum_i h[i] h[k-i] = h[k-j] + h[j] (if k-j is in range and k-j != j? No)
    # Term 1: i=j => h[j] h[k-j]. Deriv w.r.t h[j] is h[k-j].
    # Term 2: k-i=j => i=k-j. Term is h[k-j] h[j]. Deriv w.r.t h[j] is h[k-j].
    # If j = k-j (i.e., k=2j), it's the same term h[j]^2. Deriv is 2*h[j].
    # So dC[k]/dh[j] = h[k-j] + h[j] if k!=2j, else 2*h[j].
    
    # We only care about the max entry.
    # Let k_max be the index of the max entry in conv.
    # grad_M = dC[k_max]/dh
    
    # We do sub-gradient descent on the smooth approximation or just the max.
    # To be robust, we identify the top few max entries and average their gradients.
    
    lr = 0.001
    momentum = 0.9
    v = np.zeros(N)
    
    while time.time() < deadline - 5.0:
        conv = np.convolve(h, h)
        s = np.sum(h)
        if abs(s) < 1e-6:
            break
        
        # Find indices of max values
        max_val = np.max(conv)
        # Get indices where conv is close to max
        threshold = max_val - 1e-6
        max_indices = np.where(conv >= threshold)[0]
        
        if len(max_indices) == 0:
            break
            
        # Compute gradient contribution from these max indices
        grad_max = np.zeros(N)
        for k in max_indices:
            for j in range(N):
                if k - j < 0 or k - j >= N:
                    continue
                # dC[k]/dh[j]
                if k == 2*j:
                    grad_max[j] += 2 * h[j]
                else:
                    grad_max[j] += h[k-j] + h[j]
                    
        # Normalize gradient by number of max indices?
        grad_max /= len(max_indices)
        
        # Total gradient of f = max(conv)/s^2
        # df/dh = (1/s^2) * grad_max - (2*max_val/s^3) * 1
        grad_f = (1.0 / (s*s)) * grad_max - (2.0 * max_val / (s**3))
        
        # Update
        v = momentum * v - lr * grad_f
        h_new = h + v
        
        # Ensure sum is not zero and finite
        s_new = np.sum(h_new)
        if abs(s_new) < 1e-6 or not np.all(np.isfinite(h_new)):
            # Rollback and reduce learning rate
            lr *= 0.5
            continue
        
        # Check if value improved
        val_new = compute_value(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new
        else:
            # Reduce learning rate if no improvement
            lr *= 0.9
            if lr < 1e-8:
                break

    # Final check
    final_val = compute_value(best_h)
    if final_val < best_val:
        best_val = final_val
        
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-6:
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    val = compute_value(h)
    print(f"Result value: {val}")
