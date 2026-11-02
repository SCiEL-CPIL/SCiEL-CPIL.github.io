import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    def get_max_mult(h):
        """Find the shift k that maximizes the convolution peak."""
        conv = np.convolve(h, h, mode='full')
        # conv has length 2*N-1, index i corresponds to shift (i - (N-1))
        # The max is at index argmax(conv)
        return np.argmax(conv) - (N - 1)

    def greedy_build(k_size):
        """Greedily build a set of k_size indices to minimize max autoconvolution."""
        S = []
        for _ in range(k_size):
            best_idx = None
            best_max_conv = np.inf
            for idx in range(N):
                if idx in S:
                    continue
                S_trial = S + [idx]
                h_trial = np.zeros(N)
                h_trial[S_trial] = 1.0
                conv = np.convolve(h_trial, h_trial, mode='full')
                mc = np.max(conv)
                if mc < best_max_conv:
                    best_max_conv = mc
                    best_idx = idx
            if best_idx is not None:
                S.append(best_idx)
        return S

    def neg_log_ratio(h):
        """Negative log of the ratio for optimization."""
        h = np.maximum(h, 1e-15)
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-15:
            return 0.0
        return -np.log(l2_sq / (l1 * max_conv))

    def jac_neg_log_ratio(h):
        """Jacobian of negative log ratio."""
        h = np.maximum(h, 1e-15)
        l1 = np.sum(h)
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        k_max = np.argmax(conv)
        
        # d/dh_i of l2_sq = 2*h_i
        # d/dh_i of l1 = 1
        # d/dh_i of max_conv: only the peak matters
        # conv[k_max] = sum_j h_j * h[k_max - j + (N-1)]
        # Let shift = k_max - (N-1)
        # conv[k_max] = sum_j h[j] * h[j + shift]
        # d/dh_i conv[k_max] = h[i+shift] + h[i-shift] (with bounds)
        
        shift = k_max - (N - 1)
        d_conv = np.zeros(N)
        for i in range(N):
            # h[i] * h[i+shift] if i+shift in range
            if 0 <= i + shift < N:
                d_conv[i] += h[i + shift]
            # h[i-shift] * h[i] if i-shift in range
            if 0 <= i - shift < N:
                d_conv[i] += h[i - shift]
        
        # ratio = l2_sq / (l1 * max_conv)
        # log(ratio) = log(l2_sq) - log(l1) - log(max_conv)
        # d/dh_i log(ratio) = (2*h_i)/l2_sq - 1/l1 - d_conv[i]/max_conv
        
        grad = (2 * h) / l2_sq - 1.0 / l1 - d_conv / max_conv
        return -grad

    # Initial guess: Sidon set
    S_seed = [0, 1, 4, 10, 18, 23, 30]
    best_h = np.zeros(N)
    for idx in S_seed:
        best_h[idx] = 1.0 / len(S_seed)
    
    best_score = score(best_h)

    # Randomized search over sparse sets with continuous amplitude optimization
    np.random.seed(42)
    
    for iteration in range(20000):
        if time.time() > deadline - 5.0:
            break
        
        # Random sparse set
        k = np.random.randint(2, 12)  # Size 2 to 11
        S = np.random.choice(N, k, replace=False)
        h_trial = np.zeros(N)
        h_trial[S] = 1.0 / k
        
        # Continuous optimization: maximize log(ratio)
        # Use scipy minimize on negative log ratio
        try:
            result = minimize(
                neg_log_ratio,
                h_trial,
                jac=jac_neg_log_ratio,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = np.maximum(result.x, 0)
            # Normalize to avoid degenerate solutions
            if np.sum(h_opt) > 1e-10:
                h_opt = h_opt / np.sum(h_opt)
            s = score(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except:
            pass

    # Also try greedy construction for various sizes
    for k_size in range(3, 15):
        if time.time() > deadline - 5.0:
            break
        S_greedy = greedy_build(k_size)
        h_greedy = np.zeros(N)
        h_greedy[S_greedy] = 1.0 / len(S_greedy)
        
        # Optimize amplitudes
        try:
            result = minimize(
                neg_log_ratio,
                h_greedy,
                jac=jac_neg_log_ratio,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = np.maximum(result.x, 0)
            if np.sum(h_opt) > 1e-10:
                h_opt = h_opt / np.sum(h_opt)
            s = score(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
        except:
            pass

    # Final polish: coordinate ascent
    for _ in range(500):
        if time.time() > deadline - 2.0:
            break
        improved = False
        for i in range(N):
            if best_h[i] < 1e-15:
                continue
            # Try adjusting h[i] slightly
            for delta in [0.01, -0.01, 0.005, -0.005]:
                h_trial = best_h.copy()
                h_trial[i] = max(0, h_trial[i] + delta)
                s = score(h_trial)
                if s > best_score:
                    best_score = s
                    best_h = h_trial.copy()
                    improved = True
                    break
        if not improved:
            break

    return list(best_h)
