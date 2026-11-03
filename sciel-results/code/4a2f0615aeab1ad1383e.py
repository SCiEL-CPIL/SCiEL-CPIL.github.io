import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    def _objective(z_flat):
        """
        Minimize -log(ratio).
        h = z^2.
        """
        if time.time() > deadline - 3.0:
            return 1e10
        z = z_flat
        h = z**2
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 1e10
        ratio = l2_sq / (l1 * max_conv)
        if ratio < 1e-10:
            return 1e10
        # Barrier to keep h > 0
        barrier = -1e-6 * np.sum(np.log(h + 1e-12))
        return -np.log(ratio) + barrier

    def _gradient(z_flat):
        """Numerical gradient."""
        eps = 1e-7
        grad = np.zeros_like(z_flat)
        f0 = _objective(z_flat)
        for i in range(N):
            z_plus = z_flat.copy()
            z_plus[i] += eps
            z_minus = z_flat.copy()
            z_minus[i] -= eps
            grad[i] = (_objective(z_plus) - _objective(z_minus)) / (2 * eps)
        return grad

    def optimize_from_z(z0, max_iter=100):
        if time.time() > deadline - 5.0:
            return np.zeros(N), 0.0
        
        z0 = np.real(z0)
        h0 = z0**2
        s0 = score(h0)
        
        best_h = h0.copy()
        best_s = s0
        
        try:
            res = minimize(
                _objective,
                z0,
                jac=_gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if res.fun < 1e9:
                z_opt = res.x
                h_opt = np.maximum(z_opt**2, 0)
                s_opt = score(h_opt)
                if s_opt > best_s:
                    best_h = h_opt
                    best_s = s_opt
        except Exception:
            pass
        
        return best_h, best_s

    def find_sidon_set(k, seed=0):
        """Find a k-element set with minimal max autocorrelation."""
        rng = np.random.RandomState(seed)
        best_set = None
        best_max_conv = np.inf
        
        for attempt in range(20):
            S = sorted(rng.choice(N, k, replace=False).tolist())
            improved = True
            while improved:
                if time.time() > deadline - 5.0:
                    return best_set if best_set is not None else S
                improved = False
                for i in range(k):
                    for j in range(N):
                        if j in S:
                            continue
                        S_new = sorted(S[:i] + [j] + S[i+1:])
                        h_tmp = np.zeros(N)
                        h_tmp[S_new] = 1.0
                        conv = np.convolve(h_tmp, h_tmp, mode='full')
                        mc = np.max(conv)
                        if mc < best_max_conv:
                            best_max_conv = mc
                            best_set = S_new
                            S = S_new
                            improved = True
                            break
                    if improved:
                        break
        
        if best_set is None:
            best_set = sorted(rng.choice(N, k, replace=False).tolist())
        
        return best_set

    def optimize_amplitudes(S, max_iter=50):
        """Optimize amplitudes for a fixed support set S."""
        k = len(S)
        if k == 0:
            return np.zeros(N), 0.0
        
        z = np.zeros(N)
        z[S] = 1.0 / np.sqrt(k)
        
        h = z**2
        h = h / np.sum(h)
        z = np.sqrt(h)
        
        best_h, best_s = optimize_from_z(z, max_iter)
        return best_h, best_s

    # Main search loop
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try various support sizes
    for k in range(3, 15):
        if time.time() > deadline - 10.0:
            break
        for seed in range(5):
            if time.time() > deadline - 5.0:
                break
            S = find_sidon_set(k, seed=seed)
            h_opt, s = optimize_amplitudes(S)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    
    # Random restarts
    rng = np.random.RandomState(42)
    for iter in range(100):
        if time.time() > deadline - 5.0:
            break
        k = rng.randint(3, 12)
        S = sorted(rng.choice(N, k, replace=False).tolist())
        z = np.zeros(N)
        z[S] = rng.randn(k)
        h_opt, s = optimize_from_z(z, max_iter=50)
        if s > best_score:
            best_score = s
            best_h = h_opt.copy()

    return list(best_h)
