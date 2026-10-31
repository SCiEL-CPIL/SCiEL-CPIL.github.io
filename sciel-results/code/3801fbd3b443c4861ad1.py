import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_objective(x, beta):
    """Smooth approximation to max|autoconv| / S^2 using LSE."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return 1e10
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    lse = m + np.log(np.sum(np.exp(beta * (abs_ac - m))))
    return (lse / beta) / (S * S)

def _soft_gradient(x, beta):
    """Gradient of the soft objective."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.zeros_like(x)
    
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return np.zeros_like(x)
    
    exp_terms = np.exp(beta * (abs_ac - m))
    weights = exp_terms / np.sum(exp_terms)
    sign_ac = np.sign(ac)
    A = weights * sign_ac  # weighted signed autoconv
    
    # Gradient of LSE w.r.t. x[i]:
    # d/dx[i] sum_k w_k * ac_k = sum_k w_k * d(ac_k)/dx[i]
    # ac_k = sum_j x_j * x_{k-j} (with appropriate bounds)
    # d(ac_k)/dx[i] = x_{k-i} + x_{k-i} (symmetric) = 2*x_{k-i} for valid range
    # More carefully: ac_k = sum_j x[j]*x[k-j], so d(ac_k)/dx[i] = x[k-i] + (if i != k-i) x[k-i]
    # Actually for autoconvolution: ac_k = sum_j x[j] x[k-j]
    # d(ac_k)/dx[i] = x[k-i] (from j=i term) + x[k-i] (from k-j=i, j=k-i) 
    # But when i = k-i, i.e., k=2i, it's just x[i] (one term)
    # So d(ac_k)/dx[i] = 2*x[k-i] for k != 2i, and x[i] for k = 2i
    
    grad_lse = np.zeros(N)
    for i in range(N):
        # k ranges such that k-i is in [0, N-1]
        # k from i to i+N-1, but also k >= 0 and k <= 2N-2
        k_lo = max(0, i)
        k_hi = min(2 * N - 1, i + N - 1)
        if k_lo > k_hi:
            continue
        # For each k in [k_lo, k_hi], j = k - i is in [0, N-1]
        # d(ac_k)/dx[i] = 2 * x[k-i] except when k = 2i, then it's x[i]
        js = np.arange(k_lo - i, k_hi - i + 1)  # j values
        A_segment = A[k_lo:k_hi+1]
        x_segment = x[js]
        
        # Handle the diagonal case k=2i
        contrib = 2.0 * np.dot(A_segment, x_segment)
        if 2 * i >= k_lo and 2 * i <= k_hi:
            # Subtract the extra x[i] that was counted twice
            contrib -= A[2 * i] * x[i]
        
        grad_lse[i] = contrib
    
    lse_val = m + np.log(np.sum(np.exp(beta * (abs_ac - m))))
    G_val = lse_val / beta
    
    # d/dx[i] [G_val / S^2] = (dG/dx[i] * S^2 - G_val * 2*S) / S^4
    # = dG/dx[i] / S^2 - 2*G_val / S^3
    grad = grad_lse / (S * S) - 2.0 * G_val / (S * S * S)
    
    return grad

def _project_to_binary(x):
    """Project continuous vector to best binary via thresholding + local search."""
    # Try threshold at 0
    b = np.where(x >= 0, 1.0, -1.0)
    return b

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    rng = np.random.default_rng(42)

    # Phase 1: Generate diverse binary seeds
    candidates = [
        np.ones(N),
        np.ones(N//2).tolist() + [-1.0]*(N - N//2),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
    ]
    
    seed_list = []
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        sc = _score(c)
        seed_list.append((sc, c.copy()))
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Add random binary seeds
    for seed_idx in range(200):
        if time.time() > deadline - 80:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        seed_list.append((sc, x.copy()))
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Sort seeds by score
    seed_list.sort(key=lambda t: t[0])
    
    # Phase 2: Continuous optimization via soft-max relaxation from top seeds
    # For each of the top seeds, run continuous optimization to find better basins
    
    top_seeds = seed_list[:10]  # Use top 10 seeds
    
    for seed_score, seed_x in top_seeds:
        if time.time() > deadline - 20:
            break
            
        # Start from the binary seed as initial continuous point
        # Normalize so sum is reasonable
        S_init = np.sum(seed_x)
        if abs(S_init) < 1e-6:
            continue
        x0 = seed_x / S_init  # normalize to unit sum
        
        # Run continuous optimization with annealing beta
        current_x = x0.copy()
        betas = [3.0, 8.0, 20.0, 50.0]
        
        for beta in betas:
            if time.time() > deadline - 15:
                break
            
            # Use L-BFGS-B for continuous optimization
            def obj(x):
                return _soft_objective(x, beta)
            
            def grad(x):
                return _soft_gradient(x, beta)
            
            try:
                result = minimize(
                    obj, current_x, jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.fun < obj(current_x):
                    current_x = result.x
            except Exception:
                pass
        
        # Project to binary and evaluate
        b = np.where(current_x >= 0, 1.0, -1.0)
        b_score = _score(b)
        if b_score < best_score:
            best_score = b_score
            best_x = b.copy()
        
        # Also try negative projection
        b_neg = -b
        b_neg_score = _score(b_neg)
        if b_neg_score < best_score:
            best_score = b_neg_score
            best_x = b_neg.copy()

    # Phase 3: Fast binary local search using incremental FFT updates
    x = best_x.copy()
    cur_score = best_score

    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    X = np.fft.rfft(x, fft_size)
    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    # Random local search
    rng2 = np.random.default_rng(999)
    no_improve = 0
    max_no_improve = 300
    start_ls = time.time()
    
    while time.time() < deadline - 10 and no_improve < max_no_improve:
        i = rng2.integers(0, N)
        old_val = x[i]
        x[i] = -old_val
        X_new = X - 2.0 * old_val * exp_factors[i]
        ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
        max_ac = np.max(np.abs(ac))
        s = np.sum(x)
        if abs(s) < 1e-12:
            new_score = float('inf')
        else:
            new_score = max_ac / (s * s)

        if new_score < cur_score:
            X = X_new
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
        else:
            x[i] = old_val
            no_improve += 1

    # Phase 4: Systematic sweep - try flipping each index in order
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            old_val = x[i]
            x[i] = -old_val
            X_new = X - 2.0 * old_val * exp_factors[i]
            ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
            max_ac = np.max(np.abs(ac))
            s = np.sum(x)
            if abs(s) < 1e-12:
                new_score = float('inf')
            else:
                new_score = max_ac / (s * s)

            if new_score < cur_score:
                X = X_new
                cur_score = new_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
            else:
                x[i] = old_val

    # Phase 5: Multi-flip search
    if time.time() < deadline - 3:
        x = best_x.copy()
        cur_score = best_score
        X = np.fft.rfft(x, fft_size)
        rng3 = np.random.default_rng(1234)
        no_improve = 0
        max_no_improve = 150
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            i1 = rng3.integers(0, N)
            i2 = rng3.integers(0, N)
            if i1 == i2:
                continue
            v1, v2 = x[i1], x[i2]
            x[i1] = -v1
            x[i2] = -v2
            X_new = X - 2.0 * v1 * exp_factors[i1] - 2.0 * v2 * exp_factors[i2]
            ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
            max_ac = np.max(np.abs(ac))
            s = np.sum(x)
            if abs(s) < 1e-12:
                new_score = float('inf')
            else:
                new_score = max_ac / (s * s)
            if new_score < cur_score:
                X = X_new
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
            else:
                x[i1] = v1
                x[i2] = v2
                no_improve += 1

    # Final verification
    final_score = _score(best_x)
    if final_score < best_score:
        best_score = final_score

    return best_x
