import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
DEADLINE = time.time() + TIME_LIMIT - SAFETY_MARGIN


def _autoconv(f):
    """Compute linear autoconvolution of f (length N) -> length 2N-1."""
    return np.convolve(f, f, mode='full')


def _score(f):
    """Compute max|autoconv| / (sum f)^2."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    return np.max(np.abs(C)) / (S * S)


def _legendre_symbol(n, p):
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _generate_legendre_sequence(p, offset=1):
    seq = np.zeros(N)
    for i in range(N):
        val = _legendre_symbol((i + offset) % p, p)
        if val == 0:
            val = 1
        seq[i] = val
    return seq


def _soft_objective(f, t):
    """Soft max objective: sum(exp(|C|/t)) / (sum f)^2 * t. Lower is better for small t."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return 1e10
    C = _autoconv(f)
    # Use log-sum-exp for numerical stability
    abs_C = np.abs(C)
    max_val = np.max(abs_C)
    if max_val < 1e-12:
        return 0.0
    log_sum = max_val / t + np.log(np.sum(np.exp((abs_C - max_val) / t)))
    return t * np.exp(log_sum) / (S * S)


def _soft_gradient(f, t):
    """Gradient of soft objective w.r.t. f."""
    S = np.sum(f)
    if abs(S) < 1e-12:
        return np.zeros(N)
    C = _autoconv(f)
    abs_C = np.abs(C)
    max_val = np.max(abs_C)
    if max_val < 1e-12:
        return np.zeros(N)
    weights = np.exp((abs_C - max_val) / t)
    sum_w = np.sum(weights)
    
    # d/d f[i] of sum_j exp(|C_j|/t)
    # C_k = sum_i f[i] f[k-i]
    # dC_k / df[i] = f[k-i] + f[i-(k-N+1)] (for linear conv, careful with indices)
    # Actually for linear conv of f with f: C_k = sum_{i=0}^{k} f[i] f[k-i] for k < N
    # and C_k = sum_{i=k-N+1}^{N-1} f[i] f[k-i] for k >= N
    # dC_k/df[i] = f[k-i] if 0<=k-i<N, plus f[i-(k-N+1)] if applicable... 
    # Simpler: use the fact that d/d f[i] (C_k) = f[k-i] + f[i] * (contribution from other side)
    # For autocorrelation: C_k = sum_i f[i] f[i+k] (with appropriate bounds)
    # dC_k/df[i] = f[i+k] + f[i-k] (where valid)
    
    # Let's compute gradient more carefully
    # Objective = t * exp(log_sum) / S^2
    # d(obj)/df[i] = t/S^2 * exp(log_sum) * (1/t) * sum_k weights_k * sign(C_k) * dC_k/df[i]
    #             - 2*t*exp(log_sum)/S^3 * S * ... (derivative of 1/S^2)
    #             = t/S^2 * exp(log_sum) * (1/t) * sum_k w_k * sign(C_k) * dC_k/df[i]
    #             - 2*exp(log_sum)/S * sum_j f[j] * ... 
    
    # This is getting complex. Let's use a simpler approach:
    # grad_i = (t/S^2) * exp(log_sum) * (1/t) * sum_k w_k * sign(C_k) * dC_k/df[i]
    #        - 2*(t/S^2)*exp(log_sum)/S * sum_k w_k * sign(C_k) * dC_k/df[i] * (S/2) ... 
    # Actually: d(1/S^2)/df[i] = -2/S^3
    # So: d(obj)/df[i] = t*exp(log_sum) * [ (1/t) * (1/S^2) * sum_k w_k sign(C_k) dC_k/df[i] 
    #                                    - 2/S^3 * sum_k w_k sign(C_k) dC_k/df[i] * ... ]
    # Wait, let me redo: obj = t * A / S^2 where A = sum_k exp(|C_k|/t)
    # d(obj)/df[i] = t * (dA/df[i] * S^2 - A * 2*S * df[i]/S... no
    # d(obj)/df[i] = t * (dA/df[i] / S^2 - A * 2 * (dS/df[i]) / S^3)
    # dA/df[i] = sum_k exp(|C_k|/t) * (1/t) * sign(C_k) * dC_k/df[i]
    # dS/df[i] = 1
    
    # dC_k/df[i]: C_k = sum_j f[j] f[k-j]
    # dC_k/df[i] = f[k-i] (if 0<=k-i<N) + f[i] * (1 if k-i... no)
    # Actually C_k = sum_j f[j] f[k-j]. d/d f[i]: 
    #   term where j=i: f[k-i] (if 0<=k-i<N)
    #   term where k-j=i, i.e., j=k-i: f[i] (if 0<=k-i<N)
    # So dC_k/df[i] = f[k-i] + f[i] if 0<=k-i<N, but we need to be careful about double counting
    # when k-i = i, i.e., k=2i. In that case it's just f[i] (not 2*f[i]).
    # More precisely: dC_k/df[i] = f[k-i] (if valid) + f[k-i] (if valid and k-i != i)
    # = f[k-i] if k-i==i, else 2*f[k-i] if 0<=k-i<N, else 0... no that's wrong.
    
    # Let me just use a numerical or direct approach. For N=400, we can compute the gradient
    # by noting that the autocorrelation structure means:
    # dC_k/df[i] = f[k-i] + f[i-(k-N+1)] for the linear convolution case... 
    # This is getting too complex analytically. Let's use a different strategy.
    
    # Alternative: use the fact that we want to minimize max|C|/S^2.
    # Use a simpler gradient: just the weighted sum approach.
    
    # For simplicity and correctness, let's compute the gradient numerically or use
    # a subgradient of the max.
    
    # Subgradient approach: find the index k* where |C| is maximized.
    k_star = np.argmax(abs_C)
    sign_c = np.sign(C[k_star])
    if sign_c == 0:
        sign_c = 1.0
    
    # dC_k*/df[i] = f[k*-i] if 0<=k*-i<N, plus f[i] if k*-i... 
    # For linear autocorrelation C_k = sum_{j=max(0,k-N+1)}^{min(N-1,k)} f[j] f[k-j]
    # dC_k/df[i] = f[k-i] (if 0<=k-i<N) + f[k-i] (if 0<=k-i<N and k-i != i)
    # Actually: C_k = sum_j f[j]*f[k-j]. Derivative w.r.t. f[i]:
    #   from j=i: f[k-i] (if 0<=k-i<N)
    #   from k-j=i => j=k-i: f[i] (if 0<=k-i<N)
    # So if 0<=k-i<N: dC_k/df[i] = f[k-i] + f[i] if k-i != i, else f[i]
    # Wait no: if j=i, the term is f[i]*f[k-i], derivative is f[k-i].
    # If j=k-i (and k-i != i), the term is f[k-i]*f[i], derivative is f[i].
    # If k-i == i (i.e., k=2i), both terms are the same: f[i]*f[i], derivative is 2*f[i].
    # So: dC_k/df[i] = f[k-i] + f[i] if 0<=k-i<N and k-i != i
    #                    2*f[i] if 0<=k-i<N and k-i == i (i.e., k=2i)
    #                    0 otherwise
    
    # Let's compute this for k_star
    grad = np.zeros(N)
    for i in range(N):
        k_minus_i = k_star - i
        if 0 <= k_minus_i < N:
            if k_minus_i == i:
                grad[i] = 2.0 * f[i]
            else:
                grad[i] = f[k_minus_i] + f[i]
    
    # Subgradient of objective
    # obj = max|C| / S^2
    # d(obj)/df[i] = sign(C[k*]) * dC_k*/df[i] / S^2 - 2*max|C|/S^3
    # (subgradient)
    max_abs = abs_C[k_star]
    grad = sign_c * grad / (S * S) - 2.0 * max_abs / (S * S * S)
    
    return grad


def _local_search_binary_pair(f, max_iter=5000, time_budget=30.0):
    """
    Local search with pair-flip moves: flip f[i] and f[j] simultaneously.
    This escapes local minima that single-flip gets stuck in.
    """
    f = f.copy()
    f = np.sign(f)
    
    S = np.sum(f)
    if abs(S) < 1:
        for _ in range(N):
            idx = np.random.randint(N)
            if f[idx] * S < 0 or S == 0:
                f[idx] = -f[idx]
                S = np.sum(f)
                if abs(S) >= 1:
                    break
        if abs(S) < 1:
            return f, 1e10
    
    C = _autoconv(f)
    score = np.max(np.abs(C)) / (S * S)
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 80
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        best_improvement = 0.0
        best_pair = (-1, -1)
        
        # Try single flips first (fast)
        for i in range(N):
            v = f[i]
            new_S = S - 2.0 * v
            if abs(new_S) < 1e-6:
                continue
            
            k_start = i
            k_end = min(i + N, 2 * N - 1)
            delta = -2.0 * v * f[0:k_end - k_start]
            
            new_C = C.copy()
            new_C[k_start:k_end] += delta
            new_max = np.max(np.abs(new_C))
            new_score = new_max / (new_S * new_S)
            
            improvement = score - new_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_pair = (i, -1)
        
        # Try pair flips (sampled for efficiency)
        num_pair_trials = min(50, N // 4)
        for _ in range(num_pair_trials):
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            v_i, v_j = f[i], f[j]
            new_S = S - 2.0 * v_i - 2.0 * v_j
            if abs(new_S) < 1e-6:
                continue
            
            # Compute new C
            new_C = C.copy()
            # Effect of flipping i
            k_start_i = i
            k_end_i = min(i + N, 2 * N - 1)
            new_C[k_start_i:k_end_i] += -2.0 * v_i * f[0:k_end_i - k_start_i]
            # Effect of flipping j (on the already-updated C, but we need to use original f for the other term)
            # Actually, flipping both: C_new[k] = C[k] - 2*v_i*f[k-i] - 2*v_j*f[k-j]
            # But when k-i == j or k-j == i, there's a cross term... 
            # For simplicity, just recompute
            f_new = f.copy()
            f_new[i] = -v_i
            f_new[j] = -v_j
            new_C = _autoconv(f_new)
            new_max = np.max(np.abs(new_C))
            new_score = new_max / (new_S * new_S)
            
            improvement = score - new_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_pair = (i, j)
        
        if best_pair[0] >= 0:
            i, j = best_pair
            if j == -1:
                v = f[i]
                f[i] = -v
                S -= 2.0 * v
            else:
                v_i, v_j = f[i], f[j]
                f[i] = -v_i
                f[j] = -v_j
                S -= 2.0 * v_i + 2.0 * v_j
            C = _autoconv(f)
            score = np.max(np.abs(C)) / (S * S)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
    
    return f, score


def _sparse_block_init(rng, density=0.3):
    """Create a sparse block pattern initialization."""
    f = np.ones(N)
    n_blocks = rng.integers(3, 10)
    for _ in range(n_blocks):
        start = rng.integers(0, N)
        length = rng.integers(2, max(3, N // n_blocks))
        end = min(start + length, N)
        f[start:end] = -1.0
    return f


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Legendre sequences
    primes = [397, 401, 389, 383, 379, 373, 367, 359, 353, 349]
    for p in primes:
        for offset in [0, 1, 2, 3]:
            if time.time() > DEADLINE:
                break
            seq = _generate_legendre_sequence(p, offset)
            S = np.sum(seq)
            if abs(S) >= 1:
                initializations.append(seq.copy())
        if time.time() > DEADLINE:
            break
    
    # Random binary
    for _ in range(5):
        if time.time() > DEADLINE:
            break
        seq = rng.choice([-1.0, 1.0], size=N)
        if abs(np.sum(seq)) >= 1:
            initializations.append(seq)
    
    # Sparse block patterns
    for _ in range(10):
        if time.time() > DEADLINE:
            break
        seq = _sparse_block_init(rng)
        if abs(np.sum(seq)) >= 1:
            initializations.append(seq)
    
    # If no valid initializations, use all ones
    if not initializations:
        initializations = [np.ones(N)]
    
    best_f = initializations[0].copy()
    best_score = _score(best_f)
    
    # Run local search from each initialization
    for f0 in initializations:
        if time.time() > DEADLINE:
            break
        
        remaining_time = DEADLINE - time.time()
        if remaining_time < 5.0:
            break
        
        time_budget = min(25.0, remaining_time * 0.25)
        
        f_local, score_local = _local_search_binary_pair(f0, max_iter=5000, time_budget=time_budget)
        
        if score_local < best_score:
            best_score = score_local
            best_f = f_local.copy()
    
    # Final polish
    if time.time() < DEADLINE - 2.0:
        remaining = DEADLINE - time.time() - 1.0
        f_final, score_final = _local_search_binary_pair(best_f, max_iter=10000, time_budget=remaining)
        if score_final < best_score:
            best_score = score_final
            best_f = f_final.copy()
    
    return best_f
