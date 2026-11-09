# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio optimization on 10x10 grid.
    Maximizes H(x-y) / max(H(x), H(x+y), H(x+2y)).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill

    # Precompute projection indices
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    idx_minus1 = (X - Y) + 9  # 0..18
    idx_0 = X                  # 0..9
    idx_1 = X + Y              # 0..18
    idx_2 = X + 2 * Y          # 0..27
    
    def compute_entropy_fast(idx_array, P, num_vals):
        prob = np.zeros(num_vals)
        np.add.at(prob, idx_array.ravel(), P.ravel())
        prob = prob[prob > 1e-15]
        if len(prob) == 0:
            return 0.0
        return -np.sum(prob * np.log2(prob))
    
    def compute_ratio(P):
        D = compute_entropy_fast(idx_minus1, P, 19)
        H0 = compute_entropy_fast(idx_0, P, 10)
        H1 = compute_entropy_fast(idx_1, P, 19)
        H2 = compute_entropy_fast(idx_2, P, 28)
        M = max(H0, H1, H2)
        if M < 1e-9:
            return 0.0
        return D / M
    
    def compute_ratio_from_logw(log_w):
        log_w = log_w.reshape((n, n))
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        P = w / np.sum(w)
        return compute_ratio(P), P
    
    def objective(params):
        ratio, _ = compute_ratio_from_logw(params)
        if ratio == 0.0:
            return 1e9
        return -ratio
    
    # Track best
    best_params = np.zeros(n * n)
    best_score = 0.0
    best_P = np.ones((n, n)) / (n * n)
    
    def update_best(params):
        nonlocal best_params, best_score, best_P
        ratio, P = compute_ratio_from_logw(params)
        if ratio > best_score:
            best_score = ratio
            best_params = params.copy()
            best_P = P.copy()

    def refine_solution(init_params, max_iter=2000):
        try:
            result = minimize(objective, init_params, method='Nelder-Mead',
                             options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8,
                                     'adaptive': True})
            update_best(result.x)
            return result.x
        except Exception:
            return init_params

    # --- Strategy 1: Greedy support construction with better initialization ---
    def greedy_support(max_cells=100, time_limit=15.0):
        start = time.time()
        current_mask = np.zeros((n, n), dtype=bool)
        
        for step in range(min(max_cells, n*n)):
            if time.time() - start > time_limit:
                break
            best_gain = -1e9
            best_cell = None
            
            for i in range(n):
                for j in range(n):
                    if current_mask[i, j]:
                        continue
                    
                    test_mask = current_mask.copy()
                    test_mask[i, j] = True
                    size = np.sum(test_mask)
                    if size == 0:
                        continue
                        
                    P = np.zeros((n, n))
                    P[test_mask] = 1.0 / size
                    r = compute_ratio(P)
                    
                    if r > best_gain:
                        best_gain = r
                        best_cell = (i, j)
                        
            if best_cell is None:
                break
                
            i, j = best_cell
            current_mask[i, j] = True
            
        return current_mask

    greedy_mask = greedy_support(max_cells=100, time_limit=15.0)
    
    if np.sum(greedy_mask) > 0:
        init_greedy = np.zeros(n*n)
        init_greedy[greedy_mask.ravel()] = 0.0
        init_greedy[~greedy_mask.ravel()] = -100.0
        refine_solution(init_greedy, max_iter=1500)

    # --- Strategy 2: Structured initializations based on projection geometry ---
    np.random.seed(42)
    
    # Key insight: try to make x-y distribution as uniform as possible
    # while keeping other projections concentrated
    
    # 2a: Weight by inverse diagonal length to equalize x-y probabilities
    init_inv_diag = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = i - j
            # Diagonal length for x-y=d is 10-|d|
            diag_len = 10 - abs(d)
            init_inv_diag[i, j] = 1.0 / diag_len
    init_inv_diag -= np.mean(init_inv_diag)
    update_best(init_inv_diag.ravel())
    
    # 2b: Try patterns that concentrate x, x+y, or x+2y
    # Concentrate x: put weight on a single column
    for col in range(n):
        if time.time() > deadline - 100:
            break
        init_col = np.zeros((n, n))
        init_col[:, col] = 1.0
        init_col -= np.mean(init_col)
        refine_solution(init_col.ravel(), max_iter=500)
    
    # Concentrate x+y: put weight on a single anti-diagonal
    for s in range(2*n - 1):
        if time.time() > deadline - 100:
            break
        init_anti = np.zeros((n, n))
        for i in range(n):
            j = s - i
            if 0 <= j < n:
                init_anti[i, j] = 1.0
        if np.sum(init_anti) > 0:
            init_anti -= np.mean(init_anti)
            refine_solution(init_anti.ravel(), max_iter=500)
    
    # 2c: Random structured patterns
    for _ in range(15):
        if time.time() > deadline - 80:
            break
        # Random sparse support
        size = np.random.choice([5, 10, 15, 20, 30, 40, 60])
        indices = np.random.choice(n*n, size=size, replace=False)
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[indices] = True
        
        init_rand = np.zeros(n*n)
        init_rand[mask.ravel()] = 0.0
        init_rand[~mask.ravel()] = -100.0
        refine_solution(init_rand, max_iter=800)

    # --- Strategy 3: Multi-start Nelder-Mead from diverse points ---
    initializations = [
        np.zeros(n * n),  # uniform
        init_inv_diag.ravel(),
    ]
    
    # Add some structured patterns
    for pattern_name, init in [
        ('diag', np.eye(n).ravel()),
        ('anti_diag', np.fliplr(np.eye(n)).ravel()),
        ('center', np.exp(-((np.arange(n)[:,None]-4.5)**2 + (np.arange(n)[None,:]-4.5)**2)/8).ravel()),
        ('corners', np.array([[1,0,0,0,0,0,0,0,0,1],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,0,0,0,0,0,0],
                              [1,0,0,0,0,0,0,0,0,1]]).ravel()),
    ]:
        initializations.append(init - np.mean(init))
    
    # Random initializations
    for scale in [0.5, 1.0, 2.0, 3.0, 5.0]:
        initializations.append(np.random.randn(n * n) * scale)
    
    for init in initializations:
        if time.time() > deadline - 50:
            break
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True})
            update_best(result.x)
        except Exception:
            pass
    
    # --- Strategy 4: Polish with high-precision optimization ---
    if time.time() < deadline - 20 and best_score > 0:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12,
                                     'adaptive': True})
            update_best(result.x)
        except Exception:
            pass
        
        if time.time() < deadline - 10:
            try:
                result = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-14,
                                         'adaptive': True})
                update_best(result.x)
            except Exception:
                pass
    
    # Ensure valid output
    best_P = np.clip(best_P, 0, None)
    total = np.sum(best_P)
    if total <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P = best_P / total
    
    return best_P


# EVOLVE-BLOCK-END
