import numpy as np
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n
    
    def create_area_func(resolution):
        y = (np.arange(resolution) + 0.5) / resolution
        w = (1 - y) / n
        def area(x):
            # x is shape (n,)
            L = x[:, None] + a[:, None] * y[None, :]
            L_sorted = np.sort(L, axis=0)
            diffs = np.diff(L_sorted, axis=0)
            union_len = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
            return np.mean(union_len)
        return area

    area_coarse = create_area_func(2000)
    area_fine = create_area_func(20000)

    # Initial guess: Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(x_keich, dtype=np.float64)
    
    best_x = x0
    best_area_coarse = area_coarse(x0)
    
    np.random.seed(42)
    
    # Optimize from Keich on coarse grid
    res = minimize(area_coarse, x0, method='Powell', options={'maxiter': 500, 'disp': False})
    if res.fun < best_area_coarse:
        best_area_coarse = res.fun
        best_x = res.x
        
    # Optimize from random starts on coarse grid
    for _ in range(10):
        x_rand = np.random.uniform(0, 1, n)
        res = minimize(area_coarse, x_rand, method='Powell', options={'maxiter': 300, 'disp': False})
        if res.fun < best_area_coarse:
            best_area_coarse = res.fun
            best_x = res.x

    # Refine the best coarse solution on the fine grid
    res_fine = minimize(area_fine, best_x, method='Powell', options={'maxiter': 1000, 'disp': False})
    
    # Check if fine refinement improved things
    final_x = res_fine.x if res_fine.fun < area_fine(best_x) else best_x
    
    return final_x
