import numpy as np
import scipy.optimize as so
import time

def _keich_initial(n=64):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def _area_and_grad(x, n=64, M=20000):
    a = (np.arange(1, n + 1)) / n
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    L = x[None, :] + (a[None, :] * y[:, None])
    
    sorted_L = np.sort(L, axis=1)
    gaps = sorted_L[:, 1:] - sorted_L[:, :-1]
    
    mins = np.minimum(gaps, w[:, None])
    U = w + np.sum(mins, axis=1)
    area = np.mean(U)
    
    sort_idx = np.argsort(L, axis=1)
    left_indices = sort_idx[:, :-1]
    right_indices = sort_idx[:, 1:]
    
    mask = (gaps < w[:, None]).astype(np.float64)
    
    grad = np.zeros(n, dtype=np.float64)
    
    left_flat = left_indices.ravel()
    right_flat = right_indices.ravel()
    mask_flat = mask.ravel()
    
    np.add.at(grad, left_flat, -mask_flat)
    np.add.at(grad, right_flat, mask_flat)
    
    grad /= M
    
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x0 = _keich_initial(n)
    
    start_time = time.time()
    deadline = start_time + 600
    
    best_x = [x0.copy()]
    
    class TimeOut(Exception):
        pass
        
    def callback(xk):
        if time.time() > deadline - 10:
            raise TimeOut()
        best_x[0] = xk.copy()
        
    last_x = None
    last_res = None
    
    def obj(x):
        nonlocal last_x, last_res
        if last_x is not None and np.array_equal(x, last_x):
            return last_res[0]
        area, g = _area_and_grad(x, n)
        last_x = x
        last_res = (area, g)
        return area
        
    def grad(x):
        nonlocal last_x, last_res
        if last_x is not None and np.array_equal(x, last_x):
            return last_res[1]
        area, g = _area_and_grad(x, n)
        last_x = x
        last_res = (area, g)
        return g
        
    try:
        result = so.minimize(obj, x0, method='L-BFGS-B', jac=grad, 
                             options={'maxiter': 10000, 'callback': callback, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-12})
        if result.success and result.x is not None:
            # Verify result is valid
            if np.all(np.isfinite(result.x)):
                best_x[0] = result.x
    except TimeOut:
        pass
    except Exception:
        pass
        
    return best_x[0].astype(np.float64)
