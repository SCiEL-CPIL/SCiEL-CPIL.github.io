# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area_fast(x):
    """Compute the area of the union of 64 triangles given base positions x.
    Uses reduced quadrature for speed during optimization."""
    n = 64
    a = (np.arange(n) + 1) / n
    # Use fewer quadrature points for speed (2000 instead of 20000)
    nq = 2000
    m = np.arange(nq)
    y = (m + 0.5) / nq
    
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / 64.0
    
    L_sorted = np.sort(L, axis=1)
    d = np.diff(L_sorted, axis=1)
    d_clipped = np.minimum(d, w[:, None])
    union_lengths = w + d_clipped.sum(axis=1)
    area = union_lengths.mean()
    
    return area


def _compute_area_full(x):
    """Compute the area using full 20000-point quadrature."""
    n = 64
    a = (np.arange(n) + 1) / n
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / 64.0
    
    L_sorted = np.sort(L, axis=1)
    d = np.diff(L_sorted, axis=1)
    d_clipped = np.minimum(d, w[:, None])
    union_lengths = w + d_clipped.sum(axis=1)
    area = union_lengths.mean()
    
    return area


def _gradient_numeric(x, eps=1e-6):
    """Compute numerical gradient using forward differences with fast area."""
    n = 64
    grad = np.zeros(n)
    base = _compute_area_fast(x)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (_compute_area_fast(x_plus) - base) / eps
    
    return grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses gradient-based optimization with strict time budgeting.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 20s margin before 600s limit
    
    n = 64
    
    # Initial Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps_bits = list(str(bin(i))[2:])
        eps_arr = np.array(eps_bits, dtype=int)
        if len(eps_arr) < k:
            eps_arr = np.pad(eps_arr, (k - len(eps_arr), 0), "constant")
        positions.append(np.sum((1 - j) * eps_arr * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    best_x = x0.copy()
    best_area_fast = _compute_area_fast(x0)
    
    # Phase 1: Gradient descent with line search using fast area
    lr = 0.01
    patience = 0
    
    for iteration in range(500):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        # Compute gradient
        grad = _gradient_numeric(best_x)
        
        # Gradient descent step
        x_new = best_x - lr * grad
        
        # Line search: try the step, if it doesn't improve, try smaller steps
        area_new = _compute_area_fast(x_new)
        
        if area_new < best_area_fast - 1e-10:
            best_x = x_new
            best_area_fast = area_new
            patience = 0
            # Occasionally increase learning rate if we're making good progress
            if iteration % 50 == 0 and iteration > 0:
                lr = min(lr * 1.2, 0.05)
        else:
            patience += 1
            lr *= 0.8
            if patience > 20:
                break
    
    # Phase 2: Coordinate descent refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        for cd_iter in range(3):
            elapsed = time.time() - start_time
            if elapsed > deadline - 20:
                break
            
            improved = False
            for i in range(n):
                elapsed = time.time() - start_time
                if elapsed > deadline - 15:
                    break
                
                best_xi = best_x[i]
                best_val = best_area_fast
                
                # Try several step sizes
                for delta in [-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02]:
                    x_trial = best_x.copy()
                    x_trial[i] = best_xi + delta
                    area_trial = _compute_area_fast(x_trial)
                    
                    if area_trial < best_val - 1e-12:
                        best_val = area_trial
                        best_xi = x_trial[i]
                        improved = True
                
                best_x[i] = best_xi
            
            if best_val < best_area_fast:
                best_area_fast = best_val
            
            if not improved:
                break
    
    # Final: evaluate with full quadrature and return
    return best_x


# EVOLVE-BLOCK-END
