import numpy as np
import time
from scipy.optimize import minimize

def _union_area_and_grad(x, M=4000):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    
    # Sort along axis 1
    L_sorted = np.sort(L, axis=1)
    
    # Width w(y) = (1 - y) / 64
    w = (1.0 - y) / n  # shape (M,)
    
    # Diffs: shape (M, n-1)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # min(diffs, w[:, None])
    mins = np.minimum(diffs, w[:, np.newaxis])
    
    # Union lengths: shape (M,)
    union_lengths = w + np.sum(mins, axis=1)
    
    # Area
    area = np.mean(union_lengths)
    
    # Gradient computation
    # For each y_m, the union length is w + sum_j min(L_{j+1} - L_j, w)
    # where L is sorted. The gradient w.r.t. x_i is the derivative of this.
    # 
    # Let's compute the gradient using the chain rule.
    # For a fixed y, let the sorted order be a permutation pi.
    # L_{pi(0)} <= L_{pi(1)} <= ... <= L_{pi(n-1)}
    # The union length is w + sum_{j=0}^{n-2} min(L_{pi(j+1)} - L_{pi(j)}, w)
    #
    # d(union)/dx_i = sum over j where pi(j) or pi(j+1) = i of the derivative.
    #
    # For min(d, w):
    # if d < w: derivative is 1
    # if d > w: derivative is 0
    # if d == w: subgradient in [0, 1]
    #
    # So for each pair (j, j+1) in the sorted order:
    # d = L_{pi(j+1)} - L_{pi(j)}
    # if d < w: 
    #   d(L_{pi(j+1)})/dx_{pi(j+1)} = a_{pi(j+1)}
    #   d(L_{pi(j)})/dx_{pi(j)} = a_{pi(j)}
    #   contribution to grad[pi(j+1)] += a_{pi(j+1)}
    #   contribution to grad[pi(j)] -= a_{pi(j)}
    # if d >= w:
    #   no contribution (derivative of min is 0 when d > w)
    #
    # Wait, this is the derivative of the sum of mins.
    # Let's be more careful.
    # union(y) = w(y) + sum_{j=0}^{n-2} min(L_{pi(j+1)}(y) - L_{pi(j)}(y), w(y))
    #
    # Let f_j(y) = min(L_{pi(j+1)} - L_{pi(j)}, w)
    # df_j/dx_i = min(d, w)' * (dL_{pi(j+1)}/dx_i - dL_{pi(j)}/dx_i)
    #            = indicator(d < w) * (a_{pi(j+1)} * delta_{i, pi(j+1)} - a_{pi(j)} * delta_{i, pi(j)})
    #
    # So for each j where diffs[j] < w:
    #   grad[pi(j+1)] += a[pi(j+1)]
    #   grad[pi(j)] -= a[pi(j)]
    #
    # We need to track the permutation. Let's use argsort.
    
    # Get the permutation that sorts L
    perm = np.argsort(L, axis=1)  # shape (M, n), perm[m, k] is the index of the k-th smallest L at height m
    
    # Initialize gradient
    grad = np.zeros((M, n))
    
    # For each m, for each j where diffs[m, j] < w[m]:
    #   grad[m, perm[m, j+1]] += a[perm[m, j+1]]
    #   grad[m, perm[m, j]] -= a[perm[m, j]]
    
    # Vectorized approach:
    # mask = diffs < w[:, np.newaxis]  # shape (M, n-1)
    # For each (m, j) where mask[m, j] is True:
    #   idx_plus = perm[m, j+1]
    #   idx_minus = perm[m, j]
    #   grad[m, idx_plus] += a[idx_plus]
    #   grad[m, idx_minus] -= a[idx_minus]
    
    # This is tricky to vectorize directly. Let's use np.add.at or similar.
    # Alternatively, we can compute it as:
    # contrib_plus[m, i] = sum over j where perm[m, j+1] == i and mask[m, j] of a[i]
    # contrib_minus[m, i] = sum over j where perm[m, j] == i and mask[m, j] of a[i]
    # grad[m, i] = contrib_plus[m, i] - contrib_minus[m, i]
    
    # Let's use a different approach:
    # For each m, create an array of contributions.
    
    # Actually, let's just loop over m if M is not too large, or use a smarter vectorization.
    # Given M=4000 and n=64, a loop over m might be acceptable if it's fast enough.
    # But let's try to vectorize.
    
    # Create indices for the "plus" and "minus" contributions
    # For each m and j (0 <= j < n-1):
    #   if mask[m, j]:
    #     plus_idx = perm[m, j+1]
    #     minus_idx = perm[m, j]
    
    # Let's flatten the masks and indices.
    mask_flat = (diffs < w[:, np.newaxis]).flatten()  # shape (M*(n-1),)
    
    # Get the m and j indices for True entries
    m_idx, j_idx = np.where(mask_flat.reshape(M, n-1))
    
    # Get the corresponding perm values
    plus_indices = perm[m_idx, j_idx + 1]  # shape (num_true,)
    minus_indices = perm[m_idx, j_idx]  # shape (num_true,)
    
    # The contribution to grad[m_idx[k], plus_indices[k]] is a[plus_indices[k]]
    # The contribution to grad[m_idx[k], minus_indices[k]] is -a[minus_indices[k]]
    
    # Use np.add.at to accumulate
    # First, create a flat grad array
    grad_flat = np.zeros(M * n)
    
    # Plus contributions
    np.add.at(grad_flat, m_idx * n + plus_indices, a[plus_indices])
    # Minus contributions
    np.add.at(grad_flat, m_idx * n + minus_indices, -a[minus_indices])
    
    # Reshape back
    grad = grad_flat.reshape(M, n)
    
    # Average the gradient over all y_m
    grad_mean = np.mean(grad, axis=0)
    
    return area, grad_mean

def _compute_area(x):
    """
    Compute the area of the union of the 64 triangles for given positions x.
    x: np.ndarray of shape (64,)
    Returns: float area
    """
    area, _ = _union_area_and_grad(x, M=20000)
    return area

def _keich_init():
    """Generate initial positions using Keich construction."""
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x_keich

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 580.0  # 20s margin for safety
    
    # Initial guess: Keich construction
    x = _keich_init()
    best_x = x.copy()
    best_area = _compute_area(x)
    
    # Use L-BFGS-B with gradient
    # We'll use a reduced M for the gradient to speed things up
    M_grad = 4000
    
    def objective_and_grad(x_vec):
        area, grad = _union_area_and_grad(x_vec, M=M_grad)
        return area, grad
    
    # Run optimization
    try:
        res = minimize(objective_and_grad, x, jac=True, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < best_area:
            current_area = _compute_area(res.x)
            if current_area < best_area:
                best_area = current_area
                best_x = res.x.copy()
    except Exception as e:
        pass
    
    # Polish with coordinate descent using higher precision
    # Use the best_x as starting point
    x = best_x.copy()
    
    def line_search(x, i):
        # Minimize _compute_area(x) w.r.t x[i]
        from scipy.optimize import minimize_scalar
        
        # Define objective for scalar
        def obj(t):
            x_new = x.copy()
            x_new[i] = t
            return _compute_area(x_new)
        
        # Search in a reasonable window
        res = minimize_scalar(obj, bounds=(x[i] - 0.5, x[i] + 0.5), method='bounded', options={'xatol': 1e-8})
        return res.x
    
    # Coordinate descent polish
    num_passes = 0
    while time.time() < deadline - 10.0:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            new_val = line_search(x, i)
            x[i] = new_val
            improved = True
        
        if not improved:
            break
        
        current_area = _compute_area(x)
        if current_area < best_area:
            best_area = current_area
            best_x = x.copy()
        
        num_passes += 1
        
        # Stop if converged
        if num_passes > 5 and abs(best_area - current_area) < 1e-12:
            break
    
    return best_x
