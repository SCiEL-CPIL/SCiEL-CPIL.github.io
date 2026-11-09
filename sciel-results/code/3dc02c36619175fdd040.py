# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    return area

def _union_area_grad(x, n=64, M=20000):
    """
    Compute the area and its gradient with respect to x.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    
    grad = np.zeros(n, dtype=np.float64)
    mask = d < w[np.newaxis, :]
    
    for j in range(n - 1):
        mask_j = mask[j, :]
        if np.any(mask_j):
            upper_idx = sort_idx[j + 1, mask_j]
            lower_idx = sort_idx[j, mask_j]
            np.add.at(grad, upper_idx, 1.0 / M)
            np.add.at(grad, lower_idx, -1.0 / M)
            
    return area, grad

def _optimize_for_order(order, x_init, n=64, M=20000, max_iter=50):
    """
    Given a fixed permutation 'order' of triangles (0..n-1), 
    optimize x to minimize area, assuming the order is maintained 
    or at least serves as a strong guide.
    
    We use L-BFGS-B with the standard gradient, but initialize 
    and constrain the search around configurations consistent with 'order'.
    """
    # We don't strictly enforce the order in the gradient (since it's non-smooth there),
    # but we can use the order to create a better initial guess or to detect 
    # when the order has changed and potentially restart.
    
    # For this strategy, we treat the 'order' as the target sorted order at y=0.
    # This implies x_{order[0]} < x_{order[1]} < ... < x_{order[n-1]} at y=0.
    # We can add soft constraints or just use it for initialization.
    
    # Let's just run L-BFGS-B from x_init. The 'order' is used to select 
    # which initializations to try in the main loop.
    
    try:
        result = minimize(
            lambda x: _union_area_grad(x, n, M),
            x_init,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-6, 'maxfun': max_iter*2}
        )
        return result.x, result.fun
    except Exception:
        return x_init, _union_area(x_init, n, M)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    deadline = time.time() + 580.0  # Safety margin

    # 1. Initial Solution: Keich Construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    best_x = x0.copy()
    best_area = _union_area(x0, n, M)
    
    # 2. Global Polish with L-BFGS-B
    try:
        result = minimize(
            lambda x: _union_area_grad(x, n, M),
            best_x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-8, 'maxfun': 200}
        )
        if result.fun < best_area:
            best_x = result.x.copy()
            best_area = result.fun
    except Exception:
        pass

    # 3. Discrete Topological Search:
    # The key insight is that the "order" of triangles at y=0 (i.e., the sorting of x)
    # defines a combinatorial sector. Different sectors may have different local minima.
    # We will sample different permutations of the indices and use them to create
    # distinct starting points for the local optimizer.
    
    # Instead of random permutations, we use "twists" or "swaps" on the current best order.
    # Let current_order be the argsort of best_x.
    current_order = np.argsort(best_x)
    
    # We will try swapping adjacent elements in the order, and also some random swaps.
    # For each new order, we construct an initial x that respects this order 
    # (e.g., by sorting the current best_x according to the new order, or 
    # by assigning x values in the order with small perturbations).
    
    # A more effective approach: 
    # The objective is piecewise linear. The "vertices" of the polytope of orders 
    # correspond to specific configurations. 
    # We can use a "simulated annealing" style search over the space of permutations, 
    # where the energy is the area achieved by the best local optimum in that sector.
    
    # Given time constraints, we will do a limited number of "sector hops".
    
    np.random.seed(42)
    n_hops = 50
    
    for hop in range(n_hops):
        if time.time() > deadline - 60:
            break
            
        # Generate a new candidate order by swapping a few elements in current_order
        new_order = current_order.copy()
        
        # Randomly select 2-5 swaps
        n_swaps = np.random.randint(2, 6)
        for _ in range(n_swaps):
            i1, i2 = np.random.choice(n, 2, replace=False)
            new_order[i1], new_order[i2] = new_order[i2], new_order[i1]
            
        # Construct an initial x for this new order.
        # We want x_{new_order[0]} < x_{new_order[1]} < ...
        # We can take the current best_x values and sort them according to new_order?
        # No, that just reorders the same values. 
        # We need to perturb them to break the symmetry and allow the optimizer 
        # to find a different local minimum.
        
        # Strategy: Start with best_x, but add noise proportional to the 
        # "distance" in the permutation space? 
        # Simpler: Just add Gaussian noise to best_x, but bias the noise 
        # to encourage the new order.
        
        # Let's try: x_init = best_x + noise.
        # But to specifically target the new order, we can project best_x 
        # onto the subspace defined by the new order?
        
        # Actually, a robust way to explore a new sector is to start from 
        # a "canonical" configuration for that order.
        # For order P, let x_P[i] = rank of i in P. 
        # Then optimize from there.
        
        # This might be too far from the optimum. 
        # Better: Interpolate between the canonical order and the current best.
        
        # Let's just use random restarts with L-BFGS-B, but limit the number 
        # and ensure they are well-distributed. The "topological" aspect is 
        # that we are implicitly sampling different permutation sectors 
        # by random restarts.
        
        # To make it more "topological", we can try to swap adjacent 
        # elements in the *sorted* list of x, which corresponds to 
        # moving across a boundary in the arrangement of hyperplanes.
        
        # Let's stick to a simpler, effective heuristic: 
        # Random restarts with L-BFGS-B are already doing this. 
        # The "new" strategy should be something like: 
        # Use the gradient to determine the "active set" of constraints 
        # and then solve a linear program for the active set?
        
        # Given the complexity, let's implement a "Coordinate Descent 
        # on Permutations" where we swap two triangles and check if 
        # the area decreases after a quick local polish.
        
        # For this specific implementation, we will perform random swaps 
        # on the indices of best_x, and then run a short L-BFGS-B optimization.
        
        x_start = best_x.copy()
        i1, i2 = np.random.choice(n, 2, replace=False)
        
        # Swap the x values of these two triangles
        x_start[i1], x_start[i2] = x_start[i2], x_start[i1]
        
        # Add small noise to escape the saddle point
        x_start += np.random.randn(n) * 0.01
        
        try:
            result = minimize(
                lambda x: _union_area_grad(x, n, M),
                x_start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-5, 'maxfun': 50}
            )
            
            if result.fun < best_area:
                best_x = result.x.copy()
                best_area = result.fun
                current_order = np.argsort(best_x)
                
        except Exception:
            pass

    # 4. Final Polish
    if time.time() < deadline - 30:
        try:
            result = minimize(
                lambda x: _union_area_grad(x, n, M),
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-8, 'maxfun': 100}
            )
            if result.fun < best_area:
                best_x = result.x.copy()
                best_area = result.fun
        except Exception:
            pass

    if not np.all(np.isfinite(best_x)):
        return x0
    if best_area <= 0:
        return x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
