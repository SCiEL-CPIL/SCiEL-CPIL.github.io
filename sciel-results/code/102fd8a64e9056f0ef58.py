# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    # Precompute constants
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    w = 1.0 / n
    # Heights y_m = (m + 0.5) / 20000
    # Using 20000 heights as per task description for accuracy
    n_heights = 20000
    y = (np.arange(n_heights) + 0.5) / n_heights
    
    def compute_area(x):
        """Compute the union area of the n triangles using quadrature."""
        # L_i(y) = x_i + a_i * y
        # x: (n,), a: (n,), y: (n_heights,)
        # L shape: (n_heights, n)
        L = x[None, :] + a[None, :] * y[:, None]
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        # Consecutive differences
        diffs = np.diff(L_sorted, axis=1)
        # Union length at each height: w + sum over consecutive pairs of min(diffs, w)
        # Note: The task description says "w + sum over consecutive pairs of min(L_{j+1} - L_j, w)"
        # This formula assumes the first triangle contributes 'w' and subsequent ones add the overlap or gap.
        # Let's verify the formula interpretation.
        # Union of intervals [L_j, L_j+w].
        # Length = (L_{max} - L_{min}) - sum(gaps)? No.
        # The formula given in the prompt: "w + sum over consecutive pairs of min(L_{j+1} - L_j, w)"
        # Let's trust the prompt's formula.
        union_lengths = w + np.sum(np.minimum(diffs, w), axis=1)
        area = np.mean(union_lengths)
        return area

    def keich_positions():
        """Classical Keich construction for n a power of 2."""
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin

    best_x = keich_positions()
    best_area = compute_area(best_x)
    
    # Optimization strategy:
    # 1. Run L-BFGS-B from the Keich start.
    # 2. Run L-BFGS-B from several random perturbations of the Keich start.
    # 3. Polish the best result with a fast local search.

    # L-BFGS-B optimization
    # We define the objective function for scipy
    def objective(x):
        return compute_area(x)

    # Run L-BFGS-B from Keich
    try:
        result = scipy.optimize.minimize(objective, best_x, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_x

    # Random restarts
    rng = np.random.default_rng(42)
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline - 30:  # Leave 30s for polish
            break
            
        # Perturb Keich
        noise = rng.normal(0, 0.01, n)
        x_start = keich_positions() + noise
        
        try:
            result = scipy.optimize.minimize(objective, x_start, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12})
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x
        except Exception:
            pass

    # Polish with local search (Hill Climbing)
    # This is fast and can fix small local minima issues from L-BFGS-B
    current_x = best_x.copy()
    current_area = best_area
    step_size = 1e-3
    no_improve_count = 0
    max_polish_iters = 5000
    
    for _ in range(max_polish_iters):
        if time.time() > deadline:
            break
            
        i = rng.integers(0, n)
        delta = rng.normal(0, step_size)
        
        x_new = current_x.copy()
        x_new[i] += delta
        
        new_area = compute_area(x_new)
        
        if new_area < current_area:
            current_x = x_new
            current_area = new_area
            no_improve_count = 0
            # Optionally shrink step size slightly on success? Or keep it.
            # Let's keep it constant but maybe reduce if stuck.
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                step_size *= 0.95
                no_improve_count = 0
                if step_size < 1e-6:
                    break

    if current_area < best_area:
        best_x = current_x
        best_area = current_area

    return best_x

# EVOLVE-BLOCK-END
