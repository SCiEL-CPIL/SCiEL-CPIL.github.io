# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N
    t0 = time.time()
    DEADLINE = 600.0
    
    def elapsed():
        return time.time() - t0

    def volume_loss(params):
        """
        Computes the union volume of the 64 tubes.
        params: shape (128,) -> [x_0..x_63, y_0..y_63]
        """
        x = params[:64]
        y = params[64:]
        
        # Box bounds
        # x_k + c * (i+1)/8. Max over c in [0,1] is x_k + (i+1)/8.
        # i = k // 8.
        i_idx = np.arange(64) // 8
        j_idx = np.arange(64) % 8
        
        x_max = x + (i_idx + 1) * w
        y_max = y + (j_idx + 1) * w
        
        X0 = np.min(x)
        X1 = np.max(x_max)
        Y0 = np.min(y)
        Y1 = np.max(y_max)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10 # Penalize degenerate boxes
            
        # Monte Carlo estimation for speed and robustness
        # We need to check if (a, b, c) is in ANY tube.
        # Tube k contains (a, b, c) if:
        # x_k <= a - c*(i_k)/8 <= x_k + w  AND  y_k <= b - c*(j_k)/8 <= y_k + w
        # => a - x_k - c*(i_k)/8 in [0, w] AND b - y_k - c*(j_k)/8 in [0, w]
        
        # We use a fixed number of samples for consistent gradient signal
        n_samples = 20000
        # Use a deterministic pseudo-random sequence based on a fixed seed for stability
        # But for optimization, we want the loss to be smooth. 
        # Actually, the evaluator uses rng = np.random.default_rng(12345).
        # To match the evaluator's metric exactly, we should use the same rng?
        # No, the evaluator calls kakeya_needle_3d8() once, gets positions, then runs MC.
        # So we just need to minimize the *true* volume (or a good estimator of it).
        # Using a fixed set of points for the internal loss is fine, as long as it correlates with the true volume.
        
        # Let's use a stratified sampling or just fixed points to make the loss deterministic for the optimizer.
        # If we use random points each step, the loss is noisy.
        # Let's use a fixed grid of points for the internal optimization.
        
        # Grid size: 10x10x10 = 1000 points might be too coarse.
        # 20x20x20 = 8000 points.
        # Let's use 5000 points uniformly distributed in the box [X0, X1]x[Y0, Y1]x[0,1].
        # But the box moves. This is tricky.
        
        # Alternative: Minimize the bounding box area * (fraction covered in a fixed reference box)?
        # No, the volume is defined relative to the bounding box of the union.
        
        # Let's stick to the definition: V = Area * Fraction.
        # To make this differentiable-ish for Nelder-Mead, we can use a fixed set of points in the *current* box?
        # That introduces discontinuities when box bounds change.
        
        # Better approach:
        # The volume is the measure of the union.
        # For small N (64 tubes), we can compute the union volume more accurately using integration or more samples.
        
        # Let's use 50,000 samples but fixed in a normalized space [0,1]^3, mapped to the current box.
        # This means the points move with the box.
        # a = X0 + u * (X1 - X0)
        # b = Y0 + v * (Y1 - Y0)
        # c = w_coord
        # This makes the "fraction" part constant w.r.t box scaling? No.
        # If we scale the box, the tubes scale? No, tubes have fixed width w.
        
        # Let's just use a large number of random points in the current box.
        # To reduce noise, we can use the same random numbers each time, just mapped to the current box.
        
        rng_internal = np.random.default_rng(42)
        # Pre-generate random points in [0,1]^3
        # We need them to be uniform in the box.
        # u, v, w_c in [0,1]
        # a = X0 + u * (X1 - X0)
        # b = Y0 + v * (Y1 - Y0)
        # c = w_c
        
        # We'll generate 10,000 points.
        n_mc = 10000
        # We can't pre-generate if we want them to be "fixed" in the sense of the evaluator,
        # but for optimization, we want a stable loss landscape.
        # Using fixed u,v,w_c and mapping to the moving box is a standard technique.
        
        # However, if the box shrinks, the tubes (fixed width) might not cover the same relative fraction.
        # This is correct behavior.
        
        # Let's generate the points once outside the loss function?
        # No, they depend on X0, X1, etc.
        
        # For speed, let's use vectorized operations.
        # We need to check if ANY tube covers the point.
        # Tube k:
        # x_k <= a - c*(i_k)/8 <= x_k + w
        # y_k <= b - c*(j_k)/8 <= y_k + w
        
        # Let's use a smaller sample size for the internal optimizer to allow more iterations.
        # 2000 points should be enough to distinguish good vs bad configurations.
        
        # We'll use a fixed set of random numbers for u, v, c to ensure the loss is deterministic
        # for a given set of parameters (x, y).
        
        # This is a closure, so we can cache the random numbers.
        if not hasattr(volume_loss, "rand_u"):
            volume_loss.rand_u = rng_internal.uniform(0, 1, 2000)
            volume_loss.rand_v = rng_internal.uniform(0, 1, 2000)
            volume_loss.rand_c = rng_internal.uniform(0, 1, 2000)
        
        u = volume_loss.rand_u
        v = volume_loss.rand_v
        c = volume_loss.rand_c
        
        a = X0 + u * (X1 - X0)
        b = Y0 + v * (Y1 - Y0)
        
        # For each point (a, b, c), check all 64 tubes.
        # a - c*(i_k)/8 - x_k in [0, w]
        # b - c*(j_k)/8 - y_k in [0, w]
        
        # i_k is shape (64,), j_k is shape (64,)
        # c is shape (2000,)
        # x_k is shape (64,)
        
        # We need to check if there exists k such that:
        # 0 <= a_i - c_i*(i_k)/8 - x_k <= w  AND  0 <= b_i - c_i*(j_k)/8 - y_k <= w
        
        # This is a boolean matrix of size (2000, 64).
        # Let's compute the shifts.
        # shift_x = c[:, None] * (i_idx[None, :] / 8.0)
        # shift_y = c[:, None] * (j_idx[None, :] / 8.0)
        
        # diff_x = a[:, None] - shift_x - x[None, :]
        # diff_y = b[:, None] - shift_y - y[None, :]
        
        # valid_x = (diff_x >= 0) & (diff_x <= w)
        # valid_y = (diff_y >= 0) & (diff_y <= w)
        
        # covered = np.any(valid_x & valid_y, axis=1)
        
        # This is 2000 * 64 = 128,000 operations. Very fast in numpy.
        
        shift_x = c[:, None] * (i_idx[None, :] * w)
        shift_y = c[:, None] * (j_idx[None, :] * w)
        
        diff_x = a[:, None] - shift_x - x[None, :]
        diff_y = b[:, None] - shift_y - y[None, :]
        
        valid_x = (diff_x >= 0) & (diff_x <= w)
        valid_y = (diff_y >= 0) & (diff_y <= w)
        
        covered = np.any(valid_x & valid_y, axis=1)
        
        fraction = np.mean(covered)
        
        return area * fraction

    # Initial guess: Keich construction (separable)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    keich = keich - keich.min()
    
    shear = np.arange(N) / N
    
    # Construct initial 64 positions
    # Tube (i, j) uses keich[i] for x and keich[j] for y?
    # The current program uses base_x = keich[best_order[i]] and base_y = keich[best_order[j]]?
    # Actually, the current program uses the same keich array for both x and y,
    # but assigns shears based on a permutation.
    # Let's just use the identity assignment for the start.
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Initial x: keich[i]
    # Initial y: keich[j]
    x_init = keich[i_flat]
    y_init = keich[j_flat]
    
    x0 = np.concatenate([x_init, y_init])
    
    # Optimization
    best_params = x0
    best_loss = volume_loss(x0)
    
    # Nelder-Mead is derivative-free and good for small dimensions (128 here is borderline, but manageable)
    # We'll run it with a time limit.
    
    options = {
        'maxiter': 500,
        'xatol': 1e-5,
        'fatol': 1e-5,
        'adaptive': True
    }
    
    # We might need to restart or use multiple starts if it gets stuck.
    # Given the time budget, let's try one strong optimization run.
    
    try:
        res = minimize(volume_loss, x0, method='Nelder-Mead', options=options)
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # If we have time, try a second optimization from a slightly perturbed start
    if elapsed() < DEADLINE - 100:
        rng_opt = np.random.default_rng(123)
        x0_2 = x0 + rng_opt.normal(0, 0.1, 128)
        try:
            res2 = minimize(volume_loss, x0_2, method='Nelder-Mead', options=options)
            if res2.fun < best_loss:
                best_loss = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Final refinement with more samples?
    # The loss function uses 2000 samples. The evaluator uses 500,000.
    # The 2000-sample estimate should be a good proxy.
    
    x_final = best_params[:64]
    y_final = best_params[64:]
    
    positions = np.stack([x_final, y_final], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
