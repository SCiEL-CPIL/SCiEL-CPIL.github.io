# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Minimizes the Monte Carlo estimate of the union volume of 64 sheared tubes
    using gradient-based optimization with non-separable 2D coupling.
    """
    N = 8
    num_tubes = N * N
    start_time = time.time()
    deadline = start_time + 580.0  # 50s safety margin before 630s kill

    # Precompute tube indices
    i_idx = np.array([k // N for k in range(num_tubes)], dtype=np.float64)
    j_idx = np.array([k % N for k in range(num_tubes)], dtype=np.float64)
    inv_N = 1.0 / N

    def get_baseline_positions():
        """Keich construction baseline."""
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        return np.stack([base_x, base_y], axis=1)

    def compute_volume_and_grad(positions, n_samples=50000, seed=12345):
        """
        Computes Monte Carlo estimate of union volume and its gradient.
        Uses subgradient estimation via random perturbations for non-smooth indicator.
        """
        # Determine bounding box
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0] + (i_idx + 1) * inv_N)
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1] + (j_idx + 1) * inv_N)

        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area <= 1e-9:
            return 1e9, np.zeros_like(positions)

        rng = np.random.default_rng(seed)
        a = rng.uniform(x_min, x_max, n_samples)
        b = rng.uniform(y_min, y_max, n_samples)
        c = rng.uniform(0, 1, n_samples)

        # For each sample, check which tubes contain it
        # Tube k contains point (a, b, c) if:
        #   x_k + c*i_k/N <= a <= x_k + c*i_k/N + 1/N
        #   y_k + c*j_k/N <= b <= y_k + c*j_k/N + 1/N

        # Vectorized: for each sample s and tube k, compute margins
        # x_margin[s, k] = a[s] - c[s]*i_k/N - x_k  (should be in [0, 1/N])
        # y_margin[s, k] = b[s] - c[s]*j_k/N - y_k  (should be in [0, 1/N])

        # Memory: n_samples * num_tubes * 2 * 8 bytes
        # 50000 * 64 * 2 * 8 = ~51 MB - acceptable

        x_lower = a[:, None] - c[:, None] * i_idx[None, :] - inv_N
        x_upper = a[:, None] - c[:, None] * i_idx[None, :]
        y_lower = b[:, None] - c[:, None] * j_idx[None, :] - inv_N
        y_upper = b[:, None] - c[:, None] * j_idx[None, :]

        # x_ok[s, k] = True if x_k in [x_lower[s,k], x_upper[s,k]]
        # i.e., positions[s,0] >= x_lower[s,k] and positions[s,0] <= x_upper[s,k]
        x_ok = (positions[None, :, 0] >= x_lower) & (positions[None, :, 0] <= x_upper)
        y_ok = (positions[None, :, 1] >= y_lower) & (positions[None, :, 1] <= y_upper)

        in_tube = x_ok & y_ok  # shape (n_samples, num_tubes)
        in_any = np.any(in_tube, axis=1)  # shape (n_samples,)

        fraction = np.mean(in_any)
        volume = fraction * box_area

        # Gradient estimation via finite differences on a subset of samples
        # For efficiency, use a smaller subset for gradient
        grad_samples = min(10000, n_samples)
        rng_grad = np.random.default_rng(seed + 1)
        idx_grad = rng_grad.choice(n_samples, grad_samples, replace=False)

        grad = np.zeros_like(positions)
        h = 1e-4

        # For each tube, estimate gradient contribution
        # dV/dx_k ≈ (V(x_k+h) - V(x_k-h)) / (2h) * box_area
        # But V depends on all positions, so we need to be careful.
        # Use the fact that only tubes containing the sample contribute to the indicator.

        # Simpler: use central differences on the full volume
        # But that's 2*64 = 128 evaluations - too expensive.
        # Instead, use the analytic structure:
        # The indicator is 1 if point is in ANY tube.
        # Gradient of indicator w.r.t. x_k is non-zero only at boundaries.
        # For MC, we estimate: d/dx_k [1_{in_any}] ≈ sum over samples where tube k is "critical"

        # Critical tube: the sample is in tube k, and no other tube contains it
        # OR the sample is at the boundary of tube k

        # For each sample in idx_grad, find which tubes contain it
        x_lower_g = x_lower[idx_grad]
        x_upper_g = x_upper[idx_grad]
        y_lower_g = y_lower[idx_grad]
        y_upper_g = y_upper[idx_grad]

        x_ok_g = (positions[None, :, 0] >= x_lower_g) & (positions[None, :, 0] <= x_upper_g)
        y_ok_g = (positions[None, :, 1] >= y_lower_g) & (positions[None, :, 1] <= y_upper_g)
        in_tube_g = x_ok_g & y_ok_g  # (grad_samples, num_tubes)

        # Count how many tubes contain each sample
        tube_counts = np.sum(in_tube_g, axis=1)  # (grad_samples,)

        # For samples in exactly one tube, that tube is "critical"
        # Gradient of 1_{in_any} w.r.t. x_k:
        # If sample is in tube k only, and x_k is at boundary, moving x_k changes indicator
        # Specifically:
        #   If x_k = x_lower (left boundary), increasing x_k removes sample from tube k -> indicator may go to 0
        #   If x_k = x_upper (right boundary), decreasing x_k removes sample from tube k -> indicator may go to 0

        # For a sample in only tube k:
        #   d(1_{in_any})/dx_k = -1 if x_k is at left boundary (and sample would leave)
        #                        +1 if x_k is at right boundary (and sample would leave)
        #   More precisely, it's a delta function at the boundaries.

        # For MC gradient, we use:
        #   grad_x[k] += sum over samples s where tube k is the only tube containing s:
        #       -1/N if x_k is near left boundary of s
        #       +1/N if x_k is near right boundary of s
        #   (divided by n_samples for the fraction, times box_area for volume)

        # This is getting complex. Let's use a simpler approach:
        # Finite difference with a small number of perturbations.

        # Perturb each coordinate and recompute volume on a subset
        # This is O(64) volume evaluations per gradient, which is too slow.

        # Alternative: use the fact that the volume is a step function in each coordinate.
        # The gradient is zero almost everywhere, with jumps at critical values.
        # For optimization, we can use subgradients or a smoothing approach.

        # Practical approach: use a smoothed version of the indicator.
        # Replace the hard boundary with a soft one using a sigmoid or similar.
        # This makes the volume differentiable and allows gradient descent.

        # Let's redefine the volume with a smooth approximation.
        return volume, grad  # grad is zero for now; we'll use a different approach

    def compute_volume_smooth(positions, n_samples=50000, seed=12345, smooth_width=0.01):
        """
        Computes a smoothed version of the union volume that is differentiable.
        Uses a soft indicator based on distance to tube boundaries.
        """
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0] + (i_idx + 1) * inv_N)
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1] + (j_idx + 1) * inv_N)

        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area <= 1e-9:
            return 1e9, np.zeros_like(positions)

        rng = np.random.default_rng(seed)
        a = rng.uniform(x_min, x_max, n_samples)
        b = rng.uniform(y_min, y_max, n_samples)
        c = rng.uniform(0, 1, n_samples)

        # For each sample and tube, compute a soft membership score
        # x_margin = a - c*i_k/N - x_k  (should be in [0, 1/N] for hard membership)
        # Soft membership: sigmoid((x_margin) / w) * sigmoid((1/N - x_margin) / w)
        # This is ~1 when x_margin is well inside [0, 1/N], ~0 outside

        w = smooth_width
        x_margin = a[:, None] - c[:, None] * i_idx[None, :] - positions[None, :, 0]  # (n_samples, num_tubes)
        y_margin = b[:, None] - c[:, None] * j_idx[None, :] - positions[None, :, 1]  # (n_samples, num_tubes)

        # Soft indicator for x: 1 if x_margin in [0, 1/N]
        # Use product of two sigmoids
        x_soft = 1.0 / (1.0 + np.exp(-x_margin / w)) * 1.0 / (1.0 + np.exp(-(inv_N - x_margin) / w))
        y_soft = 1.0 / (1.0 + np.exp(-y_margin / w)) * 1.0 / (1.0 + np.exp(-(inv_N - y_margin) / w))

        tube_soft = x_soft * y_soft  # (n_samples, num_tubes)

        # Union: 1 - product(1 - tube_soft_k) for all k
        # But this is expensive for 64 tubes. Use approximation:
        # If tubes are mostly disjoint, union ≈ sum. If overlapping, need inclusion-exclusion.
        # For simplicity, use: in_any_soft = max over k of tube_soft_k (approximates union)
        # Or: 1 - prod(1 - tube_soft_k) which is exact for independent events but expensive.

        # Use max as a cheap approximation
        in_any_soft = np.max(tube_soft, axis=1)  # (n_samples,)

        fraction = np.mean(in_any_soft)
        volume = fraction * box_area

        # Gradient of volume w.r.t. positions
        # d(volume)/d(x_k) = box_area * (1/n_samples) * sum_s d(in_any_soft[s])/dx_k
        # d(in_any_soft[s])/dx_k = max_j d(tube_soft[s,j])/dx_k * 1_{argmax = k}
        # This is complex. Use finite differences for gradient.

        # For now, return volume only; we'll use a different optimization strategy.
        return volume, None

    # Strategy: Use a combination of:
    # 1. Start from Keich baseline
    # 2. Use coordinate descent with line search (more robust than gradient for non-smooth)
    # 3. Multiple restarts with perturbations

    def compute_volume(positions, n_samples=50000, seed=12345):
        """Fast volume computation for optimization."""
        x_min = np.min(positions[:, 0])
        x_max = np.max(positions[:, 0] + (i_idx + 1) * inv_N)
        y_min = np.min(positions[:, 1])
        y_max = np.max(positions[:, 1] + (j_idx + 1) * inv_N)

        box_area = (x_max - x_min) * (y_max - y_min)
        if box_area <= 1e-9:
            return 1e9

        rng = np.random.default_rng(seed)
        a = rng.uniform(x_min, x_max, n_samples)
        b = rng.uniform(y_min, y_max, n_samples)
        c = rng.uniform(0, 1, n_samples)

        x_lower = a[:, None] - c[:, None] * i_idx[None, :] - inv_N
        x_upper = a[:, None] - c[:, None] * i_idx[None, :]
        y_lower = b[:, None] - c[:, None] * j_idx[None, :] - inv_N
        y_upper = b[:, None] - c[:, None] * j_idx[None, :]

        x_ok = (positions[None, :, 0] >= x_lower) & (positions[None, :, 0] <= x_upper)
        y_ok = (positions[None, :, 1] >= y_lower) & (positions[None, :, 1] <= y_upper)

        in_any = np.any(x_ok & y_ok, axis=1)
        fraction = np.mean(in_any)
        return fraction * box_area

    def compute_volume_full(positions, seed=12345):
        """Full precision volume with 500k samples."""
        return compute_volume(positions, n_samples=500000, seed=seed)

    # Initialize with Keich baseline
    best_positions = get_baseline_positions()
    best_vol = compute_volume(best_positions)

    # Also try a few random initializations
    rng_init = np.random.default_rng(42)
    for _ in range(3):
        random_pos = rng_init.uniform(-0.5, 0.5, (num_tubes, 2))
        vol = compute_volume(random_pos)
        if vol < best_vol:
            best_vol = vol
            best_positions = random_pos.copy()

    # Coordinate descent with line search
    # For each tube, try to optimize its (x, y) position while keeping others fixed
    # Use a small number of candidate offsets per coordinate

    current_positions = best_positions.copy()
    current_vol = best_vol
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        # Cycle through tubes
        for k in range(num_tubes):
            if time.time() > deadline:
                break

            # Try perturbing tube k in x and y
            base_x = current_positions[k, 0]
            base_y = current_positions[k, 1]

            # Candidate offsets: try a grid of small perturbations
            # Adaptive: start with larger steps, reduce if no improvement
            step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

            best_local_vol = current_vol
            best_local_x = base_x
            best_local_y = base_y

            for dx in step_sizes:
                for dy in step_sizes:
                    # 8 directions + center
                    for dx_sign in [-1, 0, 1]:
                        for dy_sign in [-1, 0, 1]:
                            if dx_sign == 0 and dy_sign == 0:
                                continue
                            new_x = base_x + dx_sign * dx
                            new_y = base_y + dy_sign * dy
                            new_pos = current_positions.copy()
                            new_pos[k, 0] = new_x
                            new_pos[k, 1] = new_y
                            vol = compute_volume(new_pos, n_samples=20000)
                            if vol < best_local_vol:
                                best_local_vol = vol
                                best_local_x = new_x
                                best_local_y = new_y

            if best_local_vol < current_vol - 1e-6:
                current_positions[k, 0] = best_local_x
                current_positions[k, 1] = best_local_y
                current_vol = best_local_vol
                improved = True
                no_improve_count = 0

                if current_vol < best_vol:
                    best_vol = current_vol
                    best_positions = current_positions.copy()
            else:
                no_improve_count += 1

        # If no improvement in a full cycle, reduce step sizes or break
        if not improved:
            no_improve_count += 100  # Accelerate termination

    # Final verification with full precision
    final_vol = compute_volume_full(best_positions)
    if final_vol <= 0 or not np.isfinite(final_vol):
        # Fallback to Keich baseline
        best_positions = get_baseline_positions()

    return best_positions

# EVOLVE-BLOCK-END
