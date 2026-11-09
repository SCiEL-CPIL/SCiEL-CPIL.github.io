import numpy as np
import time

TOL = 1e-7
N = 10

def _is_valid(centers, sides, tol=TOL):
    """Check if the packing is valid."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False

    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False

    # Check no overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return False
    return True

def _raw_objective(centers, sides):
    """Return sum of sides if valid, else -inf."""
    n = len(sides)
    tol = TOL

    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return -np.inf
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return -np.inf
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return -np.inf

    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            min_dx = (sides[i] + sides[j])/2 - tol
            min_dy = (sides[i] + sides[j])/2 - tol
            if dx < min_dx and dy < min_dy:
                return -np.inf

    return float(np.sum(sides))

def _smooth_penalty(centers, sides):
    """
    Vectorized smooth penalty for invalid configurations.
    Returns a scalar penalty >= 0.  Zero when configuration is valid.
    """
    n = len(sides)
    penalty = 0.0

    # Boundary penalties (squared violations)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        v1 = max(0.0, s/2 - cx)
        v2 = max(0.0, cx + s/2 - 1.0)
        v3 = max(0.0, s/2 - cy)
        v4 = max(0.0, cy + s/2 - 1.0)
        penalty += (v1*v1 + v2*v2 + v3*v3 + v4*v4) * 50.0
        if s <= 0:
            penalty += (s*s) * 50.0

    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            req = (sides[i] + sides[j]) / 2.0
            ox = max(0.0, req - dx)
            oy = max(0.0, req - dy)
            # Penalize only if overlapping in BOTH axes
            if ox > 0.0 and oy > 0.0:
                penalty += (ox*ox + oy*oy) * 20.0

    return penalty

def _sa_objective(centers, sides):
    """
    Combined objective for SA: return -sum(sides) + penalty.
    Lower is better (we minimize).
    """
    n = len(sides)
    total = 0.0
    for i in range(n):
        total += sides[i]

    pen = _smooth_penalty(centers, sides)
    return -total + pen

def _clamp(centers, sides):
    """Clamp centers so squares stay within [0,1]^2 given their sides."""
    for i in range(N):
        s = sides[i]
        centers[i, 0] = np.clip(centers[i, 0], s/2.0, 1.0 - s/2.0)
        centers[i, 1] = np.clip(centers[i, 1], s/2.0, 1.0 - s/2.0)
    return centers

def _generate_initial_config():
    """Generate the known 3.0 packing."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)

    # Bottom half: 2 squares of side 0.5
    sides[0] = 0.5
    sides[1] = 0.5
    centers[0, 0] = 0.25
    centers[0, 1] = 0.25
    centers[1, 0] = 0.75
    centers[1, 1] = 0.25

    # Top half: 8 squares of side 0.25
    sides[2] = 0.25
    centers[2, 0] = 0.125
    centers[2, 1] = 0.625
    sides[3] = 0.25
    centers[3, 0] = 0.375
    centers[3, 1] = 0.625
    sides[4] = 0.25
    centers[4, 0] = 0.125
    centers[4, 1] = 0.875
    sides[5] = 0.25
    centers[5, 0] = 0.375
    centers[5, 1] = 0.875
    sides[6] = 0.25
    centers[6, 0] = 0.625
    centers[6, 1] = 0.625
    sides[7] = 0.25
    centers[7, 0] = 0.875
    centers[7, 1] = 0.625
    sides[8] = 0.25
    centers[8, 0] = 0.625
    centers[8, 1] = 0.875
    sides[9] = 0.25
    centers[9, 0] = 0.875
    centers[9, 1] = 0.875

    return centers, sides

def _sa_search(centers, sides, deadline, rng):
    """
    Simulated annealing local search.
    Returns (best_centers, best_sides, best_score).
    """
    start_time = time.time()
    n = N

    cur_centers = centers.copy()
    cur_sides = sides.copy()
    cur_val = _sa_objective(cur_centers, cur_sides)

    best_centers = cur_centers.copy()
    best_sides = cur_sides.copy()
    best_raw = _raw_objective(best_centers, best_sides)

    T0 = 0.05
    Tend = 1e-5
    # We'll run until deadline
    # Estimate iterations per second roughly; just use a fixed step and check time
    # Use geometric cooling: T *= cooling_factor each step
    # Choose cooling factor so that we use the available time well.
    # We'll just do a fixed number of steps per time check.

    step = 0
    # We'll run in batches to check time
    batch_size = 500

    # Initial step size
    pos_step = 0.02
    size_step = 0.01

    while time.time() < deadline:
        for _ in range(batch_size):
            if time.time() >= deadline:
                break
            step += 1

            # Decide what to perturb
            r = rng.random()
            if r < 0.6:
                # Perturb position of one or two squares
                k = 1 if rng.random() < 0.7 else 2
                idx = rng.choice(n, size=k, replace=False)
                for i in idx:
                    cur_centers[i, 0] += rng.normal(0, pos_step)
                    cur_centers[i, 1] += rng.normal(0, pos_step)
            else:
                # Perturb size of one square
                i = rng.randint(n)
                cur_sides[i] += rng.normal(0, size_step)
                cur_sides[i] = max(cur_sides[i], 0.01)
                cur_sides[i] = min(cur_sides[i], 1.0)

            cur_centers = _clamp(cur_centers, cur_sides)

            new_val = _sa_objective(cur_centers, cur_sides)
            delta = new_val - cur_val

            # Temperature
            # Use a simple cooling schedule based on step
            # T = T0 * (Tend/T0) ** (step / total_steps)
            # But we don't know total_steps. Use a fixed cooling rate.
            # Let's use T = T0 * exp(-step * 0.0005) roughly
            # Actually let's use a time-based temperature.
            elapsed = time.time() - start_time
            total_time = max(deadline - start_time, 1.0)
            frac = min(1.0, elapsed / total_time)
            T = T0 * (Tend / T0) ** frac

            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                cur_val = new_val

        # Update best
        raw = _raw_objective(cur_centers, cur_sides)
        if raw > best_raw:
            best_raw = raw
            best_centers = cur_centers.copy()
            best_sides = cur_sides.copy()

        # Occasionally reset to best to avoid drift
        if step % 5000 == 0:
            cur_centers = best_centers.copy()
            cur_sides = best_sides.copy()
            cur_val = _sa_objective(cur_centers, cur_sides)

    return best_centers, best_sides, best_raw

def _polish(centers, sides, deadline):
    """
    Fine local search with small steps to polish the configuration.
    """
    start_time = time.time()
    n = N
    cur_centers = centers.copy()
    cur_sides = sides.copy()
    best_centers = cur_centers.copy()
    best_sides = cur_sides.copy()
    best_raw = _raw_objective(best_centers, best_sides)

    rng = np.random.RandomState(12345)
    pos_step = 0.005
    size_step = 0.002
    T0 = 0.01
    Tend = 1e-6

    step = 0
    batch_size = 500

    while time.time() < deadline:
        for _ in range(batch_size):
            if time.time() >= deadline:
                break
            step += 1

            r = rng.random()
            if r < 0.6:
                k = 1 if rng.random() < 0.7 else 2
                idx = rng.choice(n, size=k, replace=False)
                for i in idx:
                    cur_centers[i, 0] += rng.normal(0, pos_step)
                    cur_centers[i, 1] += rng.normal(0, pos_step)
            else:
                i = rng.randint(n)
                cur_sides[i] += rng.normal(0, size_step)
                cur_sides[i] = max(cur_sides[i], 0.01)
                cur_sides[i] = min(cur_sides[i], 1.0)

            cur_centers = _clamp(cur_centers, cur_sides)

            new_val = _sa_objective(cur_centers, cur_sides)
            old_val = _sa_objective(cur_centers, cur_sides)  # same as new after clamp? No, we need old before perturbation
            # Actually we need to store old_val before perturbation. Let me restructure.
            # For simplicity in polish, just accept if it improves the raw objective
            raw = _raw_objective(cur_centers, cur_sides)
            if raw > best_raw:
                best_raw = raw
                best_centers = cur_centers.copy()
                best_sides = cur_sides.copy()
                # Reset current to best
                cur_centers = best_centers.copy()
                cur_sides = best_sides.copy()

    return best_centers, best_sides, best_raw

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)

    Uses simulated annealing with a smooth penalty to search for a better packing
    than the known 3.0 configuration, followed by a polish pass.
    """
    deadline = time.time() + 580  # Safety margin before 630s limit

    # Start with known 3.0 packing
    centers, sides = _generate_initial_config()
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = _raw_objective(best_centers, best_sides)

    # Try multiple SA runs with different starting points
    seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]

    for seed in seeds:
        if time.time() > deadline - 30:
            break

        rng = np.random.RandomState(seed)

        # Perturb the initial configuration
        if seed % 3 == 0:
            # Small perturbation
            perturbed_centers = centers + rng.normal(0, 0.01, (N, 2))
            perturbed_sides = sides * (1 + rng.normal(0, 0.02, N))
        elif seed % 3 == 1:
            # Medium perturbation
            perturbed_centers = centers + rng.normal(0, 0.03, (N, 2))
            perturbed_sides = sides * (1 + rng.normal(0, 0.05, N))
        else:
            # Larger perturbation
            perturbed_centers = centers + rng.normal(0, 0.05, (N, 2))
            perturbed_sides = sides * (1 + rng.normal(0, 0.1, N))

        perturbed_sides = np.clip(perturbed_sides, 0.01, 1.0)
        perturbed_centers = _clamp(perturbed_centers, perturbed_sides)

        remaining_time = deadline - time.time()
        if remaining_time < 30:
            break

        search_deadline = time.time() + min(40.0, remaining_time - 10)
        if search_deadline < time.time() + 5:
            break

        new_centers, new_sides, new_score = _sa_search(
            perturbed_centers, perturbed_sides, search_deadline, rng
        )

        if new_score > best_score:
            best_centers = new_centers
            best_sides = new_sides
            best_score = new_score

    # Final polish pass
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        polish_deadline = time.time() + min(remaining_time - 10, 60.0)
        if polish_deadline > time.time() + 5:
            best_centers, best_sides, best_score = _polish(
                best_centers, best_sides, polish_deadline
            )

    # Final validation
    if not _is_valid(best_centers, best_sides):
        # Fall back to known valid packing
        best_centers, best_sides = _generate_initial_config()

    return best_centers, best_sides
