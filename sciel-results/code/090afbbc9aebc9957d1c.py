import numpy as np
from scipy.spatial import KDTree
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_valid(xy, r, tol=1e-9):
        n = len(r)
        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    def get_score(xy, r):
        return float(np.sum(r))

    def resolve_overlaps(xy, r, max_iters=500):
        """Resolve overlaps by moving circles, keeping radii fixed.
        Uses iterative pairwise separation with boundary clamping."""
        n = len(r)
        xy = xy.copy()
        for _ in range(max_iters):
            moved = False
            # Build spatial index for efficiency
            tree = KDTree(xy)
            for i in range(n):
                # Find neighbors within 2*max_r
                max_r = np.max(r)
                neighbors = tree.query_ball_point(xy[i], 2 * max_r)
                for j in neighbors:
                    if j <= i:
                        continue
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - 1e-12:
                        if dist < 1e-12:
                            dx, dy = 1.0, 0.0
                            dist = 1.0
                        else:
                            dx /= dist
                            dy /= dist
                        overlap = min_dist - dist
                        # Move each by half the overlap
                        xy[i, 0] -= 0.5 * overlap * dx
                        xy[i, 1] -= 0.5 * overlap * dy
                        xy[j, 0] += 0.5 * overlap * dx
                        xy[j, 1] += 0.5 * overlap * dy
                        moved = True
            # Clamp to boundaries
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], r[i], 1 - r[i])
                xy[i, 1] = np.clip(xy[i, 1], r[i], 1 - r[i])
            if not moved:
                break
        return xy

    def max_growth_for_circle(xy, r, idx, max_step=0.05, n_steps=50):
        """Try to grow circle idx by finding the maximum feasible radius increase.
        Returns new xy, r, and whether growth occurred."""
        n = len(r)
        best_r = r[idx]
        best_xy = xy.copy()
        
        # Binary search for max growth
        lo = 0.0
        hi = max_step
        for _ in range(n_steps):
            mid = (lo + hi) / 2
            r_trial = r.copy()
            r_trial[idx] = r[idx] + mid
            # Check boundary
            if r_trial[idx] > 0.5:
                hi = mid
                continue
            # Try to resolve overlaps
            xy_trial = resolve_overlaps(xy, r_trial, max_iters=200)
            if is_valid(xy_trial, r_trial, tol=1e-9):
                best_r = r_trial[idx]
                best_xy = xy_trial
                lo = mid
            else:
                hi = mid
        return best_xy, np.where(np.arange(n) == idx, best_r, r), best_r > r[idx] + 1e-12

    def grow_all_passes(xy, r, max_passes=10, max_step=0.03):
        """Multiple passes of growing circles in different orders."""
        n = len(r)
        for p in range(max_passes):
            if time.time() > deadline - 5:
                break
            # Alternate between random and sorted order
            if p % 2 == 0:
                order = np.random.permutation(n)
            else:
                order = np.argsort(-r)  # largest first
            for idx in order:
                if time.time() > deadline - 5:
                    break
                new_xy, new_r, grew = max_growth_for_circle(xy, r, idx, max_step=max_step)
                if grew:
                    xy, r = new_xy, new_r
        return xy, r

    def initial_grid_config(rng):
        """Create a dense initial grid configuration."""
        # Use a 6x6 grid but only take 32 circles
        side = 6
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
            r[i] = 0.5 / side - 0.002
        return xy, r

    def initial_hex_config(rng):
        """Hexagonal packing initial config."""
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        # Hexagonal grid
        dx = 0.15
        dy = dx * np.sqrt(3) / 2
        count = 0
        row = 0
        while count < N:
            col = 0
            while count < N:
                x = col * dx + (dx / 2 if row % 2 == 1 else 0)
                y = row * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xy[count] = [x, y]
                    r[count] = dx / 2 - 0.002
                    count += 1
                col += 1
            row += 1
        return xy, r

    def initial_random_config(rng):
        """Random initial config with small radii."""
        xy = rng.uniform(0.1, 0.9, size=(N, 2))
        r = rng.uniform(0.04, 0.07, size=N)
        # Resolve overlaps
        xy = resolve_overlaps(xy, r, max_iters=300)
        return xy, r

    best_xy = None
    best_r = None
    best_score = -np.inf

    rng = np.random.RandomState(42)

    # Strategy 1: Grid start
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        if attempt == 0:
            xy, r = initial_grid_config(rng)
        elif attempt == 1:
            xy, r = initial_hex_config(rng)
        elif attempt == 2:
            xy, r = initial_random_config(rng)
        else:
            xy, r = initial_grid_config(rng)
            xy += rng.uniform(-0.01, 0.01, size=(N, 2))
            xy = resolve_overlaps(xy, r, max_iters=200)

        if not is_valid(xy, r):
            xy = resolve_overlaps(xy, r, max_iters=500)
            if not is_valid(xy, r):
                continue

        # Grow
        xy, r = grow_all_passes(xy, r, max_passes=8, max_step=0.02)

        if is_valid(xy, r):
            score = get_score(xy, r)
            if score > best_score:
                best_score = score
                best_xy = xy.copy()
                best_r = r.copy()

    # Strategy 2: From best, try more aggressive growth
    if best_xy is not None and time.time() < deadline - 20:
        xy, r = best_xy.copy(), best_r.copy()
        xy, r = grow_all_passes(xy, r, max_passes=15, max_step=0.01)
        if is_valid(xy, r):
            score = get_score(xy, r)
            if score > best_score:
                best_score = score
                best_xy = xy.copy()
                best_r = r.copy()

    # Strategy 3: Try different random seeds for random starts
    for seed in [123, 456, 789]:
        if time.time() > deadline - 10:
            break
        rng2 = np.random.RandomState(seed)
        xy, r = initial_random_config(rng2)
        xy = resolve_overlaps(xy, r, max_iters=500)
        if not is_valid(xy, r):
            continue
        xy, r = grow_all_passes(xy, r, max_passes=10, max_step=0.02)
        if is_valid(xy, r):
            score = get_score(xy, r)
            if score > best_score:
                best_score = score
                best_xy = xy.copy()
                best_r = r.copy()

    if best_xy is not None:
        r = np.clip(best_r, 0, 0.5)
        best_xy[:, 0] = np.clip(best_xy[:, 0], r, 1 - r)
        best_xy[:, 1] = np.clip(best_xy[:, 1], r, 1 - r)
        if is_valid(best_xy, r):
            return np.column_stack([best_xy, r])

    # Fallback
    side = 6
    cells = []
    for i in range(N):
        row = i // side
        col = i % side
        cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.5 / side - 1e-3])
    return np.array(cells)
