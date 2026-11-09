import numpy as np
from scipy.optimize import minimize


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximize the entropy ratio D / M where:
      D = H_{-1} (entropy of X - Y)
      M = max(H_0, H_1, H_2) (entropies of X, X+Y, X+2Y)

    We use a geometric heuristic: place mass on a set of points that lie close
    to a line of slope r* in the (y, x) plane, which keeps X - Y nearly constant
    (high D) while spreading out X, X+Y, X+2Y (low M relative to D).

    The optimal slope for minimizing max(H_0, H_1, H_2) while keeping D high
    is found by balancing the projection entropies. We try several candidate
    supports and pick the one with the best ratio.
    """
    n = 10

    def compute_ratio(P):
        """Compute D/M ratio for a given 10x10 distribution P."""
        P = np.clip(P, 0, None)
        s = P.sum()
        if s < 1e-12:
            return 0.0
        P = P / s

        # X is column index, Y is row index
        x_idx = np.arange(n)
        y_idx = np.arange(n)

        # Projection for slope r: value = x + r*y
        # We need the distribution of this value, then its Shannon entropy

        def proj_entropy(r):
            # For each cell (y, x), value = x + r*y
            values = x_idx[np.newaxis, :] + r * y_idx[:, np.newaxis]
            # Pool cells with equal values
            # Since values may be non-integer for non-integer r, use unique
            unique_vals, inverse = np.unique(values, return_inverse=True)
            counts = np.zeros(len(unique_vals))
            for idx, val in enumerate(inverse):
                counts[val] += P[idx // n, idx % n]
            # Shannon entropy in bits, ignoring zero probabilities
            probs = counts / counts.sum()
            nonzero = probs > 1e-15
            if not np.any(nonzero):
                return 0.0
            return -np.sum(probs[nonzero] * np.log2(probs[nonzero]))

        D = proj_entropy(-1.0)
        H0 = proj_entropy(0.0)
        H1 = proj_entropy(1.0)
        H2 = proj_entropy(2.0)
        M = max(H0, H1, H2)

        if M < 1e-12:
            return 0.0
        return D / M

    # Try different geometric supports
    best_ratio = 0.0
    best_P = np.ones((n, n)) / (n * n)

    # Strategy 1: Diagonal bands - pick points where x - y is close to a constant
    for c in range(-9, 10):
        # Points where x - y = c exactly
        mask = np.zeros((n, n), dtype=bool)
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                mask[y, x] = True
        if mask.sum() == 0:
            continue
        P = np.zeros((n, n))
        P[mask] = 1.0 / mask.sum()
        ratio = compute_ratio(P)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()

    # Strategy 2: Lines with various slopes - pick points along a line
    # Line: x = a*y + b
    for a in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0,
              0.2, 0.3, 0.7, 1.3, 1.7, 2.5, -0.3, -0.7, -1.3, -1.7, -2.5]:
        for b in np.arange(-10, 11, 1.0):
            mask = np.zeros((n, n), dtype=bool)
            for y in range(n):
                x_val = a * y + b
                x = int(round(x_val))
                # Check if close to integer
                if abs(x_val - x) < 0.01 and 0 <= x < n:
                    mask[y, x] = True
            if mask.sum() < 2:
                continue
            P = np.zeros((n, n))
            P[mask] = 1.0 / mask.sum()
            ratio = compute_ratio(P)
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P.copy()

    # Strategy 3: Weighted diagonals - give more weight to cells that help
    # balance the projections
    for c in range(-9, 10):
        mask = np.zeros((n, n), dtype=bool)
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                mask[y, x] = True
        if mask.sum() == 0:
            continue

        # Try weighting by x to shift the center of mass
        for shift in np.arange(-2, 3, 0.5):
            weights = np.zeros((n, n))
            for y in range(n):
                x = y + c
                if 0 <= x < n:
                    # Weight based on deviation from center
                    w = 1.0 + 0.1 * (x + y + 2*shift - n)
                    weights[y, x] = max(w, 0.1)
            if weights[mask].sum() < 1e-12:
                continue
            P = np.zeros((n, n))
            P[mask] = weights[mask]
            P /= P.sum()
            ratio = compute_ratio(P)
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P.copy()

    # Strategy 4: Random search with refinement
    np.random.seed(42)
    for _ in range(500):
        # Generate a random sparse support
        n_points = np.random.randint(3, 12)
        points = []
        for _ in range(n_points):
            y = np.random.randint(0, n)
            x = np.random.randint(0, n)
            points.append((y, x))
        # Remove duplicates
        points = list(set(points))
        if len(points) < 2:
            continue

        P = np.zeros((n, n))
        for y, x in points:
            P[y, x] = 1.0
        P /= P.sum()

        ratio = compute_ratio(P)
        if ratio > best_ratio:
            best_ratio = ratio
            best_P = P.copy()

    # Strategy 5: Optimized weights on a fixed support
    # Use the best diagonal support and optimize weights
    for c in range(-9, 10):
        mask = np.zeros((n, n), dtype=bool)
        for y in range(n):
            x = y + c
            if 0 <= x < n:
                mask[y, x] = True
        k = mask.sum()
        if k < 2 or k > 10:
            continue

        # Get indices
        ys, xs = np.where(mask)
        indices = np.arange(k)

        def neg_ratio(params):
            w = np.exp(params)  # Ensure positivity
            w = w / w.sum()
            P = np.zeros((n, n))
            for i, idx in enumerate(indices):
                P[ys[idx], xs[idx]] = w[i]
            return -compute_ratio(P)

        # Try multiple starting points
        for _ in range(5):
            x0 = np.random.randn(k) * 0.1
            try:
                result = minimize(neg_ratio, x0, method='Nelder-Mead',
                                  options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                w = np.exp(result.x)
                w = w / w.sum()
                P = np.zeros((n, n))
                for i, idx in enumerate(indices):
                    P[ys[idx], xs[idx]] = w[i]
                ratio = compute_ratio(P)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = P.copy()
            except Exception:
                pass

    # Final cleanup
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s < 1e-12:
        best_P = np.ones((n, n)) / (n * n)
    else:
        best_P /= s

    return best_P
