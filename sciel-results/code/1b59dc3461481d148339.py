# EVOLVE-BLOCK-START
import numpy as np
import time


def _entropy_from_counts(counts):
    """Shannon entropy in bits from a count array (zeros ignored)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def _fast_ratio(P):
    """
    Fast computation of D/M using vectorized bincount-style approach.
    P[y, x] is the joint probability for x, y in 0..9.
    """
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total = P.sum()
    if total <= 1e-9:
        return 0.0

    P = P.copy()
    P[P < 0] = 0.0
    P = P / P.sum()

    y_idx, x_idx = np.indices((10, 10))
    flat_P = P.flatten()

    # x - y: range -9..9, offset by 9 to get 0..18
    v_minus1 = (x_idx - y_idx + 9).flatten()
    counts_minus1 = np.bincount(v_minus1, weights=flat_P, minlength=19)
    H_minus1 = _entropy_from_counts(counts_minus1)

    # x: range 0..9
    v_x = x_idx.flatten()
    counts_x = np.bincount(v_x, weights=flat_P, minlength=10)
    H_0 = _entropy_from_counts(counts_x)

    # x + y: range 0..18
    v_plus1 = (x_idx + y_idx).flatten()
    counts_plus1 = np.bincount(v_plus1, weights=flat_P, minlength=19)
    H_1 = _entropy_from_counts(counts_plus1)

    # x + 2y: range 0..27
    v_plus2 = (x_idx + 2 * y_idx).flatten()
    counts_plus2 = np.bincount(v_plus2, weights=flat_P, minlength=28)
    H_2 = _entropy_from_counts(counts_plus2)

    M = max(H_0, H_1, H_2)
    if M <= 1e-9:
        return 0.0

    return H_minus1 / M


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    Optimizes the ratio D/M where D = H_{-1} and M = max(H_0, H_1, H_2).
    Uses a structured search over diagonal-weighted distributions combined with
    local perturbations to find high-ratio configurations.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s limit

    n = 10
    rng = np.random.RandomState(12345)

    y_idx, x_idx = np.indices((n, n))
    diff = x_idx - y_idx  # x - y, range -9..9

    # Precompute which cells belong to each diagonal (x - y = d)
    diagonals = {}
    for d in range(-9, 10):
        mask = (diff == d)
        diagonals[d] = mask
        # Store flat indices for quick access
        diagonals[d + 9] = mask.flatten().nonzero()[0]

    # The key insight: to maximize D/M, we want the distribution of (x-y) to be
    # as spread out as possible (high H_{-1}), while keeping the marginal distributions
    # of x, x+y, x+2y as concentrated as possible (low max entropy).
    #
    # Strategy: assign weights to diagonals. Within each diagonal, the cells have
    # different x, y values. We can choose how to distribute weight within a diagonal.
    #
    # A simple approach: put all weight on specific diagonals, and within each diagonal,
    # put weight on a single cell (or a few cells) to minimize the marginal entropies.
    #
    # Let's try: for each subset of diagonals, assign equal weight to those diagonals,
    # and within each diagonal, put all weight on the cell that minimizes the marginal
    # entropies.
    #
    # Actually, a better approach: parameterize by weights on diagonals.
    # For each diagonal d, choose a cell (x, y) with x - y = d. The weight on that
    # cell contributes to the marginals.
    #
    # Let's try a simpler structured search:
    # 1. Choose a set of k diagonals to use.
    # 2. Assign weights w_d to each chosen diagonal (sum to 1).
    # 3. For each diagonal, pick the cell that best minimizes the marginal entropies.
    #
    # Even simpler: try all single-diagonal distributions (weight 1 on one diagonal,
    # spread evenly or concentrated).

    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = _fast_ratio(best_P)

    # Try single diagonal: all weight on one diagonal
    for d in range(-9, 10):
        mask = diagonals[d]
        num_cells = mask.sum()
        if num_cells == 0:
            continue
        # Spread evenly on the diagonal
        P = np.zeros((n, n), dtype=np.float64)
        P[mask] = 1.0 / num_cells
        score = _fast_ratio(P)
        if score > best_score:
            best_P = P.copy()
            best_score = score

        # Concentrate on one cell of the diagonal
        cells = np.argwhere(mask)
        for (yy, xx) in cells:
            P = np.zeros((n, n), dtype=np.float64)
            P[yy, xx] = 1.0
            score = _fast_ratio(P)
            if score > best_score:
                best_P = P.copy()
                best_score = score

    # Try two diagonals: weight w on diagonal d1, 1-w on diagonal d2
    # For each pair, try several weight splits and cell choices
    diag_list = list(range(-9, 10))
    for i, d1 in enumerate(diag_list):
        if time.time() > deadline - 60:
            break
        for d2 in diag_list[i+1:]:
            mask1 = diagonals[d1]
            mask2 = diagonals[d2]
            cells1 = np.argwhere(mask1)
            cells2 = np.argwhere(mask2)

            # Try a few weight splits
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                # Try concentrating on first cell of each diagonal
                if len(cells1) > 0 and len(cells2) > 0:
                    P = np.zeros((n, n), dtype=np.float64)
                    P[cells1[0][0], cells1[0][1]] = w
                    P[cells2[0][0], cells2[0][1]] = 1.0 - w
                    score = _fast_ratio(P)
                    if score > best_score:
                        best_P = P.copy()
                        best_score = score

    # Now do a local search: perturb the best solution found
    # Use a simple coordinate ascent / local perturbation approach
    current_P = best_P.copy()
    current_score = best_score

    for iteration in range(2000):
        if time.time() > deadline - 30:
            break

        # Pick a random cell to perturb
        yy, xx = rng.randint(n), rng.randint(n)

        # Try moving some weight from (yy, xx) to another cell
        # Or adjust the weight at (yy, xx)
        old_val = current_P[yy, xx]

        # Try a few perturbations
        for factor in [0.5, 0.7, 1.3, 2.0, 0.0]:
            new_P = current_P.copy()
            new_P[yy, xx] = old_val * factor if factor > 0 else 0.0

            # Renormalize
            total = new_P.sum()
            if total <= 1e-9:
                continue
            new_P = new_P / total

            score = _fast_ratio(new_P)
            if score > current_score:
                current_P = new_P
                current_score = score
                if current_score > best_score:
                    best_P = current_P.copy()
                    best_score = current_score

    # Final cleanup: ensure validity
    best_P[best_P < 0] = 0
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)

    return best_P


# EVOLVE-BLOCK-END
