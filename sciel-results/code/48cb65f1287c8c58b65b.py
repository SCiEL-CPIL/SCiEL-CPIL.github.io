"""Stronger packing: L-BFGS-B with soft-repulsion energy + wall penalty."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200
P = 4  # exponent for soft repulsion (lower = smoother)


def _score(cents):
    """Return the packing radius for a set of centres."""
    d = np.sqrt(np.sum((cents[:, None, :] - cents[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    walls = np.min(np.minimum(np.abs(cents), 0.5 - np.abs(cents)))
    return min(min_pair / 2.0, walls)


def _energy(cents):
    """Soft energy: repulsion between close pairs + wall penalty."""
    d = np.sqrt(np.sum((cents[:, None, :] - cents[None, :, :]) ** 2, axis=-1))
    np.fill_diagonal(d, 1.0)
    # Only penalise when distance < 1 (which in unit-square normalised coords means very close)
    # Use a threshold so that far-away pairs contribute 0
    mask = d < 0.5  # within half a unit
    # For pairs closer than threshold, add 1/d^P
    inv_d = np.zeros_like(d)
    np.fill_diagonal(inv_d, 0)
    inv_d[mask] = (1.0 / d[mask]) ** P
    repulsion = inv_d.sum()

    # Wall penalty: push centres away from boundaries
    wall_viol = np.maximum(0.5 - np.abs(cents), 0)  # 0 at boundary, 0.5 at center
    # Actually we want penalty when close to boundary; use (0.5 - |c|)^2 where |c| > 0.5 - r
    # Simpler: quadratic penalty for being within a margin of the wall
    margin = 0.05
    wall_pen = 0.0
    for axis in range(2):
        # Distance to nearest wall along this axis
        dist = np.minimum(np.abs(cents[:, axis]), 0.5)
        # Penalise if dist < margin
        close = np.maximum(margin - dist, 0)
        wall_pen += (close ** 2).sum()

    return repulsion + 10.0 * wall_pen


def _grad_energy(cents):
    """Gradient of the soft energy (numerical or analytical)."""
    # Use central differences for simplicity; N=200, 400 coords, manageable
    eps = 1e-6
    g = np.zeros_like(cents)
    base = _energy(cents)
    for i in range(N):
        for j in range(2):
            orig = cents[i, j]
            cents[i, j] = orig + eps
            ep = _energy(cents)
            cents[i, j] = orig - eps
            em = _energy(cents)
            cents[i, j] = orig
            g[i, j] = (ep - em) / (2 * eps)
    return g


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # 10s safety margin

    # Initialisation: perturbed grid
    rng = np.random.default_rng(42)
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(np.float64)
    # Add small random perturbation to break symmetry
    cents = grid + rng.normal(0, 0.005, size=(N, 2))
    # Clip to interior
    cents = np.clip(cents, -0.49, 0.49)

    best_cents = cents.copy()
    best_score = _score(best_cents)

    # Try L-BFGS-B with the soft energy
    try:
        # Use gradient approximation via scipy's finite diff (more efficient than my loop)
        res = minimize(
            _energy,
            cents,
            method='L-BFGS-B',
            bounds=[(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 2),
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000},
        )
        if time.time() > deadline - 5:
            pass  # keep current best
        else:
            new_cents = res.x.reshape(N, 2)
            new_score = _score(new_cents)
            if new_score > best_score:
                best_cents = new_cents
                best_score = new_score
    except Exception:
        pass

    # Second pass: local refinement with a smaller perturbation and more iterations
    if time.time() < deadline - 20:
        # Use the best so far as start, add tiny noise
        cents2 = best_cents + rng.normal(0, 0.001, size=(N, 2))
        cents2 = np.clip(cents2, -0.49, 0.49)
        try:
            res2 = minimize(
                _energy,
                cents2,
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 2),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 8000},
            )
            new_cents2 = res2.x.reshape(N, 2)
            new_score2 = _score(new_cents2)
            if new_score2 > best_score:
                best_cents = new_cents2
                best_score = new_score2
        except Exception:
            pass

    # Third pass: coordinate ascent / Nelder-Mead on small groups if time allows
    if time.time() < deadline - 10:
        # Use a simple hill-climbing: randomly pick a few points and nudge them
        for iteration in range(2000):
            if time.time() > deadline - 5:
                break
            # Pick a random subset of points to move
            k = rng.integers(3, 10)
            idx = rng.choice(N, size=k, replace=False)
            for i in idx:
                # Small random step
                step = rng.normal(0, 0.002, size=2)
                new_pos = best_cents[i] + step
                # Keep inside
                new_pos = np.clip(new_pos, -0.49, 0.49)
                old = best_cents[i].copy()
                best_cents[i] = new_pos
                sc = _score(best_cents)
                if sc >= best_score - 1e-12:
                    best_score = sc
                else:
                    best_cents[i] = old

    # Final validation: ensure all centres are inside the square
    best_cents = np.clip(best_cents, -0.5, 0.5)

    return best_cents
