"""Sutton-Chen TM cluster global minima search.
Uses basin-hopping with geometric templates and a multi-start strategy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if positions.shape[0] == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Vectorized distance check
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _energy(pos, kind):
    """Compute Sutton-Chen energy."""
    try:
        e, _ = sutton_chen_energy_gradient(pos, kind)
        return e if np.isfinite(e) else 1e10
    except Exception:
        return 1e10


def _polish(positions, kind, max_iter=1000, tol=1e-10):
    """High-precision polish pass using gradient-based optimization."""
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        return _energy(pos, kind)

    def gradient_fn(x):
        pos = x.reshape(-1, 3)
        try:
            _, g = sutton_chen_energy_gradient(pos, kind)
            return g.ravel()
        except Exception:
            return np.zeros_like(x)

    try:
        result = minimize(
            energy_fn,
            positions.ravel(),
            jac=gradient_fn,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol, 'maxls': 50}
        )
        if result.success and np.isfinite(result.fun):
            polished = result.x.reshape(-1, 3)
            if _is_valid(polished):
                return result.fun, polished
        elif np.isfinite(result.fun):
            polished = result.x.reshape(-1, 3)
            if _is_valid(polished):
                return result.fun, polished
    except Exception:
        pass
    return _energy(positions, kind), positions


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms."""
    max_r = int(np.ceil(n ** (1 / 3))) + 2
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_random_compact(n, rng, radius_scale=1.0):
    """Generate a random compact cluster."""
    radius = 1.2 + 0.4 * (n / 50.0) * radius_scale
    pos = rng.uniform(-radius, radius, (n, 3))
    return pos


def _basin_hopping(positions, kind, rng, n_steps=30, step_size=0.5, max_inner_iter=200):
    """
    Basin-hopping algorithm: alternate between random perturbations and local minimization.
    This is a global optimization method that explores the landscape by making random moves
    and accepting them if they lead to a lower energy local minimum.
    """
    current_energy, current_pos = _polish(positions, kind, max_iter=max_inner_iter)
    best_energy = current_energy
    best_pos = current_pos.copy()

    for step in range(n_steps):
        if _time_left() < 30.0:
            break

        # Random perturbation
        noise = rng.normal(0, step_size, positions.shape)
        perturbed = current_pos + noise

        if not _is_valid(perturbed):
            # Try smaller perturbation
            noise = rng.normal(0, step_size * 0.5, positions.shape)
            perturbed = current_pos + noise
            if not _is_valid(perturbed):
                continue

        # Local minimization
        new_energy, new_pos = _polish(perturbed, kind, max_iter=max_inner_iter)

        if not np.isfinite(new_energy):
            continue

        # Metropolis acceptance criterion (always accept if better, accept worse with prob)
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        else:
            # Accept with probability exp(-(new_energy - current_energy) / T)
            # Use a simple temperature that decreases over time
            T = 0.1 * (1.0 - step / n_steps) + 0.01
            if rng.random() < np.exp(-(new_energy - current_energy) / T):
                current_energy = new_energy
                current_pos = new_pos

    return best_energy, best_pos


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]

    best_energy = np.inf
    best_pos = None

    rng = np.random.default_rng(42 + n)

    # Generate diverse initial templates
    templates = []
    for gen_func in [_generate_fcc, _generate_bcc, _generate_icosahedron]:
        try:
            base = gen_func(n)
            if _is_valid(base):
                templates.append(base)
        except Exception:
            pass

    # Add random templates
    for i in range(5):
        pos = _generate_random_compact(n, rng, radius_scale=1.0 + 0.2 * i)
        if _is_valid(pos):
            templates.append(pos)

    # First pass: polish all templates to find good local minima
    polished_templates = []
    for template in templates:
        if _time_left() < 40.0:
            break
        try:
            e, pos = _polish(template, kind, max_iter=500)
            if np.isfinite(e) and _is_valid(pos):
                polished_templates.append((e, pos))
                if e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
        except Exception:
            continue

    # Second pass: basin-hopping from the best polished templates
    # Sort by energy and take the top ones
    polished_templates.sort(key=lambda x: x[0])
    top_templates = polished_templates[:min(6, len(polished_templates))]

    for e_init, pos_init in top_templates:
        if _time_left() < 45.0:
            break
        try:
            bh_energy, bh_pos = _basin_hopping(
                pos_init, kind, rng,
                n_steps=40,
                step_size=0.6,
                max_inner_iter=150
            )
            if np.isfinite(bh_energy) and _is_valid(bh_pos):
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
        except Exception:
            continue

    # Third pass: additional basin-hopping with larger steps for more exploration
    if _time_left() > 50.0:
        for pos_init in top_templates[:3]:
            if _time_left() < 45.0:
                break
            try:
                bh_energy, bh_pos = _basin_hopping(
                    pos_init, kind, rng,
                    n_steps=50,
                    step_size=1.0,
                    max_inner_iter=100
                )
                if np.isfinite(bh_energy) and _is_valid(bh_pos):
                    if bh_energy < best_energy:
                        best_energy = bh_energy
                        best_pos = bh_pos.copy()
            except Exception:
                continue

    # Fourth pass: final polish of the best result
    if best_pos is not None and _time_left() > 20.0:
        try:
            final_energy, final_pos = _polish(best_pos, kind, max_iter=500, tol=1e-12)
            if np.isfinite(final_energy) and _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pos = final_pos.copy()
        except Exception:
            pass

    # Fallback if nothing worked
    if best_pos is None:
        best_pos = _generate_fcc(n)

    return best_pos

# EVOLVE-BLOCK-END
