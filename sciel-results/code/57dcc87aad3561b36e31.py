"""Program for Sutton-Chen TM cluster global minima search.
Combines growth-based construction with symmetry-constrained parameterization
(radial shells) to reduce dimensionality and improve global search efficiency.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


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


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    a = 1.0 / np.sqrt(2.0)
    c = a * np.sqrt(8.0 / 3.0)
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                x = a * (i + j / 2.0)
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                candidates.append((np.sqrt(x*x + y*y + z*z), np.array([x, y, z])))
                x2 = a * (i + j / 2.0 + 0.5)
                y2 = a * np.sqrt(3.0) / 2.0 * (j + 1.0 / np.sqrt(3.0))
                z2 = c * (k + 1.0 / 3.0)
                candidates.append((np.sqrt(x2*x2 + y2*y2 + z2*z2), np.array([x2, y2, z2])))
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


def _polish(positions, kind, max_iter=500, tol=1e-10):
    """High-precision polish pass using gradient-based optimization."""
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        e, _ = sutton_chen_energy_gradient(pos, kind)
        return e

    def gradient_fn(x):
        pos = x.reshape(-1, 3)
        _, g = sutton_chen_energy_gradient(pos, kind)
        return g.ravel()

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
    except Exception:
        pass
    return np.inf, positions


def _growth_construct(n, kind, rng):
    """
    Deterministic growth algorithm: build cluster by adding atoms one at a time
    to the lowest-energy interstitial site of the (N-1)-atom cluster.
    """
    if n <= 1:
        return np.zeros((n, 3))

    best_energy = np.inf
    best_pos = None

    for d0 in [0.9, 1.0, 1.1, 1.2]:
        pos = np.array([[0, 0, 0], [d0, 0, 0]])
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue

    if best_pos is None:
        best_pos = np.array([[0, 0, 0], [1.0, 0, 0]])

    current = best_pos.copy()

    for target_n in range(3, n + 1):
        if _time_left() < 40.0:
            break

        candidates = []

        for i in range(len(current)):
            for trial in range(6):
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                for dist in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    new_pos = current[i] + direction * dist
                    candidates.append(new_pos)

        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                midpoint = (current[i] + current[j]) / 2
                bond_vec = current[j] - current[i]
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.1:
                    perp = np.cross(bond_vec, [0, 0, 1])
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(bond_vec, [1, 0, 0])
                    perp /= np.linalg.norm(perp)
                    for offset in [0.3, 0.5, 0.7]:
                        candidates.append(midpoint + perp * offset)
                        candidates.append(midpoint - perp * offset)

        best_e = np.inf
        best_p = None

        rng.shuffle(candidates)
        for candidate in candidates[:12]:
            if _time_left() < 35.0:
                break
            test_pos = np.vstack([current, candidate.reshape(1, 3)])
            try:
                energy, relaxed = relax(test_pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_e:
                        best_e = energy
                        best_p = relaxed
            except Exception:
                continue

        if best_p is not None:
            current = best_p
        else:
            new_atom = current[0] + rng.normal(0, 0.5, 3)
            current = np.vstack([current, new_atom.reshape(1, 3)])

    return current


def _symmetry_constrained_search(n, kind, rng, n_shells=6, max_iter=300):
    """
    Structural change: parameterize the cluster using radial shells with
    symmetry-constrained angular offsets. This reduces the effective dimension
    from 3N to ~3*n_shells + small angular perturbations, forcing the optimizer
    to explore only physically reasonable compact structures.

    The parameterization:
    - n_shells radial distances r_1 < r_2 < ... < r_k (with r_1 >= 0.35)
    - Each atom is assigned to a shell and given a small angular offset
    - The offsets are constrained to stay within a compact region

    This exploits the fact that TM cluster global minima are typically
    compact, nearly-spherical structures with atoms arranged in concentric
    shells (like Mackay icosahedra, FCC fragments, etc.).
    """
    n_atoms = n

    # Estimate typical cluster radius
    r_typical = 0.55 * n_atoms ** (1.0 / 3.0) + 0.5

    # Number of shells to use (adaptive)
    n_shells_eff = min(n_shells, max(2, n_atoms // 4))

    def _decode(params):
        """Decode parameter vector into (n, 3) positions.

        params layout:
        - [0:n_shells_eff] : shell radii r_1..r_k (sorted ascending)
        - [n_shells_eff : n_shells_eff + n_atoms*3] : angular offsets for each atom
        """
        radii = params[:n_shells_eff]
        # Enforce minimum separation and ordering
        radii = np.maximum(radii, 0.36)
        # Sort radii to enforce shell ordering
        radii_sorted = np.sort(radii)
        # Ensure monotonic: r_i >= r_{i-1} + 0.35
        for i in range(1, n_shells_eff):
            radii_sorted[i] = max(radii_sorted[i], radii_sorted[i-1] + 0.35)

        offsets = params[n_shells_eff:].reshape(n_atoms, 3)

        # Assign atoms to shells: first few to inner shell, rest distributed
        # Use a simple scheme: atom i goes to shell (i * n_shells_eff) // n_atoms
        positions = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            shell_idx = min(int(i * n_shells_eff / n_atoms), n_shells_eff - 1)
            base_r = radii_sorted[shell_idx]
            # Add angular offset (clipped to keep atoms near the shell)
            offset = offsets[i]
            # Clip offset magnitude to avoid atoms flying too far
            om = np.linalg.norm(offset)
            if om > 0.5:
                offset = offset * (0.5 / om)
            pos = np.array([0.0, 0.0, base_r]) + offset
            # Ensure minimum distance from origin
            pos_norm = np.linalg.norm(pos)
            if pos_norm < 0.36:
                if pos_norm < 1e-8:
                    pos = np.array([0.36, 0.0, 0.0])
                else:
                    pos = pos * (0.36 / pos_norm)
            positions[i] = pos
        return positions

    def _energy_from_params(params):
        pos = _decode(params)
        if not _is_valid(pos):
            return 1e10
        try:
            e, _ = sutton_chen_energy_gradient(pos, kind)
            return e if np.isfinite(e) else 1e10
        except Exception:
            return 1e10

    # Initial parameter guess based on a compact spherical arrangement
    best_e = np.inf
    best_params = None

    n_trials = 8
    for trial in range(n_trials):
        if _time_left() < 20.0:
            break

        # Initialize radii as a roughly uniform shell spacing
        r_min = 0.4
        r_max = r_typical * (1.0 + 0.3 * trial / n_trials)
        radii0 = np.linspace(r_min, r_max, n_shells_eff)
        # Add small random perturbation
        radii0 += rng.normal(0, 0.05, n_shells_eff)
        radii0 = np.sort(np.maximum(radii0, 0.36))

        # Initialize offsets as small random perturbations
        offsets0 = rng.normal(0, 0.15, (n_atoms, 3))

        params0 = np.concatenate([radii0, offsets0.ravel()])

        result = minimize(
            _energy_from_params,
            params0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 40}
        )

        if np.isfinite(result.fun) and result.fun < best_e:
            best_e = result.fun
            best_params = result.x.copy()

    if best_params is None:
        return np.inf, None

    best_pos = _decode(best_params)
    if not _is_valid(best_pos):
        return np.inf, None

    # Final polish in full 3N space
    try:
        polished_e, polished_pos = _polish(best_pos, kind, max_iter=200, tol=1e-12)
        if polished_e < best_e:
            best_e = polished_e
            best_pos = polished_pos
    except Exception:
        pass

    return best_e, best_pos


def _multi_start_refine(n, kind, rng, positions_list, max_relax=15):
    """
    Refinement machinery: take a list of candidate positions, relax each,
    keep the best. This is the exploitation component.
    """
    best_e = np.inf
    best_pos = None

    for pos in positions_list[:max_relax]:
        if _time_left() < 15.0:
            break
        if not _is_valid(pos):
            continue
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_e:
                    best_e = energy
                    best_pos = relaxed.copy()
        except Exception:
            continue

    return best_e, best_pos


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]

    best_energy = np.inf
    best_pos = None

    rng = np.random.default_rng(42 + n)

    # Strategy 1: Deterministic growth (exploitation of hierarchical structure)
    try:
        growth_pos = _growth_construct(n, kind, rng)
        if _is_valid(growth_pos):
            energy, relaxed = relax(growth_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                best_energy = energy
                best_pos = relaxed
    except Exception:
        pass

    # Strategy 2: Symmetry-constrained radial shell search (structural change)
    # This reduces dimensionality and forces exploration of compact structures
    if _time_left() > 50.0:
        try:
            sc_energy, sc_pos = _symmetry_constrained_search(
                n, kind, rng, n_shells=7, max_iter=250
            )
            if np.isfinite(sc_energy) and sc_pos is not None and _is_valid(sc_pos):
                if sc_energy < best_energy:
                    best_energy = sc_energy
                    best_pos = sc_pos
        except Exception:
            pass

    # Strategy 3: FCC/BCC/HCP/Mackay templates with relaxation (diverse basins)
    for gen_func in [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedron]:
        if _time_left() < 25.0:
            break
        base = gen_func(n)
        if not _is_valid(base):
            continue
        for noise_scale in [0.0, 0.05, 0.1, 0.2]:
            if _time_left() < 20.0:
                break
            noise = rng.normal(0, noise_scale, base.shape) if noise_scale > 0 else np.zeros_like(base)
            pos = base + noise
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                if not np.isfinite(energy):
                    continue
                if _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue

    # Strategy 4: Differential Evolution for global exploration
    if _time_left() > 60.0:
        try:
            if best_pos is not None:
                de_seed = best_pos.copy()
            else:
                de_seed = _generate_fcc(n)

            # Use scipy DE with a bounded box
            n_atoms = n
            radius = 1.5 + 0.5 * (n_atoms / 50.0)
            bounds = [(-radius, radius)] * (n_atoms * 3)

            def de_energy(x):
                pos = x.reshape(-1, 3)
                if not _is_valid(pos):
                    return 1e10
                try:
                    e, _ = sutton_chen_energy_gradient(pos, kind)
                    return e if np.isfinite(e) else 1e10
                except Exception:
                    return 1e10

            de_result = differential_evolution(
                de_energy,
                bounds,
                x0=de_seed.ravel(),
                maxiter=30,
                popsize=8,
                tol=1e-8,
                seed=rng,
                polish=False,
                workers=1
            )

            if de_result.success and np.isfinite(de_result.fun):
                de_pos = de_result.x.reshape(-1, 3)
                if _is_valid(de_pos):
                    # Polish the DE result
                    try:
                        p_e, p_pos = _polish(de_pos, kind, max_iter=200, tol=1e-12)
                        if p_e < de_result.fun:
                            de_result.fun = p_e
                            de_pos = p_pos
                    except Exception:
                        pass
                    if de_result.fun < best_energy:
                        best_energy = de_result.fun
                        best_pos = de_pos
        except Exception:
            pass

    # Strategy 5: Random restarts if time remains
    if _time_left() > 35.0:
        for i in range(20):
            if _time_left() < 20.0:
                break
            radius_scale = 1.0 + 0.2 * (i % 3)
            pos = _generate_random_compact(n, rng, radius_scale)
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                if not np.isfinite(energy):
                    continue
                if _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue

    # Strategy 6: Final high-precision polish
    if best_pos is not None and _time_left() > 15.0:
        try:
            polished_energy, polished_pos = _polish(best_pos, kind, max_iter=300, tol=1e-12)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
        except Exception:
            pass

    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)

    return best_pos

# EVOLVE-BLOCK-END
