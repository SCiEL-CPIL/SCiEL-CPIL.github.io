import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
from scipy.optimize import minimize

def _get_icosahedron_vertices(radius=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_dodecahedron_vertices(radius=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _generate_mackay_55():
    """Perfect 55-atom Mackay icosahedron: 1 + 12 + 42."""
    r1 = 1.0
    r2 = 1.87  # near the Dzugutov equilibrium distance
    
    points = [np.zeros(3)]
    
    # First shell: 12 icosahedron vertices
    ico1 = _get_icosahedron_vertices(r1)
    for v in ico1:
        points.append(v.copy())
    
    # Second shell: 42 atoms = 12 vertices + 30 edge midpoints of icosahedron
    base = _get_icosahedron_vertices(1.0)
    # Find the 30 edges (nearest-neighbor pairs)
    min_d = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(base[i] - base[j])
            if d < min_d:
                min_d = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(np.linalg.norm(base[i] - base[j]), min_d, atol=1e-6):
                edges.append((i, j))
    
    shell2 = []
    for v in base:
        shell2.append(v.copy())
    for i, j in edges:
        mid = (base[i] + base[j]) / 2.0
        shell2.append(mid)
    
    shell2 = np.array(shell2)
    norms = np.linalg.norm(shell2, axis=1)
    shell2 = shell2 / norms[:, np.newaxis] * r2
    for v in shell2:
        points.append(v)
    
    return np.array(points[:55])

def _defect_seed_55_to_n(n, variant=0):
    """
    Create an n-atom seed from the 55-atom Mackay icosahedron by removing atoms.
    Different variants remove different sets of atoms to create diversity.
    """
    mackay = _generate_mackay_55()
    if n >= 55:
        if n == 55:
            return mackay
        # For 57, 60: add atoms from a third shell (dodecahedron-like)
        result = mackay.copy()
        remaining = n - 55
        # Add atoms at face-center-like positions of the icosahedron, scaled out
        ico1 = _get_icosahedron_vertices(1.0)
        # Face centers: for each face (triangle) of the icosahedron, take centroid
        # The icosahedron has 20 faces
        # Approximate: use dodecahedron vertices at a larger radius
        dod = _get_dodecahedron_vertices(2.6)
        added = 0
        for v in dod:
            if added >= remaining:
                break
            # Check min distance
            dists = np.linalg.norm(result - v, axis=1)
            if np.min(dists) > 0.5:
                result = np.vstack([result, v])
                added += 1
        # If still not enough, add more
        while len(result) < n:
            v = dod[added % len(dod)] * (1.0 + 0.1 * (added // len(dod)))
            result = np.vstack([result, v])
            added += 1
        return result[:n]
    
    # n < 55: remove atoms from the Mackay structure
    # Strategy: remove from the outer shell (42 atoms) first, keeping the 13-core
    # The 13-core (1+12) is very stable
    core = mackay[:13]
    outer = mackay[13:]  # 42 atoms
    
    need_outer = n - 13
    if need_outer <= 0:
        return core[:n]
    
    # Sort outer atoms by distance from center (remove farthest first for variety)
    # Or by index for deterministic removal
    if variant == 0:
        # Remove last atoms (arbitrary but deterministic)
        selected_outer = outer[:need_outer]
    elif variant == 1:
        # Remove atoms with highest index (different set)
        selected_outer = outer[-need_outer:]
    elif variant == 2:
        # Remove every other atom
        indices = np.arange(42)
        if need_outer % 2 == 0:
            selected_outer = outer[indices[::2][:need_outer]]
        else:
            selected_outer = np.vstack([outer[indices[::2][:need_outer//2]], outer[indices[1::2][:need_outer//2 + 1]]])
    else:
        # Random removal with seed
        rng = np.random.default_rng(variant * 1000 + n)
        perm = rng.permutation(42)
        selected_outer = outer[perm[:need_outer]]
    
    result = np.vstack([core, selected_outer])
    return result[:n]

def _l_bfgs_minimize(positions, potential, maxiter=200, tol=1e-8):
    """Use L-BFGS-B with the Dzugutov gradient for fast local minimization."""
    n = positions.shape[0]
    x0 = positions.flatten()
    
    def energy_and_grad(x):
        pos = x.reshape(n, 3)
        # Ensure minimum distance constraint is handled softly
        try:
            e, g = dzugutov_energy_gradient(pos)
            return e, g.flatten()
        except Exception:
            # Fallback: numerical gradient
            eps = 1e-5
            e0 = _dzugutov_energy(pos)
            grad = np.zeros_like(pos)
            for i in range(n):
                for j in range(3):
                    pos_p = pos.copy()
                    pos_p[i, j] += eps
                    pos_m = pos.copy()
                    pos_m[i, j] -= eps
                    grad[i, j] = (_dzugutov_energy(pos_p) - _dzugutov_energy(pos_m)) / (2*eps)
            return e0, grad.flatten()
    
    try:
        res = minimize(energy_and_grad, x0, method='L-BFGS-B',
                       jac=True, options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-6, 'maxfun': 5000})
        if res.success or res.fun < float('inf'):
            return res.fun, res.x.reshape(n, 3)
    except Exception:
        pass
    
    # Fallback to relax
    try:
        e, pos = relax(positions, potential, None)
        return e, pos
    except Exception:
        return float('inf'), positions

def _dzugutov_energy(pos):
    """Compute Dzugutov energy (for fallback use)."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-8:
                continue
            if r < a:
                E += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E

def _perturb_and_relax(positions, potential, seed, amp=0.05, maxiter=150):
    """Perturb positions and re-minimize."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    perturbed = positions + rng.normal(0, amp, positions.shape)
    return _l_bfgs_minimize(perturbed, potential, maxiter=maxiter)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    best_energy = float('inf')
    best_positions = None
    
    def try_candidate(pos, maxiter=200):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        try:
            e, pos = _l_bfgs_minimize(pos, potential, maxiter=maxiter)
            if e < best_energy and np.all(np.isfinite(pos)):
                best_energy = e
                best_positions = pos
        except Exception:
            pass
    
    # Generate diverse defect-seeded candidates
    # Variant 0: standard removal
    # Variant 1: different removal pattern
    # Variant 2: alternating removal
    # Variant 3,4,5: random removals with different seeds
    
    n_variants = 6
    for v in range(n_variants):
        if time.time() > deadline:
            break
        try:
            seed_pos = _defect_seed_55_to_n(size, variant=v)
            if seed_pos is not None and len(seed_pos) == size:
                try_candidate(seed_pos, maxiter=200)
        except Exception:
            pass
    
    # Additional: slightly distorted icosahedral seeds
    if time.time() < deadline - 30:
        for v in range(3):
            if time.time() > deadline:
                break
            try:
                base_seed = _defect_seed_55_to_n(size, variant=v)
                if base_seed is not None and len(base_seed) == size:
                    rng = np.random.default_rng(1000 + v * 100 + size)
                    distorted = base_seed + rng.normal(0, 0.08, base_seed.shape)
                    try_candidate(distorted, maxiter=150)
            except Exception:
                pass
    
    # Perturbation + re-relax on best found
    if best_positions is not None and time.time() < deadline - 20:
        for attempt in range(4):
            if time.time() > deadline:
                break
            try:
                e, pos = _perturb_and_relax(best_positions, potential, seed=2000 + attempt * 10 + size, amp=0.06, maxiter=150)
                if e < best_energy and np.all(np.isfinite(pos)):
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
    
    # Final fallback: use relax on the best seed
    if best_positions is None or time.time() < deadline - 10:
        try:
            seed_pos = _defect_seed_55_to_n(size, variant=0)
            if seed_pos is not None:
                e, pos = relax(seed_pos, potential, None)
                if e < best_energy and np.all(np.isfinite(pos)):
                    best_energy = e
                    best_positions = pos
        except Exception:
            pass
    
    if best_positions is None:
        # Absolute fallback
        best_positions = _defect_seed_55_to_n(size, variant=0)
    
    return best_positions
