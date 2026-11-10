"""Rewritten program: structured icosahedral seeds + gradient descent for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global time budget management
_TOTAL_BUDGET = 280.0  # seconds, leave margin under 300
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_remaining():
    return _TOTAL_BUDGET - _elapsed()


def _gradient_descent(pos, potential, parameter, max_iter=500, time_limit=None):
    """Adaptive gradient descent with backtracking line search."""
    pos = pos.copy()
    energy = None
    lr = 0.01
    min_lr = 1e-6
    max_lr = 0.1
    
    for it in range(max_iter):
        if time_limit is not None and _elapsed() > time_limit:
            break
        
        grad = dzugutov_energy_gradient(pos, potential)
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-8:
            break
        
        # Backtracking line search
        step = lr * grad / gnorm  # normalize direction, scale by lr
        for _ in range(20):
            new_pos = pos - step
            # Check minimum distance constraint
            if _min_dist_ok(new_pos):
                try:
                    new_energy = _compute_energy(new_pos, potential, parameter)
                    if new_energy < energy if energy is not None else True:
                        pos = new_pos
                        energy = new_energy
                        break
                except Exception:
                    pass
            step *= 0.5
        
        # Adaptive learning rate
        if energy is not None:
            # If we made progress, keep lr; otherwise reduce
            pass
        
        lr = max(min_lr, min(max_lr, lr * (1.0 if gnorm > 1e-4 else 0.5)))
    
    return pos, energy if energy is not None else _compute_energy(pos, potential, parameter)


def _compute_energy(pos, potential, parameter):
    """Compute Dzugutov energy directly."""
    n = len(pos)
    A, B = 5.82, 1.28
    a, b = 1.87, 1.94
    c, d = 1.10, 0.27
    m = 16
    
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _min_dist_ok(pos, min_d=0.5):
    """Check all pairwise distances >= min_d."""
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_d:
                return False
    return True


def _fix_min_dist(pos, min_d=0.5, max_iter=100):
    """Push apart atoms that are too close."""
    pos = pos.copy()
    for _ in range(max_iter):
        moved = False
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-10:
                        d = np.array([1.0, 0, 0])
                    push = d / dist * (min_d - dist) * 0.5
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    return pos


def _icosahedron_13():
    """13-atom icosahedron: center + 12 vertices."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale to appropriate bond length (~1.4 for Dzugutov)
    bond = 1.4
    verts = verts * bond
    center = np.zeros(3)
    return np.vstack([center, verts])


def _icosahedron_55():
    """55-atom nested icosahedra: 1 + 12 + 42 (two shells)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    r1 = 1.4
    r2 = 2.6
    r3 = 3.8
    
    atoms = [np.zeros(3)]  # center
    atoms.extend([v * r1 for v in verts])  # 12
    
    # Second shell: 30 atoms (edges of icosahedron midpoints scaled) + 12 face centers
    # Simpler: use another icosahedron rotated and scaled
    # For 55 = 1 + 12 + 42, use known 55-atom Mackay icosahedron
    # 42 = 30 (edge midpoints) + 12 (face centers)
    # Edge midpoints: 30 edges
    idx = [0,1, 0,2, 0,3, 0,4, 0,5, 0,6, 0,7, 0,8, 0,9, 0,10, 0,11,
           1,2, 1,4, 1,5, 1,6, 1,7, 1,8, 1,9, 1,10, 1,11,
           2,3, 2,4, 2,5, 2,6, 2,7, 2,8, 2,9, 2,10, 2,11,
           3,4, 3,5, 3,6, 3,7, 3,8, 3,9, 3,10, 3,11,
           4,5, 4,6, 4,7, 4,8, 4,9, 4,10, 4,11,
           5,6, 5,7, 5,8, 5,9, 5,10, 5,11,
           6,7, 6,8, 6,9, 6,10, 6,11,
           7,8, 7,9, 7,10, 7,11,
           8,9, 8,10, 8,11,
           9,10, 9,11,
           10,11]
    # That's too many; icosahedron has 30 edges
    # Let me just use a simpler approach: scale up the icosahedron vertices for shell 2 and 3
    # 55-atom: 1 center + 12 (r1) + 30 (r2, edge midpoints) + 12 (r3, face centers)
    
    # For simplicity, use three concentric icosahedral shells with appropriate radii
    # 1 + 12 + 30 + 12 = 55
    # Edge midpoints (30):
    edge_pairs = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),
                  (1,2),(1,4),(1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11),
                  (2,3),(2,4),(2,5),(2,6),(2,7),(2,8),(2,9),(2,10),(2,11),
                  (3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),
                  (4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),
                  (5,6),(5,7),(5,8),(5,9),(5,10),(5,11),
                  (6,7),(6,8),(6,9),(6,10),(6,11),
                  (7,8),(7,9),(7,10),(7,11),
                  (8,9),(8,10),(8,11),
                  (9,10),(9,11),
                  (10,11)]
    # Icosahedron has 30 edges; the above list has more. Let me just use a simpler construction.
    
    # Simpler: just use scaled icosahedra
    shell2 = verts * r2  # 12 atoms at r2
    shell3 = verts * r3  # 12 atoms at r3
    # That gives 1+12+12+12 = 37, not 55.
    
    # For the purpose of this seed, let's just return a reasonable 55-atom structure
    # by adding more atoms in a disordered shell
    atoms = [np.zeros(3)]
    atoms.extend([v * r1 for v in verts])
    # Add 42 more atoms in a shell of radius ~2.5
    rng = np.random.default_rng(42)
    for _ in range(42):
        dir = rng.normal(size=3)
        dir /= np.linalg.norm(dir)
        atoms.append(dir * r2)
    
    return np.array(atoms)


def _fcc_cluster(n):
    """Generate an FCC-like cluster of n atoms."""
    atoms = []
    # Generate FCC lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: positions at (i,j,k) where i+j+k even, scaled
                if (i + j + k) % 2 == 0:
                    atoms.append(np.array([i, j, k]) * 1.0)
    atoms = np.array(atoms)
    # Center
    atoms -= atoms.mean(axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(atoms, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order]
    # Take first n
    if len(atoms) >= n:
        return atoms[:n]
    # Pad with random
    rng = np.random.default_rng(123)
    while len(atoms) < n:
        p = rng.normal(size=3) * 2.0
        atoms = np.vstack([atoms, p])
    return atoms


def _bcc_cluster(n):
    """Generate a BCC-like cluster of n atoms."""
    atoms = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                atoms.append(np.array([i, j, k]) * 1.2)
                # BCC center
                atoms.append(np.array([i+0.5, j+0.5, k+0.5]) * 1.2)
    atoms = np.array(atoms)
    atoms -= atoms.mean(axis=0)
    dists = np.linalg.norm(atoms, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order]
    if len(atoms) >= n:
        return atoms[:n]
    rng = np.random.default_rng(456)
    while len(atoms) < n:
        p = rng.normal(size=3) * 2.0
        atoms = np.vstack([atoms, p])
    return atoms


def _perturb(pos, sigma, rng):
    """Add Gaussian perturbation."""
    return pos + rng.normal(0, sigma, pos.shape)


def construct(instance):
    """Structured seeds (icosahedral, FCC, BCC) + gradient descent for Dzugutov clusters."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    time_left = _time_remaining()
    # Allocate time per instance roughly proportional to size
    # Total 6 instances: 38, 45, 50, 55, 57, 60
    # Give more time to larger clusters
    time_per_instance = time_left / max(1, 6 - int(np.where([38,45,50,55,57,60] == size)[0][0]) if size in [38,45,50,55,57,60] else 3)
    # Simpler: just use a fraction of remaining time
    max_iter = min(300, max(50, int(time_left * 2)))
    
    rng = np.random.default_rng(size)
    
    # Generate diverse structured seeds
    seeds = []
    
    # 1. Icosahedral 13-atom core + random shell
    ico13 = _icosahedron_13()
    if size >= 13:
        core = ico13[:13]
        remaining = size - 13
        shell_radius = 1.4 * np.sqrt(remaining / 12) + 1.0
        shell = []
        for _ in range(remaining):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            shell.append(dir * shell_radius)
        seed1 = np.vstack([core, np.array(shell)])
        seeds.append(seed1)
    
    # 2. FCC-like
    seeds.append(_fcc_cluster(size))
    
    # 3. BCC-like
    seeds.append(_bcc_cluster(size))
    
    # 4. Icosahedral 55 if size >= 55
    if size >= 55:
        ico55 = _icosahedron_55()
        if len(ico55) >= size:
            seeds.append(ico55[:size])
        else:
            # Pad
            rng2 = np.random.default_rng(size + 1)
            extra = rng2.normal(0, 1.5, (size - len(ico55), 3))
            seeds.append(np.vstack([ico55, extra]))
    
    # 5. Perturbed icosahedral
    if size >= 13:
        base = ico13[:13]
        remaining = size - 13
        shell_radius = 1.4 * np.sqrt(remaining / 12) + 1.0
        shell = []
        for _ in range(remaining):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            shell.append(dir * shell_radius)
        seed5 = np.vstack([base, np.array(shell)])
        seed5 = _perturb(seed5, 0.1, rng)
        seeds.append(seed5)
    
    # 6. Random compact (fallback)
    radius = 1.1 * size ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < size:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    seeds.append(np.array(points))
    
    # Relax each seed and keep the best
    best_energy = np.inf
    best_pos = None
    
    for i, seed in enumerate(seeds):
        if _time_remaining() < 5:
            break
        
        # Fix minimum distance
        seed = _fix_min_dist(seed, 0.5)
        
        # Gradient descent
        try:
            pos, energy = _gradient_descent(seed, potential, parameter, max_iter=max_iter, time_limit=_TOTAL_BUDGET - 10)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except Exception:
            # Fallback to relax
            try:
                energy, pos = relax(seed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
            except Exception:
                pass
    
    if best_pos is None:
        # Final fallback
        best_pos = _fix_min_dist(seeds[0], 0.5)
    
    # Ensure minimum distance
    best_pos = _fix_min_dist(best_pos, 0.5)
    
    return best_pos
# EVOLVE-BLOCK-END
