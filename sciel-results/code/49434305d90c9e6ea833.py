"""Lennard-Jones cluster N=55: Mackay icosahedron + liquid quench with polish."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _lj_energy_fast(pos):
    """Vectorized LJ energy: 4*sum(r^-12 - r^-6) over all pairs."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 * r6
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    return 0.5 * np.sum(pair_energy)

def _min_dist(pos):
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, np.inf)
    return np.min(r)

def _try_relax(pos):
    try:
        import science_chemistry
        relaxed = science_chemistry.relax(pos.copy())
        if relaxed is not None and relaxed.shape == pos.shape:
            if _min_dist(relaxed) >= 0.49:  # small tolerance
                return relaxed
    except Exception:
        pass
    return None

def _construct_mackay_icosahedron():
    """Construct Mackay icosahedron N=55 with icosahedral symmetry.
    
    Shells:
    - 1 atom at center
    - 12 atoms at icosahedron vertices (radius r1)
    - 42 atoms at second shell (radius r2): 12 on 5-fold axes + 30 on 2-fold axes
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices (unit vectors)
    ico_verts = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis]
    
    # Optimal radii for LJ Mackay icosahedron (approximate, will be refined by relaxation)
    r1 = 1.06
    r2 = 1.92
    
    # Shell 0: center
    shell0 = np.array([[0.0, 0.0, 0.0]])
    
    # Shell 1: 12 icosahedron vertices
    shell1 = ico_verts * r1
    
    # Shell 2: 42 atoms
    # Group A: 12 atoms along 5-fold axes (icosahedron vertex directions) at radius r2
    shell2_A = ico_verts * r2
    
    # Group B: 30 atoms along 2-fold axes (edge midpoint directions) at radius r2
    # Find edges: two vertices are connected if angle between them is ~138.19° (cos = -1/phi)
    dot_mat = ico_verts @ ico_verts.T
    target_dot = -1.0 / phi  # ≈ -0.618034
    
    edge_dir_set = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dot_mat[i, j] - target_dot) < 0.02:
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-10:
                    mid_dir = mid / mid_norm
                    edge_dir_set.append(mid_dir)
    
    edge_dirs = np.array(edge_dir_set)
    # Should have 15 unique edge midpoint directions
    # Add their negatives to get 30 directions
    all_2fold_dirs = np.vstack([edge_dirs, -edge_dirs])
    
    # Deduplicate (in case of numerical issues)
    unique_dirs = []
    for d in all_2fold_dirs:
        is_dup = False
        for u in unique_dirs:
            if np.dot(d, u) > 0.999:
                is_dup = True
                break
        if not is_dup:
            unique_dirs.append(d)
    
    unique_dirs = np.array(unique_dirs)
    # Take first 30
    if len(unique_dirs) >= 30:
        shell2_B = unique_dirs[:30] * r2
    else:
        # Fallback: generate more directions
        shell2_B = np.zeros((30, 3))
        for k in range(30):
            # Use remaining icosahedral directions or random
            angle = k * 2.0 * math.pi / 30.0
            shell2_B[k] = [math.cos(angle), math.sin(angle), 0.0] * r2
    
    all_atoms = np.vstack([shell0, shell1, shell2_A, shell2_B])
    
    if all_atoms.shape[0] != N:
        if all_atoms.shape[0] > N:
            all_atoms = all_atoms[:N]
        else:
            pad = np.random.randn(N - all_atoms.shape[0], 3) * 0.1
            all_atoms = np.vstack([all_atoms, pad])
    
    return all_atoms

def _random_liquid_init(n, radius=2.0, seed=42):
    """Generate random configuration with min distance 0.5."""
    rng = np.random.RandomState(seed)
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        # Rejection sampling in sphere
        while True:
            p = rng.randn(3) * radius
            if np.linalg.norm(p) <= radius:
                break
        if len(points) == 0:
            points.append(p)
        else:
            arr = np.array(points)
            dists = np.linalg.norm(arr - p, axis=1)
            if np.min(dists) >= 0.5:
                points.append(p)
    if len(points) < n:
        # Fallback: place remaining randomly with small offset
        for _ in range(n - len(points)):
            p = rng.randn(3) * 0.5
            points.append(p)
    return np.array(points)

def _liquid_quench(n_restarts, deadline, seed_base=100):
    """Multiple random restarts with relaxation. Returns best (pos, energy)."""
    best_pos = None
    best_energy = np.inf
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        seed = seed_base + restart
        pos = _random_liquid_init(N, radius=2.0, seed=seed)
        relaxed = _try_relax(pos)
        if relaxed is not None:
            e = _lj_energy_fast(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    return best_pos, best_energy

def _polish(pos, deadline):
    """Multiple relaxation passes for final polish."""
    try:
        import science_chemistry
        current = pos.copy()
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            new_pos = science_chemistry.relax(current.copy())
            if new_pos is not None and new_pos.shape == current.shape:
                if _min_dist(new_pos) >= 0.49:
                    current = new_pos
        return current
    except Exception:
        return pos

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin
    
    # Strategy 1: Analytical Mackay icosahedron
    mackay_pos = _construct_mackay_icosahedron()
    mackay_relaxed = _try_relax(mackay_pos)
    if mackay_relaxed is not None:
        best_pos = mackay_relaxed
        best_energy = _lj_energy_fast(mackay_relaxed)
    else:
        best_pos = mackay_pos
        best_energy = _lj_energy_fast(mackay_pos)
    
    # Strategy 2: Liquid quench restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - 25:
        n_restarts = min(8, int((deadline - 25 - elapsed) / 5) + 1)
        n_restarts = max(1, n_restarts)
        lq_pos, lq_energy = _liquid_quench(n_restarts, deadline)
        if lq_pos is not None and lq_energy < best_energy:
            best_pos = lq_pos
            best_energy = lq_energy
    
    # Final polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 8:
        polished = _polish(best_pos, deadline)
        if polished is not None:
            pe = _lj_energy_fast(polished)
            if pe < best_energy:
                best_pos = polished
                best_energy = pe
    
    # Validate
    if _min_dist(best_pos) < 0.5:
        # Try mackay fallback
        if _min_dist(mackay_pos) >= 0.5:
            best_pos = mackay_pos
        else:
            # Last resort: simple valid configuration
            side = int(np.ceil(N ** (1/3)))
            pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
            best_pos = np.array(pts[:N], dtype=float) * 1.12
    
    return best_pos
# EVOLVE-BLOCK-END
