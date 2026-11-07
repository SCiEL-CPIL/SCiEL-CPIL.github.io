"""Sutton-Chen cluster global minima: symmetry-locked I_h orbit optimization + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 295.0
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    return bool(np.all(d >= 0.35))

# ---- Icosahedral group I_h (120 rotations) ----
_PHI = (1.0 + np.sqrt(5.0)) / 2.0
_INV_PHI = 1.0 / _PHI

def _icosahedral_group():
    """Return list of 120 rotation matrices of the full icosahedral group I_h."""
    # 60 proper rotations of the icosahedron
    c = _INV_PHI
    s = 1.0
    # identity
    mats = [np.eye(3)]
    # 6 C5 rotations (about 6 5-fold axes through face centers)
    # 10 C3 rotations (about 10 3-fold axes through vertex pairs)
    # 10 C2 rotations (about 10 2-fold axes through edge midpoints)
    # 15 C2' rotations (about 15 2-fold axes through edge pairs)
    # We generate them systematically.
    # 5-fold axis: (1, phi, 0) normalized
    axis5 = np.array([1.0, _PHI, 0.0])
    axis5 /= np.linalg.norm(axis5)
    # 3-fold axis: (1, 1, 1) normalized
    axis3 = np.array([1.0, 1.0, 1.0])
    axis3 /= np.linalg.norm(axis3)
    # 2-fold axis: (1, 0, 0)
    axis2 = np.array([1.0, 0.0, 0.0])
    axis2b = np.array([0.0, 1.0, 0.0])
    axis2c = np.array([0.0, 0.0, 1.0])
    axis2d = np.array([1.0, 1.0, 0.0])
    axis2d /= np.linalg.norm(axis2d)
    axis2e = np.array([1.0, -1.0, 0.0])
    axis2e /= np.linalg.norm(axis2e)

    def _rotx(a):
        return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    def _roty(a):
        return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    def _rotz(a):
        return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])

    # C5 rotations about axis5: angles 2*pi/5 * k
    for k in range(5):
        a = 2.0 * np.pi * k / 5.0
        # rotate axis5 to z, rotate about z, rotate back
        # Simpler: use Rodrigues
        def _rodrigues(axis, angle):
            axis = axis / np.linalg.norm(axis)
            K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
            return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        mats.append(_rodrigues(axis5, a))

    # C3 rotations about axis3: angles 2*pi/3 * k
    for k in range(3):
        a = 2.0 * np.pi * k / 3.0
        mats.append(_rodrigues(axis3, a))

    # C2 rotations about axis2, axis2b, axis2c: angle pi
    for ax in [axis2, axis2b, axis2c]:
        mats.append(_rodrigues(ax, np.pi))

    # C2' rotations about diagonal 2-fold axes
    for ax in [axis2d, axis2e]:
        mats.append(_rodrigues(ax, np.pi))
    axis2f = np.array([0.0, 1.0, 1.0]); axis2f /= np.linalg.norm(axis2f)
    axis2g = np.array([0.0, 1.0, -1.0]); axis2g /= np.linalg.norm(axis2g)
    mats.append(_rodrigues(axis2f, np.pi))
    mats.append(_rodrigues(axis2g, np.pi))

    # Now generate the full 60 proper rotations by closure
    # Take the 15 generators and close under multiplication
    generators = list(mats)
    all_mats = [np.eye(3)]
    changed = True
    while changed:
        changed = False
        for m in list(all_mats):
            for g in generators:
                prod = m @ g
                # check if already in list
                found = False
                for existing in all_mats:
                    if np.allclose(prod, existing, atol=1e-10):
                        found = True
                        break
                if not found:
                    all_mats.append(prod)
                    changed = True
        # also try g @ m
        for g in generators:
            for m in list(all_mats):
                prod = g @ m
                found = False
                for existing in all_mats:
                    if np.allclose(prod, existing, atol=1e-10):
                        found = True
                        break
                if not found:
                    all_mats.append(prod)
                    changed = True

    # Add inversion and all -M
    for m in list(all_mats):
        all_mats.append(-m)

    # Deduplicate
    unique = []
    for m in all_mats:
        found = False
        for u in unique:
            if np.allclose(m, u, atol=1e-10):
                found = True
                break
        if not found:
            unique.append(m)

    return unique

_IH_MATS = None
def _get_ih():
    global _IH_MATS
    if _IH_MATS is None:
        _IH_MATS = _icosahedral_group()
    return _IH_MATS

# ---- Orbit generators for I_h ----
def _orbit_12():
    """12 vertices of icosahedron (5-fold axis vertices)."""
    phi = _PHI
    pts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    return pts

def _orbit_6():
    """6 face centers (5-fold axes)."""
    phi = _PHI
    pts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi]
    ], dtype=float)
    # Actually face centers are different; use the 6 directions of 5-fold axes
    # The 6 5-fold axes point to face centers of the icosahedron
    # They are at: (0, ±1, ±phi), (±phi, 0, ±1), (±1, ±phi, 0) -- no that's 12
    # 6 axes = 12 directions. The 6 axes are the 6 pairs of opposite faces.
    # Let me just use the 12 vertex directions and pick 6 that form axes
    # Actually for I_h, the 6 5-fold axes go through face centers.
    # Face centers of icosahedron:
    # The 20 faces of icosahedron have centers at specific positions.
    # Let me compute: the icosahedron has vertices at the 12 points above.
    # Each face is a triangle of 3 vertices. The center of each face is the average.
    # But by symmetry, the 20 face centers form two orbits of 10 under I_h.
    # Wait, no. Under the full I_h (120 elements), the 20 face centers form a single orbit of 20? No.
    # The icosahedron has 20 faces. Under I_h, all 20 faces are equivalent, so 20 face centers form one orbit of size 20.
    # Hmm, but 20 is not a standard orbit size for I_h. Let me reconsider.
    # Actually, the stabilizer of a face in I_h is C_s (order 2), so orbit size = 120/2 = 60? No, that's too many.
    # I think I'm overcomplicating. Let me just use known orbit sizes: 1, 6, 12, 24, 30, 60.
    # The 6 5-fold axes: these are lines, not points. As points on a sphere, each axis gives 2 antipodal points, so 12 points total.
    # So the "6 orbit" doesn't exist for points. The smallest non-trivial orbit is 12 (vertices) or 6 (if we consider the 6 5-fold axes as 6 points on the sphere, but that's not a closed orbit under I_h).
    # Actually, the 6 5-fold axes intersect the sphere at 12 points (6 pairs of antipodes). These 12 points are the vertices of a dodecahedron (dual of icosahedron). So they form an orbit of 12, not 6.
    # So valid orbit sizes for I_h acting on R^3: 1 (origin), 6 (no, 6 is not a valid orbit size), 12, 24, 30, 60.
    # Wait, 6 IS a valid orbit size. The 6 3-fold axes intersect the sphere at 12 points (6 pairs), forming an orbit of 12. The 10 2-fold axes intersect at 20 points, forming an orbit of 20? No, 20 is not standard.
    # Let me just use: 12 (icosahedron vertices), 30 (edge midpoints), 60 (face centers + something), 24, etc.
    # For simplicity, I'll use orbits of size 12, 30, 60.
    return None  # Not using 6

def _orbit_30():
    """30 edge midpoints of icosahedron."""
    verts = _orbit_12()
    # Find edges: pairs of vertices at minimum distance
    min_d = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_d:
                min_d = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_d) < 1e-9:
                edges.append((i, j))
    mids = [(verts[i] + verts[j]) / 2.0 for i, j in edges]
    return np.array(mids)

def _orbit_60():
    """60 points: 20 face centers + 40 more? Actually, let's use 20 face centers and 40 others.
    Or just use 20 face centers as a 20-orbit. But 20 is not a standard I_h orbit size.
    Let me just generate 60 points by taking the 30 edge midpoints and their antipodes, plus... no.
    Actually, for the purposes of this code, I'll use orbits of size 12, 30, and 20 (face centers).
    The 20 face centers: each face of the icosahedron has a center. There are 20 faces.
    Under I_h, are all 20 face centers in one orbit? The stabilizer of a face center in I_h is C_s (order 2), so orbit size = 120/2 = 60. But there are only 20 face centers. Contradiction.
    I think the issue is that I_h has 120 elements, and the stabilizer of a face (as a set of 3 vertices) is of order 2 (the reflection in the face plane), so the orbit of a face is 120/2 = 60 faces? But an icosahedron only has 20 faces.
    I'm confusing myself. Let me just use the 12 vertices and 30 edge midpoints, which are well-defined.
    """
    # Use 20 face centers
    verts = _orbit_12()
    min_d = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_d:
                min_d = d
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_d) < 1e-9:
                edges.add((i, j))
    # Find faces: triples of mutually connected vertices
    faces = set()
    for i in range(12):
        for j in range(i+1, 12):
            if (i,j) not in edges: continue
            for k in range(j+1, 12):
                if (j,k) not in edges: continue
                if (i,k) not in edges: continue
                faces.add((i,j,k))
    centers = [(verts[i]+verts[j]+verts[k])/3.0 for i,j,k in faces]
    return np.array(centers)

# ---- Symmetry-locked optimization ----
def _symmetric_positions(params, orbit_dirs_list):
    """
    params: array of radial distances, one per orbit.
    orbit_dirs_list: list of arrays of unit direction vectors, one per orbit.
    Returns (n, 3) array of positions.
    """
    positions = []
    for r, dirs in zip(params, orbit_dirs_list):
        positions.append(dirs * r)
    return np.vstack(positions)

def _symmetric_energy(params, orbit_dirs_list, kind):
    """Energy of symmetric configuration."""
    pos = _symmetric_positions(params, orbit_dirs_list)
    try:
        e, _ = sutton_chen_energy_gradient(pos, kind)
        if not np.isfinite(e):
            return 1e10
        return e
    except:
        return 1e10

def _symmetric_optimize(n, kind, seed=0):
    """
    Optimize a symmetric configuration with n atoms using I_h orbits.
    Returns (energy, positions) or (None, None).
    """
    # Determine orbit composition for n atoms
    # Orbits available: 1 (center), 12 (vertices), 30 (edge mids), 20 (face centers)
    # Total must equal n.
    orbits_info = {
        1: np.array([[0.0, 0.0, 0.0]]),
        12: _orbit_12(),
        30: _orbit_30(),
        20: _orbit_60(),
    }
    # Normalize directions to unit length
    for k in orbits_info:
        if k > 1:
            norms = np.linalg.norm(orbits_info[k], axis=1, keepdims=True)
            orbits_info[k] = orbits_info[k] / norms

    # Find combinations of orbits that sum to n
    # Possible orbit sizes: 1, 12, 20, 30
    # We need sum of (count_i * size_i) = n
    # Try different combinations
    best_energy = np.inf
    best_pos = None

    # Generate possible orbit compositions
    # For each number of 30-orbits (0, 1, ...), 20-orbits, 12-orbits, and 1-orbit (0 or 1)
    for n30 in range(0, n // 30 + 1):
        rem30 = n - n30 * 30
        for n20 in range(0, rem30 // 20 + 1):
            rem20 = rem30 - n20 * 20
            for n12 in range(0, rem20 // 12 + 1):
                rem12 = rem20 - n12 * 12
                if rem12 == 0 or rem12 == 1:
                    # Valid composition
                    orbit_list = []
                    param_list = []
                    if rem12 == 1:
                        orbit_list.append(orbits_info[1])
                        param_list.append([0.0])  # center at origin
                    for _ in range(n12):
                        orbit_list.append(orbits_info[12])
                        param_list.append([1.0])
                    for _ in range(n20):
                        orbit_list.append(orbits_info[20])
                        param_list.append([1.5])
                    for _ in range(n30):
                        orbit_list.append(orbits_info[30])
                        param_list.append([1.3])

                    # Initial radial parameters
                    x0 = np.array([p[0] for p in param_list], dtype=float)
                    # Add some variation based on seed
                    rng = np.random.default_rng(seed)
                    x0 = x0 * (1.0 + rng.uniform(-0.1, 0.1, size=len(x0)))
                    x0 = np.clip(x0, 0.1, 5.0)

                    # Optimize
                    try:
                        result = minimize(
                            _symmetric_energy, x0,
                            args=(orbit_list, kind),
                            method='L-BFGS-B',
                            bounds=[(0.1, 5.0)] * len(x0),
                            options={'maxiter': 200, 'ftol': 1e-12}
                        )
                        if result.success and np.isfinite(result.fun):
                            pos = _symmetric_positions(result.x, orbit_list)
                            if len(pos) == n and _validate(pos):
                                if result.fun < best_energy:
                                    best_energy = result.fun
                                    best_pos = pos
                    except:
                        pass

    if best_pos is not None:
        return best_energy, best_pos
    return None, None

# ---- Basin hopping for unconstrained optimization ----
def _perturb(positions, magnitude=0.2, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, magnitude, positions.shape)
    return positions + noise

def _basin_hopping_worker(start, kind, steps=3, seed=0):
    """Basin hopping: relax, perturb, relax, ..."""
    try:
        current_pos = start.copy()
        best_energy = np.inf
        best_pos = None

        for step in range(steps):
            if _time_left() < 10.0:
                break
            energy, pos = relax(current_pos, 'sutton_chen', kind)
            if np.all(np.isfinite(pos)) and _validate(pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos.copy()
                if step < steps - 1:
                    mag = 0.1 + 0.05 * step
                    current_pos = _perturb(pos, magnitude=mag, seed=seed + step * 100)
            else:
                break

        if best_pos is None:
            if _validate(start):
                e, _ = sutton_chen_energy_gradient(start, kind)
                return e, start
            return None, None
        return best_energy, best_pos
    except:
        return None, None

# ---- Initialization generators ----
def _generate_fcc(n, seed=0):
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.3
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

def _generate_bcc(n, seed=0):
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (2.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.3
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append((np.array([ix, iy, iz]) + 0.5) * a)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

def _generate_icosahedral(n, seed=0):
    """Generate icosahedral cluster with shells."""
    rng = np.random.default_rng(seed)
    verts = _orbit_12()
    points = [np.array([0.0, 0.0, 0.0])]
    R1 = 1.0
    for v in verts:
        points.append(v * R1)
    # Edge midpoints
    min_d = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_d:
                min_d = d
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_d) < 1e-9:
                edges.append((i, j))
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid_dir = mid / np.linalg.norm(mid)
        R2 = np.sqrt(R1**2 - (min_d/2.0)**2)
        points.append(mid_dir * R2)
    # Face centers
    faces = set()
    for i in range(12):
        for j in range(i+1, 12):
            if (i,j) not in edges: continue
            for k in range(j+1, 12):
                if (j,k) not in edges: continue
                if (i,k) not in edges: continue
                faces.add((i,j,k))
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        points.append(center)
    points = np.array(points)
    if len(points) >= n:
        return points[:n]
    else:
        # Add more on outer shell
        remaining = n - len(points)
        for _ in range(remaining):
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            r = 1.5 + rng.uniform(0, 0.3)
            points = np.vstack([points, [dir_vec * r]])
        return points[:n]

def _generate_random_compact(n, seed=0):
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    max_attempts = n * 3000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.45:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _polish(positions, kind):
    """Final polish with small perturbations."""
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if _time_left() > 15.0 and np.all(np.isfinite(pos)) and _validate(pos):
            for seed in [42, 123, 777]:
                if _time_left() < 10.0:
                    break
                perturbed = _perturb(pos, magnitude=0.04, seed=seed)
                e2, pos2 = relax(perturbed, 'sutton_chen', kind)
                if np.all(np.isfinite(pos2)) and _validate(pos2) and e2 < energy:
                    energy, pos = e2, pos2
        return energy, pos
    except:
        return None, None

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()

    n = instance["n"]
    kind = instance.get("kind", "12-6")

    best_energy = np.inf
    best_positions = None

    # 1. Symmetry-locked optimization (fast, low-dimensional)
    for seed in range(6):
        if _time_left() < 30.0:
            break
        e, pos = _symmetric_optimize(n, kind, seed=seed)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos

    # 2. Generate diverse starting configurations
    starts = []
    for i in range(6):
        try:
            ico = _generate_icosahedral(n, seed=i)
            if len(ico) == n and _validate(ico):
                starts.append(ico)
        except:
            pass
    for i in range(4):
        try:
            fcc = _generate_fcc(n, seed=100+i)
            if len(fcc) == n and _validate(fcc):
                starts.append(fcc)
        except:
            pass
    for i in range(4):
        try:
            bcc = _generate_bcc(n, seed=200+i)
            if len(bcc) == n and _validate(bcc):
                starts.append(bcc)
        except:
            pass
    for i in range(6):
        try:
            rand = _generate_random_compact(n, seed=300+i)
            if len(rand) == n and _validate(rand):
                starts.append(rand)
        except:
            pass

    if not starts:
        starts = [_generate_random_compact(n, seed=0)]

    # 3. Basin hopping on diverse starts
    if n <= 50:
        num_starts = min(20, len(starts))
        steps_per_start = 4
    elif n <= 70:
        num_starts = min(16, len(starts))
        steps_per_start = 3
    else:
        num_starts = min(12, len(starts))
        steps_per_start = 2

    starts = starts[:num_starts]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_basin_hopping_worker, start, kind, steps=steps_per_start, seed=i*7): i
                   for i, start in enumerate(starts)}
        for future in as_completed(futures):
            if _time_left() < 30.0:
                break
            res = future.result()
            if res[0] is not None:
                energy, positions = res
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()

    # 4. Final polish
    if best_positions is not None and _time_left() > 15.0 and _validate(best_positions):
        polished_e, polished_pos = _polish(best_positions, kind)
        if polished_e is not None and polished_e < best_energy:
            best_energy = polished_e
            best_positions = polished_pos

    # Fallback
    if best_positions is None or not _validate(best_positions):
        best_positions = _generate_random_compact(n, seed=999)
        # Fix any close pairs
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36

    return best_positions
# EVOLVE-BLOCK-END
