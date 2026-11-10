# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(inner_hex_data):
    """Find min side length of outer hex (centered at origin) containing all inner hexes."""
    all_verts = []
    for i in range(len(inner_hex_data)):
        v = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    def s_of_angle(a_rad):
        angles = a_rad + np.deg2rad(60 * np.arange(6))
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        dots = all_verts[:, 0] * cos_a + all_verts[:, 1] * sin_a
        max_dot = np.max(dots, axis=0)
        return (2.0 / np.sqrt(3.0)) * np.max(max_dot)
    
    coarse = np.linspace(0, np.pi / 3, 181)
    s_vals = np.array([s_of_angle(a) for a in coarse])
    best_idx = np.argmin(s_vals)
    lo = coarse[max(0, best_idx - 1)]
    hi = coarse[min(len(coarse) - 1, best_idx + 1)]
    for _ in range(30):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if s_of_angle(m1) < s_of_angle(m2):
            hi = m2
        else:
            lo = m1
    best_a = (lo + hi) / 2
    return s_of_angle(best_a), best_a


def _check_no_overlap(inner_hex_data, tol=1e-6):
    n = len(inner_hex_data)
    for i in range(n):
        vi = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
        for j in range(i + 1, n):
            vj = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            normals = []
            for v in [vi, vj]:
                for k in range(6):
                    edge = v[(k + 1) % 6] - v[k]
                    nrm = np.array([-edge[1], edge[0]])
                    ln = np.linalg.norm(nrm)
                    if ln > 1e-12:
                        normals.append(nrm / ln)
            separated = False
            for nrm in normals:
                pi = vi @ nrm
                pj = vj @ nrm
                gap = max(np.min(pj) - np.max(pi), np.min(pi) - np.max(pj))
                if gap > tol:
                    separated = True
                    break
            if not separated:
                return False
    return True


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620
    
    sqrt3 = np.sqrt(3)
    
    # Start with hexagonal lattice arrangement (compact)
    # Use axial coordinates for a hexagonal cluster of 12
    # Best compact shapes for 12 hexes on hex grid:
    # Try several initial arrangements and pick the best
    
    candidates = []
    
    # Arrangement 1: 2 rows of 6 (staggered)
    arr1 = np.zeros((12, 3))
    for i in range(6):
        arr1[i, 0] = i * sqrt3
        arr1[i, 1] = 0
    for i in range(6):
        arr1[6 + i, 0] = (i + 0.5) * sqrt3
        arr1[6 + i, 1] = 1.5
    candidates.append(arr1)
    
    # Arrangement 2: 3 rows of 4 (staggered)
    arr2 = np.zeros((12, 3))
    for i in range(4):
        arr2[i, 0] = i * sqrt3
        arr2[i, 1] = 0
    for i in range(4):
        arr2[4 + i, 0] = (i + 0.5) * sqrt3
        arr2[4 + i, 1] = 1.5
    for i in range(4):
        arr2[8 + i, 0] = i * sqrt3
        arr2[8 + i, 1] = 3.0
    candidates.append(arr2)
    
    # Arrangement 3: Centered hex cluster (1 + 6 + 5 from ring 2)
    arr3 = np.zeros((12, 3))
    arr3[0, 0] = 0; arr3[0, 1] = 0
    ring1 = [(sqrt3, 0), (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
             (-sqrt3, 0), (-sqrt3/2, -1.5), (sqrt3/2, -1.5)]
    for k, (x, y) in enumerate(ring1):
        arr3[1 + k, 0] = x
        arr3[1 + k, 1] = y
    # 5 from ring 2: pick adjacent ones
    ring2_subset = [
        (2*sqrt3, 0), (3*sqrt3/2, 1.5), (sqrt3/2, 3.0),
        (-sqrt3/2, 3.0), (-3*sqrt3/2, 1.5)
    ]
    for k, (x, y) in enumerate(ring2_subset):
        arr3[7 + k, 0] = x
        arr3[7 + k, 1] = y
    candidates.append(arr3)
    
    # Center each arrangement at origin
    for arr in candidates:
        center = np.mean(arr[:, :2], axis=0)
        arr[:, :2] -= center
    
    # Evaluate each candidate
    best_arr = None
    best_s = np.inf
    best_angle = 0
    for arr in candidates:
        if not _check_no_overlap(arr):
            continue
        s, a = _min_containing_hex_side(arr)
        if s < best_s:
            best_s = s
            best_arr = arr.copy()
            best_angle = a
    
    if best_arr is None:
        # Fallback
        best_arr = candidates[0].copy()
        best_s, best_angle = _min_containing_hex_side(best_arr)
    
    # Local optimization
    np.random.seed(42)
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        trial = best_arr.copy()
        idx = np.random.randint(12)
        trial[idx, 0] += np.random.normal(0, 0.1)
        trial[idx, 1] += np.random.normal(0, 0.1)
        
        if _check_no_overlap(trial):
            s_trial, a_trial = _min_containing_hex_side(trial)
            if s_trial < best_s - 1e-10:
                best_s = s_trial
                best_arr = trial
                best_angle = a_trial
    
    outer_hex_data = np.array([0, 0, np.rad2deg(best_angle)])
    return best_arr, outer_hex_data, best_s


# EVOLVE-BLOCK-END
