# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(pos, ang):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    # Outer hexagon centered at origin, angle 0
    # Its 6 inward normals point at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.radians([30, 90, 150, 210, 270, 330])
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    apothem_factor = np.sqrt(3) / 2
    
    max_proj = 0.0
    for i in range(len(pos)):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        projs = verts @ normals.T
        max_proj = max(max_proj, float(np.max(projs)))
    s = max_proj / apothem_factor
    return max(s, np.sqrt(12) - 1e-12)

def _check_nonoverlap(pos, ang, tol=1e-6):
    """Check that no two inner hexagons overlap using SAT."""
    n = len(pos)
    verts_list = [_hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i]) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            vi = verts_list[i]
            vj = verts_list[j]
            separated = False
            for v in (vi, vj):
                for k in range(6):
                    e = v[(k+1)%6] - v[k]
                    norm = np.array([-e[1], e[0]])
                    ln = np.linalg.norm(norm)
                    if ln > 1e-12:
                        norm = norm / ln
                        pi = vi @ norm
                        pj = vj @ norm
                        if np.max(pi) + tol < np.min(pj) or np.max(pj) + tol < np.min(pi):
                            separated = True
                            break
                if separated:
                    break
            if not separated:
                return False
    return True

def _check_containment(pos, ang, s, tol=1e-6):
    """Check all inner vertices are inside the outer hexagon."""
    outer_verts = _hex_vertices(0, 0, s, 0.0)
    for i in range(len(pos)):
        verts = _hex_vertices(pos[i,0], pos[i,1], 1.0, ang[i])
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                cross = (p2[0]-p1[0])*(v[1]-p1[1]) - (p2[1]-p1[1])*(v[0]-p1[0])
                if cross < -tol:
                    return False
    return True

def _validity_check(pos, ang, s):
    """Full validity check."""
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(ang)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    if not _check_nonoverlap(pos, ang):
        return False
    if not _check_containment(pos, ang, s):
        return False
    return True

def _honeycomb_init():
    """Natural honeycomb arrangement of 12 hexagons."""
    sq3 = np.sqrt(3)
    # Center + 6 neighbors + 5 more in second ring
    pos = np.array([
        [0, 0],
        [3, 0],
        [1.5, sq3],
        [-1.5, sq3],
        [-3, 0],
        [-1.5, -sq3],
        [1.5, -sq3],
        [4.5, sq3],
        [4.5, -sq3],
        [3, 2*sq3],
        [0, 2*sq3],
        [-3, 2*sq3],
    ])
    ang = np.zeros(12)
    return pos, ang

def _simulated_annealing(pos0, ang0, max_iter=50000, t_start=0.5, t_end=0.001, deadline=None):
    """SA on positions only (angles fixed)."""
    pos = pos0.copy()
    ang = ang0.copy()
    s0 = _min_outer_side(pos, ang)
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_s = s0
    
    n = len(pos)
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Random move: perturb one hexagon's position
        i = rng.randint(n)
        dx, dy = rng.randn(2) * 0.1
        old_x, old_y = pos[i, 0], pos[i, 1]
        pos[i, 0] += dx
        pos[i, 1] += dy
        
        s_new = _min_outer_side(pos, ang)
        delta = s_new - best_s
        
        t = t_start * (t_end / t_start) ** (it / max_iter)
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-10)):
            if s_new < best_s:
                best_s = s_new
                best_pos = pos.copy()
                best_ang = ang.copy()
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
    
    return best_pos, best_ang, best_s

def _local_polish(pos, ang, deadline=None):
    """Local polish: try small perturbations to reduce s."""
    s = _min_outer_side(pos, ang)
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_s = s
    n = len(pos)
    rng = np.random.RandomState(123)
    
    for it in range(20000):
        if deadline is not None and time.time() > deadline:
            break
        i = rng.randint(n)
        dx, dy = rng.randn(2) * 0.01
        old_x, old_y = best_pos[i, 0], best_pos[i, 1]
        test_pos = best_pos.copy()
        test_pos[i, 0] += dx
        test_pos[i, 1] += dy
        s_new = _min_outer_side(test_pos, best_ang)
        if s_new < best_s:
            best_s = s_new
            best_pos = test_pos
        else:
            pass
    return best_pos, best_ang, best_s

def hexagon_packing_12():
    start = time.time()
    deadline = start + 580.0  # safety margin
    
    # Start from honeycomb
    pos, ang = _honeycomb_init()
    s0 = _min_outer_side(pos, ang)
    best_pos = pos.copy()
    best_ang = ang.copy()
    best_s = s0
    
    # SA phase
    if time.time() < deadline - 100:
        pos_sa, ang_sa, s_sa = _simulated_annealing(pos, ang, max_iter=80000, t_start=0.3, t_end=0.0005, deadline=deadline - 100)
        if s_sa < best_s:
            best_s = s_sa
            best_pos = pos_sa
            best_ang = ang_sa
    
    # Local polish
    if time.time() < deadline - 30:
        pos_p, ang_p, s_p = _local_polish(best_pos, best_ang, deadline=deadline - 30)
        if s_p < best_s:
            best_s = s_p
            best_pos = pos_p
            best_ang = ang_p
    
    # Try a few angle variations
    if time.time() < deadline - 20:
        for angle_offset in [0, 30, 15]:
            ang_test = best_ang.copy()
            ang_test[:] = angle_offset
            s_test = _min_outer_side(best_pos, ang_test)
            if s_test < best_s:
                best_s = s_test
                best_ang = ang_test
    
    # Build output
    inner = np.zeros((12, 3))
    for i in range(12):
        inner[i, 0] = best_pos[i, 0]
        inner[i, 1] = best_pos[i, 1]
        inner[i, 2] = best_ang[i]
    outer = np.array([0.0, 0.0, 0.0])
    s = _min_outer_side(best_pos, best_ang)
    
    # Validate; if invalid, fall back to a known-valid config
    if not _validity_check(best_pos, best_ang, s):
        # Fallback: slightly expanded honeycomb
        pos_f, ang_f = _honeycomb_init()
        s_f = _min_outer_side(pos_f, ang_f)
        if not _validity_check(pos_f, ang_f, s_f):
            # Even more expanded
            pos_f = pos_f * 1.05
            s_f = _min_outer_side(pos_f, ang_f)
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = pos_f[i, 0]
            inner[i, 1] = pos_f[i, 1]
            inner[i, 2] = ang_f[i]
        outer = np.array([0.0, 0.0, 0.0])
        s = s_f
    
    return inner, outer, s

# EVOLVE-BLOCK-END
