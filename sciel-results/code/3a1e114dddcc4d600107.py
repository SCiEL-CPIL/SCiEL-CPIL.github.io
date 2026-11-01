import time
import math
import numpy as np

N = 150
_DEADLINE_MARGIN = 5.0
_WALL_CLOCK_LIMIT = 150.0

def _icosahedron_vertices():
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts

def _rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    c = math.cos(angle)
    s = math.sin(angle)
    k = 1.0 - c
    x, y, z = axis
    R = np.array([
        [c + k*x*x, k*x*y - s*z, k*x*z + s*y],
        [k*x*y + s*z, c + k*y*y, k*y*z - s*x],
        [k*x*z - s*y, k*y*z + s*x, c + k*z*z]
    ])
    return R

def _generate_icosahedral_group():
    verts = _icosahedron_vertices()
    
    # 5-fold axis: through face centers
    faces = [
        [0, 1, 4], [0, 1, 8], [0, 8, 10], [0, 10, 3], [0, 3, 2],
        [1, 4, 9], [1, 9, 8], [2, 3, 10], [2, 10, 11], [2, 11, 5],
        [3, 5, 11], [4, 5, 1], [4, 9, 6], [4, 6, 7], [4, 7, 5],
        [6, 9, 7], [7, 9, 5], [7, 11, 6], [8, 9, 10], [10, 11, 3]
    ]
    
    face_centers = []
    for f in faces:
        fc = np.mean(verts[f], axis=0)
        fc /= np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    z_axis = np.array([0, 0, 1.0])
    z_dots = np.dot(face_centers, z_axis)
    idx_max_z = np.argmax(z_dots)
    axis_5fold = face_centers[idx_max_z]
    
    R_5fold = _rotation_matrix(axis_5fold, 2.0 * math.pi / 5.0)
    
    # 2-fold axis: through edge midpoints
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 1.4 < d < 1.5:
                edges.append((i, j))
    
    edge_mids = []
    for i, j in edges:
        em = (verts[i] + verts[j]) / 2.0
        em /= np.linalg.norm(em)
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    
    x_axis = np.array([1.0, 0, 0])
    x_dots = np.dot(edge_mids, x_axis)
    idx_max_x = np.argmax(x_dots)
    axis_2fold = edge_mids[idx_max_x]
    
    R_2fold = _rotation_matrix(axis_2fold, math.pi)
    
    group = [np.eye(3)]
    for _ in range(50):
        new_rots = []
        for g in group:
            for gen in [R_5fold, R_2fold]:
                new_rot = gen @ g
                is_new = True
                for existing in group:
                    if np.allclose(new_rot, existing, atol=1e-10):
                        is_new = False
                        break
                if is_new:
                    new_rots.append(new_rot)
        if not new_rots:
            break
        group.extend(new_rots)
    
    return np.array(group)

def _generate_orbit(pt, group):
    orbit = group @ pt
    unique_orbit = [orbit[0]]
    for p in orbit[1:]:
        is_dup = False
        for up in unique_orbit:
            if np.linalg.norm(p - up) < 1e-10:
                is_dup = True
                break
        if not is_dup:
            unique_orbit.append(p)
    return np.array(unique_orbit)

def _log_energy(pts):
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad_symmetric(reps, group):
    k = len(reps)
    full_pts = []
    orbit_sizes = []
    for rep in reps:
        orbit = _generate_orbit(rep, group)
        full_pts.append(orbit)
        orbit_sizes.append(len(orbit))
    
    full_pts = np.vstack(full_pts)
    n = len(full_pts)
    
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(full_pts[i] - full_pts[j])
            if d < 1e-12:
                return float('inf'), np.zeros_like(reps)
            energy += math.log(1.0 / d)
    
    grad_full = np.zeros((n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = full_pts[i] - full_pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                continue
            grad_full[i] += -diff / (d * d)
    
    for i in range(n):
        dot_prod = np.dot(grad_full[i], full_pts[i])
        grad_full[i] -= dot_prod * full_pts[i]
    
    grad_reps = np.zeros_like(reps)
    idx = 0
    for rep_idx in range(k):
        orbit_size = orbit_sizes[rep_idx]
        grad_reps[rep_idx] = np.sum(grad_full[idx:idx+orbit_size], axis=0)
        idx += orbit_size
    
    for i in range(k):
        dot_prod = np.dot(grad_reps[i], reps[i])
        grad_reps[i] -= dot_prod * reps[i]
    
    return energy, grad_reps

def _exponential_map(pts, tangent_vecs):
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _lbfgs_symmetric(reps, group, time_limit=10.0, max_iter=200, m=10):
    start = time.time()
    current = reps.copy()
    energy, grad = _energy_and_grad_symmetric(current, group)
    
    if not np.isfinite(energy):
        return current, energy
    
    s_history = []
    y_history = []
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        q = grad.copy()
        rho = []
        
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            alpha_k = np.sum(s_k * q) / (np.sum(y_k * y_k) + 1e-30)
            rho.append(alpha_k)
            q -= alpha_k * y_k
        
        r = q.copy()
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            beta_k = np.sum(y_k * r) / (np.sum(y_k * y_k) + 1e-30)
            d += (rho[k] - beta_k) * s_k
        
        for i in range(len(current)):
            dot_d = np.sum(d[i] * current[i])
            d[i] -= dot_d * current[i]
        
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad_symmetric(new_pts, group)
            
            if not np.isfinite(new_energy):
                alpha *= 0.5
                continue
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts.copy()
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy <= energy + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.7
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            break
        
        s_vec = best_pts - current
        y_vec = best_grad - grad
        
        for i in range(len(s_vec)):
            dot_s = np.sum(s_vec[i] * current[i])
            s_vec[i] -= dot_s * current[i]
            dot_y = np.sum(y_vec[i] * best_pts[i])
            y_vec[i] -= dot_y * best_pts[i]
        
        y_norm = np.linalg.norm(y_vec)
        s_norm = np.linalg.norm(s_vec)
        if y_norm > 1e-12 and s_norm > 1e-12:
            s_history.append(s_vec)
            y_history.append(y_vec)
            if len(s_history) > m:
                s_history.pop(0)
                y_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy

def _build_symmetric_config(reps, group):
    full_pts = []
    for rep in reps:
        orbit = _generate_orbit(rep, group)
        full_pts.append(orbit)
    return np.vstack(full_pts)

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + _WALL_CLOCK_LIMIT - _DEADLINE_MARGIN
    
    group = _generate_icosahedral_group()
    
    best_pts = None
    best_energy = float('inf')
    
    # Strategy: 120-orbit + 30-orbit
    verts = _icosahedron_vertices()
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 1.4 < d < 1.5:
                edges.append((i, j))
    
    edge_mids = []
    for i, j in edges:
        em = (verts[i] + verts[j]) / 2.0
        em /= np.linalg.norm(em)
        edge_mids.append(em)
    
    unique_axes_2fold = []
    for em in edge_mids:
        is_new = True
        for ua in unique_axes_2fold:
            if abs(np.dot(em, ua)) > 0.9999:
                is_new = False
                break
        if is_new:
            unique_axes_2fold.append(em)
    
    axis_2fold = unique_axes_2fold[0]
    
    rng = np.random.default_rng(42)
    rep1 = rng.standard_normal(3)
    rep1 /= np.linalg.norm(rep1)
    rep2 = axis_2fold.copy()
    
    reps = np.array([rep1, rep2])
    
    orbit1 = _generate_orbit(reps[0], group)
    orbit2 = _generate_orbit(reps[1], group)
    
    if len(orbit1) + len(orbit2) == 150:
        time_left = deadline - start_time
        if time_left > 10:
            opt_reps, opt_energy = _lbfgs_symmetric(reps, group, time_limit=time_left - 5, max_iter=300)
            if np.isfinite(opt_energy):
                full_pts = _build_symmetric_config(opt_reps, group)
                if len(full_pts) == 150:
                    e = _log_energy(full_pts)
                    if np.isfinite(e) and e < best_energy:
                        best_energy = e
                        best_pts = full_pts
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy(best_pts)
    
    return _normalize(best_pts)
