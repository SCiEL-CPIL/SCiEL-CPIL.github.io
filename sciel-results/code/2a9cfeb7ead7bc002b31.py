import numpy as np
import time
import math
import random
from scipy.optimize import minimize, LbfgsB

N = 12
DEADLINE = 148.0

def _hex_verts(xys_rots):
    """(N,3) -> (N,6,2) vertex array. xys_rots = [x, y, angle_deg]."""
    base = np.radians(np.array([0, 60, 120, 180, 240, 300], dtype=float))
    rots = np.radians(xys_rots[:, 2])
    ang = base[None, :] + rots[:, None]
    cx = xys_rots[:, 0, None]
    cy = xys_rots[:, 1, None]
    x = cx + np.cos(ang)
    y = cy + np.sin(ang)
    return np.stack((x, y), axis=-1)

def _hex_normals(xys_rots):
    """(N,3) -> (N,6,2) outward unit normals for each hexagon's 6 edges."""
    base = np.radians(np.array([30, 90, 150, 210, 270, 330], dtype=float))
    rots = np.radians(xys_rots[:, 2])
    ang = base[None, :] + rots[:, None]
    nx = np.cos(ang)
    ny = np.sin(ang)
    return np.stack((nx, ny), axis=-1)

def _sat_penetrations(xys_rots):
    """Vectorized SAT: return array of positive penetration depths for overlapping pairs."""
    verts = _hex_verts(xys_rots)       # (N,6,2)
    norms = _hex_normals(xys_rots)     # (N,6,2)
    centers = xys_rots[:, :2]
    
    # For each hex, project vertices onto its own normals -> (N, 6, 6)
    # proj[i, a, b] = dot(vert_i[b], norm_i[a])
    # Use einsum: verts (N,6,2) @ norms (N,6,2)^T -> (N,6,6)
    proj_own = np.einsum('nbd,ndd->nbn', verts, norms)  # (N, 6, 6): for hex n, axis a, vertex b
    # Actually simpler: proj[n, a, b] = verts[n,b] . norms[n,a]
    proj_own = np.einsum('nba,naa->nab', verts, norms)  # (N, 6_axes, 6_verts)
    
    min_own = np.min(proj_own, axis=2)  # (N, 6)
    max_own = np.max(proj_own, axis=2)  # (N, 6)
    
    # For cross-projections: proj_cross[i, j, a, b] = dot(vert_i[b], norm_j[a])
    # = verts[i, b] . norms[j, a]
    # Shape: (N, N, 6_axes_of_j, 6_verts_of_i)
    # proj_cross[i,j,a,b] = einsum over last dim
    # verts: (N, 6, 2), norms: (N, 6, 2)
    # We want: for each pair (i,j), project verts[i] onto norms[j]'s axes
    # proj_cross[i, j, a, b] = verts[i, b, d] * norms[j, a, d]
    # = np.einsum('ibd,jad->ijba', verts, norms)  -> (N,N,6,6)
    proj_cross = np.einsum('ibd,jad->ijba', verts, norms)  # (N, N, 6, 6)
    
    min_cross = np.min(proj_cross, axis=3)  # (N, N, 6): for pair (i,j), axis a of j, min over verts of i
    max_cross = np.max(proj_cross, axis=3)  # (N, N, 6)
    
    # Now for pair (i,j): check 12 axes (6 from i's normals, 6 from j's normals)
    # Axes from i: use min_own[i], max_own[i] vs proj_cross[j,i,:,:] projected onto i's normals
    # Actually, let's be more careful:
    # For pair (i,j), the separation axes are normals of i (6) and normals of j (6).
    # For axis a of hex i: project hex i -> [min_own[i,a], max_own[i,a]]
    #                   project hex j -> [min_cross[j,i,a], max_cross[j,i,a]]
    # For axis a of hex j: project hex i -> [min_cross[i,j,a], max_cross[i,j,a]]
    #                   project hex j -> [min_own[j,a], max_own[j,a]]
    
    # Let's compute overlap for all pairs and all 12 axes
    # Shape target: (N, N, 12) where 12 = 6 axes from first + 6 axes from second
    
    # Overlap on axis a of hex i for pair (i,j):
    # min_i = min_own[i, a], max_i = max_own[i, a]
    # min_j = min_cross[j, i, a], max_j = max_cross[j, i, a]
    # overlap = min(max_i, max_j) - max(min_i, min_j)
    
    # For all i,j:
    # overlap_i_axis[i,j,a] = min(max_own[i,a], max_cross[j,i,a]) - max(min_own[i,a], min_cross[j,i,a])
    # overlap_j_axis[i,j,a] = min(max_cross[i,j,a], max_own[j,a]) - max(min_cross[i,j,a], min_own[j,a])
    
    # max_own: (N,6), need (N,N,6): max_own[i,a] -> max_own[i,:,None] broadcast with j
    mo_i = max_own[:, :, None]   # (N, 6, 1)  -> broadcast over j
    mo_j = max_own[None, :, :]   # (1, N, 6)   -> broadcast over i
    mn_i = min_own[:, :, None]   # (N, 6, 1)
    mn_j = min_own[None, :, :]   # (1, N, 6)
    
    # max_cross[j, i, a]: shape (N, N, 6) where [j,i,a]
    # We need for pair (i,j), axis a of i: max_cross[j, i, a]
    # max_cross has shape (N, N, 6, 6) = [first_hex, second_hex, axis_of_second, vert_of_first]
    # Wait, I defined proj_cross[i,j,a,b] = dot(vert_i[b], norm_j[a])
    # So proj_cross[j, i, a, b] = dot(vert_j[b], norm_i[a]) -> project hex j onto hex i's axis a
    # min_cross[j, i, a] = min over b of dot(vert_j[b], norm_i[a])
    # That's what we need for "project hex j onto axis a of hex i"
    
    # min_cross[j, i, a] from min_cross[j, i, a] where min_cross = np.min(proj_cross, axis=3)
    # min_cross shape: (N, N, 6) = [first, second, axis_of_second]
    # min_cross[j, i, a] = min over verts of hex j projected onto axis a of hex i? 
    # No! proj_cross[i,j,a,b] = dot(vert_i[b], norm_j[a])
    # min_cross[i,j,a] = min_b dot(vert_i[b], norm_j[a]) = min projection of hex i onto axis a of hex j
    
    # So for pair (i,j), axis a of hex i:
    #   hex i projected: [min_own[i,a], max_own[i,a]]
    #   hex j projected onto axis a of i: we need min/max of dot(vert_j, norm_i[a])
    #   = min_cross[j, i, a], max_cross[j, i, a]  (since proj_cross[j,i,a,b]=dot(vert_j[b],norm_i[a]))
    
    # For pair (i,j), axis a of hex j:
    #   hex j projected: [min_own[j,a], max_own[j,a]]
    #   hex i projected onto axis a of j: min_cross[i,j,a], max_cross[i,j,a]
    
    # Overlap on axis a of hex i for pair (i,j):
    # ov_i[i,j,a] = min(max_own[i,a], max_cross[j,i,a]) - max(min_own[i,a], min_cross[j,i,a])
    
    # max_own[i,a]: (N,6) -> for fixed a, max_own[:,a] shape (N,) -> need (N,N): max_own[i,a] for all i,j
    # Use broadcasting: max_own[i,a] is independent of j, so:
    # mo_i_a = max_own[:, a] for all i, repeated over j -> shape (N, N) per a
    # Let's do it with full 3D:
    
    # ov_i: (N, N, 6)
    # max_own[i,a] -> expand to (N, 1, 6) and broadcast over axis 1 (j)
    mo_i3 = max_own[:, None, :]  # (N, 1, 6)
    mn_i3 = min_own[:, None, :]  # (N, 1, 6)
    mo_j3 = max_own[None, :, :]  # (1, N, 6)
    mn_j3 = min_own[None, :, :]  # (1, N, 6)
    
    # max_cross[j, i, a]: we need to index as [j, i, a] from max_cross[i,j,a] shape (N,N,6)
    # max_cross[i,j,a] = max over b of dot(vert_i[b], norm_j[a])
    # We want for pair (i,j): max_cross[j, i, a]
    # So transpose axes 0 and 1: mc_t[i,j,a] = max_cross[j,i,a]
    mc_t = np.transpose(max_cross, (1, 0, 2))  # (N, N, 6): [i, j, a] = max_cross[j, i, a]
    mc_i = np.transpose(min_cross, (1, 0, 2))  # (N, N, 6): [i, j, a] = min_cross[j, i, a]
    
    # Now for pair (i,j), axis a of hex i:
    # hex i: [mn_i3[i,a], mo_i3[i,a]]
    # hex j projected: [mc_i[i,j,a], mc_t[i,j,a]]
    # overlap = min(mo_i3[i,a], mc_t[i,j,a]) - max(mn_i3[i,a], mc_i[i,j,a])
    ov_i = np.minimum(mo_i3, mc_t) - np.maximum(mn_i3, mc_i)  # (N, N, 6)
    
    # For pair (i,j), axis a of hex j:
    # hex j: [mn_j3[j,a], mo_j3[j,a]]
    # hex i projected onto axis a of j: min_cross[i,j,a], max_cross[i,j,a]
    # mc[i,j,a] = max_cross[i,j,a], mc_i... wait
    # min_cross[i,j,a] = min over b dot(vert_i[b], norm_j[a]) -> projection of hex i onto axis a of hex j
    # So:
    ov_j = np.minimum(mo_j3, max_cross) - np.maximum(mn_j3, min_cross)  # (N, N, 6)
    
    # Combine 12 axes per pair
    # Stack: (N, N, 12)
    ov_all = np.concatenate([ov_i, ov_j], axis=2)  # (N, N, 12)
    
    # Separated if any axis has overlap <= 0
    # Penetration = min positive overlap if no separation
    # Use: min_ov = np.min(ov_all, axis=2)
    # If min_ov > 0, they overlap with depth min_ov
    # If min_ov <= 0, they are separated (or just touching)
    
    min_ov = np.min(ov_all, axis=2)  # (N, N)
    
    # Only consider i < j
    # Set diagonal to 0
    min_ov = np.where(np.eye(N, dtype=bool), 0.0, min_ov)
    # Only upper triangle
    tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pen = np.where(tri_mask, np.maximum(min_ov, 0.0), 0.0)
    
    return pen  # (N, N) upper triangle has penetration depths, rest 0

def _smooth_pen(pen_arr, beta=8.0):
    """Smooth sum of penetration depths using soft-max style. pen_arr is (N,N) upper tri."""
    # Extract positive penetrations
    pos = pen_arr[pen_arr > 1e-10]
    if len(pos) == 0:
        return 0.0
    # Soft sum: beta * logsumexp(pos/beta) approximates sum for large beta
    # But we want a smooth approximation of sum(pos^2) or sum(pos)
    # Use: sum(pos) is already smooth where pos > 0, but has kink at 0
    # Better: use a smooth hinge: for each p, f(p) = (p + sqrt(p^2 + eps))/2 * 2 / sqrt(eps) approx p for p>>0
    # Or simply: use softplus(p/beta)*beta as smooth max(0,p)
    # sum of softplus: sum(beta * log(1 + exp(p/beta)))
    # This is smooth everywhere and approximates sum(max(0,p)) for small beta
    sp = beta * np.log1p(np.exp(pos / beta))
    return float(np.sum(sp))

def _outer_side_smooth(xys_rots, outer_rot_deg, beta=10.0):
    """Smooth outer hex side length using log-sum-exp on squared projections."""
    verts = _hex_verts(xys_rots)  # (N, 6, 2)
    all_pts = verts.reshape(-1, 2)  # (N*6, 2)
    
    # Outer hex normals: 6 directions
    base_ox = np.radians(np.array([90, 150, 210, 270, 330, 30], dtype=float) + outer_rot_deg)
    ox = np.cos(base_ox)  # (6,)
    oy = np.sin(base_ox)  # (6,)
    
    # Projections: (N*6, 6)
    proj = all_pts[:, 0:1] * ox[None, :] + all_pts[:, 1:2] * oy[None, :]
    # (N*6, 6)
    
    # For each of 6 axes, max |proj|
    abs_proj = np.abs(proj)  # (N*6, 6)
    
    # max |proj| per axis: (6,)
    # Use soft-max: beta * logsumexp(log(abs_proj)) but log(0) is -inf
    # Better: for each axis a, max_b |proj[b,a]|
    # Use: (1/beta) * log(sum(exp(beta * |proj|))) approximates max |proj|
    # But values can be large; use the stable version
    # max_val = np.max(abs_proj, axis=0)  # (6,) - exact
    # smooth_max = max_val + (1/beta) * np.log(np.mean(np.exp(beta * (abs_proj - max_val[None,:]))))
    
    max_abs = np.max(abs_proj, axis=0)  # (6,)
    # Smooth version for optimization:
    shifted = abs_proj - max_abs[None, :]
    smooth_max = max_abs + (1.0 / beta) * np.log(np.mean(np.exp(beta * shifted), axis=0))
    
    # Side length = (2/sqrt(3)) * max_over_axes(smooth_max)
    # Again smooth the max over 6 axes
    m2 = np.max(smooth_max)
    shifted2 = smooth_max - m2
    smooth_m2 = m2 + (1.0 / beta) * np.log(np.mean(np.exp(beta * shifted2)))
    
    return (2.0 / math.sqrt(3.0)) * smooth_m2

def _exact_outer_side(xys_rots, outer_rot_deg):
    """Exact outer hex side length."""
    verts = _hex_verts(xys_rots)
    all_pts = verts.reshape(-1, 2)
    base_ox = np.radians(np.array([90, 150, 210, 270, 330, 30], dtype=float) + outer_rot_deg)
    ox = np.cos(base_ox)
    oy = np.sin(base_ox)
    proj = all_pts[:, 0:1] * ox[None, :] + all_pts[:, 1:2] * oy[None, :]
    max_abs = np.max(np.abs(proj), axis=0)
    return (2.0 / math.sqrt(3.0)) * np.max(max_abs)

def _is_valid(xys_rots, tol=1e-7):
    pen = _sat_penetrations(xys_rots)
    return np.max(pen) < tol

def _make_grid(rows, cols, gap=0.02):
    pts = []
    s3 = math.sqrt(3)
    sx = s3 + gap
    sy = 1.5 + gap * 0.3
    for r in range(rows):
        y = r * sy
        for c in range(cols):
            x = c * sx
            if r % 2 == 1:
                x += sx / 2.0
            pts.append([x, y, 0.0])
    pts = np.array(pts, dtype=float)
    if len(pts) > N:
        pts = pts[:N]
    # Pad if needed
    while len(pts) < N:
        pts = np.vstack([pts, pts[-1:]])
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def _make_ring(R=2.6, gap=0.05):
    pts = []
    for i in range(N):
        th = 2 * math.pi * i / N
        pts.append([R * math.cos(th), R * math.sin(th), math.degrees(th)])
    pts = np.array(pts, dtype=float)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def _make_hex_lattice(gap=0.03):
    """Compact hex lattice: 1 center, 6 ring, 5 more."""
    pts = []
    s3 = math.sqrt(3)
    d1 = s3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        a = math.radians(60 * k)
        pts.append([d1 * math.cos(a), d1 * math.sin(a), 0.0])
    # 5 more at slightly larger radius, offset angles
    d2 = 2 * s3 + gap
    for k in range(5):
        a = math.radians(60 * k + 30)
        pts.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])
    pts = np.array(pts, dtype=float)
    pts[:, :2] -= np.mean(pts[:, :2], axis=0)
    return pts

def _perturb(xys_rots, scale):
    """Add small random perturbation."""
    p = xys_rots.copy()
    p[:, 0] += np.random.normal(0, scale, N)
    p[:, 1] += np.random.normal(0, scale, scale)
    p[:, 2] += np.random.normal(0, scale * 5, N)
    p[:, 2] %= 360.0
    return p

def construct():
    t0 = time.time()
    
    def tleft():
        return DEADLINE - (time.time() - t0)
    
    def objective(params):
        xys_rots = params[:3*N].reshape(N, 3)
        orot = params[3*N]
        
        if time.time() - t0 > DEADLINE - 3:
            return 1e6
        
        pen = _sat_penetrations(xys_rots)
        pen_val = _smooth_pen(pen, beta=6.0)
        
        L = _outer_side_smooth(xys_rots, orot, beta=12.0)
        
        return L + 200.0 * pen_val
    
    def exact_L(params):
        xys_rots = params[:3*N].reshape(N, 3)
        orot = params[3*N]
        return _exact_outer_side(xys_rots, orot)
    
    best_L = float('inf')
    best_params = None
    
    # Initial configs
    init_cfgs = [
        _make_grid(4, 3),
        _make_grid(3, 4),
        _make_grid(2, 6),
        _make_grid(6, 2),
        _make_hex_lattice(),
        _make_ring(2.6),
        _make_ring(2.4),
    ]
    # Add rotated variants
    for base in init_cfgs[:4]:
        r = base.copy()
        r[:, 2] = 30.0
        init_cfgs.append(r)
    
    outer_rots = [0.0, 30.0, 15.0, -15.0, 45.0]
    
    # Phase 1: Evaluate and optimize from diverse starts
    for cfg in init_cfgs:
        if tleft() < 15.0:
            break
        for orot in outer_rots:
            if tleft() < 15.0:
                break
            
            p0 = np.concatenate([cfg.flatten(), [orot]])
            
            # Check initial
            xys0 = p0[:3*N].reshape(N, 3)
            if _is_valid(xys0):
                L0 = exact_L(p0)
                if L0 < best_L:
                    best_L = L0
                    best_params = p0.copy()
            
            # Nelder-Mead
            try:
                res = minimize(objective, p0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7,
                                        'adaptive': True})
                if _is_valid(res.x[:3*N].reshape(N, 3)):
                    L = exact_L(res.x)
                    if L < best_L:
                        best_L = L
                        best_params = res.x.copy()
            except Exception:
                pass
            
            # L-BFGS-B with finite diff gradients
            try:
                res2 = minimize(objective, p0, method='L-BFGS-B',
                                options={'maxiter': 300, 'gtol': 1e-6, 'ftol': 1e-8})
                if _is_valid(res2.x[:3*N].reshape(N, 3)):
                    L = exact_L(res2.x)
                    if L < best_L:
                        best_L = L
                        best_params = res2.x.copy()
            except Exception:
                pass
    
    # Phase 2: Simulated annealing for global exploration
    if tleft() > 25.0 and best_params is not None:
        cur = best_params.copy()
        cur_xys = cur[:3*N].reshape(N, 3)
        cur_orot = cur[3*N]
        
        if _is_valid(cur_xys):
            cur_L = exact_L(cur)
        else:
            cur_L = exact_L(cur) + 10.0 * _smooth_pen(_sat_penetrations(cur_xys), beta=4.0)
        
        T = 0.3
        cool_rate = 0.995
        
        while tleft() > 20.0:
            # Propose move: perturb one hexagon
            idx = np.random.randint(N)
            new_xys = cur_xys.copy()
            new_xys[idx, 0] += np.random.normal(0, 0.15)
            new_xys[idx, 1] += np.random.normal(0, 0.15)
            new_xys[idx, 2] = (new_xys[idx, 2] + np.random.normal(0, 5.0)) % 360.0
            
            # Sometimes perturb outer rotation
            if np.random.rand() < 0.1:
                new_orot = cur_orot + np.random.normal(0, 2.0)
            else:
                new_orot = cur_orot
            
            new_params = np.concatenate([new_xys.flatten(), [new_orot]])
            
            if _is_valid(new_xys):
                new_L = exact_L(new_params)
            else:
                new_L = exact_L(new_params) + 10.0 * _smooth_pen(_sat_penetrations(new_xys), beta=4.0)
            
            delta = new_L - cur_L
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-10)):
                cur = new_params
                cur_xys = new_xys
                cur_orot = new_orot
                cur_L = new_L
                if new_L < best_L and _is_valid(new_xys):
                    best_L = new_L
                    best_params = new_params.copy()
            
            T *= cool_rate
    
    # Phase 3: High-precision polish
    if best_params is not None and tleft() > 10.0:
        # Multiple polish restarts
        for attempt in range(3):
            if tleft() < 8.0:
                break
            try:
                p_start = best_params.copy()
                if attempt > 0:
                    p_start[:3*N] = _perturb(p_start[:3*N].reshape(N, 3), 0.01).flatten()
                
                res = minimize(objective, p_start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9,
                                        'adaptive': True})
                if _is_valid(res.x[:3*N].reshape(N, 3)):
                    L = exact_L(res.x)
                    if L < best_L:
                        best_L = L
                        best_params = res.x.copy()
            except Exception:
                pass
    
    # Final: try a few more outer rotations with best xys
    if best_params is not None and tleft() > 5.0:
        xys_best = best_params[:3*N].reshape(N, 3)
        for orot in np.arange(0, 60, 5.0):
            if tleft() < 3.0:
                break
            p_test = np.concatenate([xys_best.flatten(), [orot]])
            if _is_valid(xys_best):
                L = exact_L(p_test)
                if L < best_L:
                    best_L = L
                    best_params = p_test.copy()
    
    # Fallback
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        xys_f = np.array(inner)
        L_f = _exact_outer_side(xys_f, 0.0)
        return inner, (0.0, 0.0), float(L_f), 0.0
    
    xys_f = best_params[:3*N].reshape(N, 3)
    orot_f = best_params[3*N]
    
    # Final validation
    if not _is_valid(xys_f):
        # Use fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        xys_f = np.array(inner)
        L_f = _exact_outer_side(xys_f, 0.0)
        return inner, (0.0, 0.0), float(L_f), 0.0
    
    inner = [[float(xys_f[i,0]), float(xys_f[i,1]), float(xys_f[i,2])] for i in range(N)]
    side = float(_exact_outer_side(xys_f, orot_f))
    
    return inner, (0.0, 0.0), side, float(orot_f)

