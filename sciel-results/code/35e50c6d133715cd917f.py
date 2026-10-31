import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def _hex_verts_local():
    # Unit hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    return np.column_stack([np.cos(angles), np.sin(angles)])

_LOCAL_VERTS = _hex_verts_local()

def _rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])

def _hex_verts(center, angle_deg):
    a = np.deg2rad(angle_deg)
    R = _rot(a)
    return _LOCAL_VERTS @ R.T + center

def _edge_normals_local():
    # For each edge from v[k] to v[(k+1)%6], compute outward normal
    verts = _LOCAL_VERTS
    normals = np.zeros((6, 2))
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k+1) % 6]
        edge = p2 - p1
        # Outward normal (90 deg CCW from edge for CCW polygon)
        # For CCW polygon, outward is right of edge direction
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals[k] = n
    return normals

_LOCAL_NORMALS = _edge_normals_local()
_LOCAL_CONSTANTS = np.dot(_LOCAL_VERTS, _LOCAL_NORMALS.T).T  # c_k = n_k . v_k

def _point_hex_signed_dist(px, py, cx, cy, angle_deg):
    # Returns min of n_k . (p - v_k) over k. Positive means inside (far from all edges)
    a = np.deg2rad(angle_deg)
    R = _rot(a)
    # Transform p to local frame
    p_local = R.T @ np.array([px - cx, py - cy])
    dists = np.dot(_LOCAL_NORMALS, p_local) - _LOCAL_CONSTANTS
    return np.min(dists)

def _hex_overlap_penalty(verts_a, cx_b, cy_b, angle_b):
    # For each vertex of A, compute distance to B. Penalty for vertices inside B
    a = np.deg2rad(angle_b)
    R = _rot(a)
    # Transform all vertices of A to B's local frame
    rel = verts_a - np.array([cx_b, cy_b])
    p_locals = rel @ R
    dists = np.dot(p_locals, _LOCAL_NORMALS.T) - _LOCAL_CONSTANTS  # (6, 6): dist to each edge
    min_dists = np.min(dists, axis=1)  # (6,)
    neg = np.maximum(0, -min_dists)
    return np.sum(neg ** 2)

def _hex_container_penalty(verts, cx, cy, side, angle):
    # For each vertex of inner hex, check if inside container.
    # Container has side length `side` (circumradius = side for regular hexagon)
    # Actually for a regular hexagon with side s, circumradius = s.
    # Let me verify: side length s, vertices at distance s from center. Yes.
    # So container is a regular hexagon with circumradius = side.
    # Transform vertices to container's local frame
    a = np.deg2rad(angle)
    R = _rot(a)
    rel = verts - np.array([cx, cy])
    p_locals = rel @ R
    # Container edge normals and constants (for circumradius = side)
    # Scale local normals/constants by side
    c_cont = _LOCAL_CONSTANTS * side
    n_cont = _LOCAL_NORMALS  # normals unchanged
    dists = np.dot(p_locals, n_cont.T) - c_cont
    min_dists = np.min(dists, axis=1)
    neg = np.maximum(0, -min_dists)
    return np.sum(neg ** 2)

def _objective(params, side):
    # params: 3*N = [x1,y1,th1, x2,y2,th2, ...]
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    total_pen = 0.0
    # Compute all vertices
    all_verts = []
    for i in range(N):
        v = _hex_verts(centers[i], angles[i])
        all_verts.append(v)
    
    # Containment penalty
    for i in range(N):
        total_pen += _hex_container_penalty(all_verts[i], 0.0, 0.0, side, 0.0)
    
    # Pairwise overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            total_pen += _hex_overlap_penalty(all_verts[i], centers[j][0], centers[j][1], angles[j])
            total_pen += _hex_overlap_penalty(all_verts[j], centers[i][0], centers[i][1], angles[i])
    
    return total_pen

def _initial_guess():
    # Start with a rough arrangement: central hex + 6 around + 4 more
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    # Center
    centers[0] = [0, 0]
    # 6 around center at distance 2 (touching)
    for k in range(1, 7):
        a = np.deg2rad(60 * (k-1))
        centers[k] = [2*np.cos(a), 2*np.sin(a)]
    # 4 more at distance 4
    for k in range(7, 11):
        a = np.deg2rad(60 * (k-7) + 30)
        centers[k] = [4*np.cos(a), 4*np.sin(a)]
    return np.concatenate([centers.flatten(), angles])

def construct():
    deadline = time.time() + 110.0  # 110s limit
    best_side = None
    best_params = None
    
    # Binary search on side length
    lo, hi = 2.0, 6.0
    
    # Try to find minimum side
    # For a fixed side, check if packing is feasible
    def feasible(side):
        if time.time() > deadline - 10:
            return None
        best_pen = float('inf')
        best_p = None
        # Multiple starts
        starts = [_initial_guess()]
        # Add some random perturbations
        for _ in range(3):
            p = _initial_guess().copy()
            p += np.random.randn(len(p)) * 0.5
            starts.append(p)
        
        for p0 in starts:
            if time.time() > deadline - 10:
                break
            res = minimize(_objective, p0, args=(side,), method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_pen:
                best_pen = res.fun
                best_p = res.x
        
        if best_p is None:
            return None
        return best_pen < 1e-4, best_p
    
    # Binary search
    for _ in range(8):  # 8 iterations of binary search
        if time.time() > deadline - 30:
            break
        mid = (lo + hi) / 2
        result = feasible(mid)
        if result is None:
            break
        ok, params = result
        if ok:
            hi = mid
            best_side = mid
            best_params = params
        else:
            lo = mid
    
    if best_side is not None and best_params is not None:
        centers = best_params[:2*N].reshape(N, 2)
        angles = best_params[2*N:]
        inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
        return inner, (0.0, 0.0), float(best_side), 0.0
    else:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
