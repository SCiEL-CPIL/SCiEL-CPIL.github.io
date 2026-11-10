import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_required_S(positions, s_inner=1.0):
    all_vertices = np.vstack([_hex_vertices(p[0], p[1], s_inner, 0.0) for p in positions])
    normals = np.array([[np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)])
    max_dot = np.max(all_vertices @ normals.T)
    return max_dot / (np.sqrt(3) / 2.0)

def _min_separation(positions, s_inner=1.0):
    hex_verts = [_hex_vertices(p[0], p[1], s_inner, 0.0) for p in positions]
    min_sep = np.inf
    for i in range(len(hex_verts)):
        for j in range(i+1, len(hex_verts)):
            v1, v2 = hex_verts[i], hex_verts[j]
            # Compute edge normals for v1
            def _normals(v):
                ns = []
                for k in range(6):
                    e = v[(k+1)%6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    c = v.mean(axis=0)
                    if np.dot(n, (v[k]+v[(k+1)%6])/2 - c) < 0:
                        n = -n
                    ns.append(n)
                return np.array(ns)
            all_n = np.vstack([_normals(v1), _normals(v2)])
            p1 = v1 @ all_n.T
            p2 = v2 @ all_n.T
            gaps = np.maximum(0, np.maximum(p1.min(0) - p2.max(0), p2.min(0) - p1.max(0)))
            min_sep = min(min_sep, np.max(gaps))
    return min_sep

def _obj(x, n=11):
    pos = x.reshape(n, 2)
    S = _compute_required_S(pos)
    sep = _min_separation(pos)
    return S + 1e4 * max(0, 1e-5 - sep)

def hexagon_packing_11():
    n = 11
    pos = np.zeros((n,2))
    pos[0] = [0,0]
    for k in range(6):
        a = np.radians(60*k)
        pos[k+1] = [np.sqrt(3)*1.02*np.cos(a), np.sqrt(3)*1.02*np.sin(a)]
    for k in range(4):
        a = np.radians(60*k + 30)
        pos[k+7] = [2*np.sqrt(3)*1.02*np.cos(a), 2*np.sqrt(3)*1.02*np.sin(a)]
    x0 = pos.ravel()
    res = minimize(_obj, x0, method='Nelder-Mead', options={'maxiter':2000,'xatol':1e-6,'fatol':1e-6})
    pos = res.x.reshape(n,2)
    S = _compute_required_S(pos) + 1e-5
    inner = np.column_stack([pos[:,0], pos[:,1], np.zeros(n)])
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(S)
