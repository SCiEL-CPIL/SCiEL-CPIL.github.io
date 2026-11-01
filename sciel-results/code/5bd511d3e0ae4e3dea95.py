"""Contact-graph-driven hexagonal packing: enumerate topologies, solve distance constraints,
optimize container via support-function minimization."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution

N = 11
SQRT3 = math.sqrt(3)
MIN_DIST = SQRT3  # minimum center distance for unit hexagons

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=60.0):
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty(inner, min_dist=MIN_DIST - 0.001):
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def solve_contact_graph(edges, n=N):
    """
    Given a contact graph (list of (i,j) pairs indicating hexagons i and j touch),
    solve for positions using least_squares. Each touching pair has distance = MIN_DIST.
    Non-touching pairs must have distance >= MIN_DIST (handled by penalty).
    Returns positions array (n, 2) or None if no solution found.
    """
    if not edges:
        return None
    
    # Build constraint function: for each edge, distance should be MIN_DIST
    # For non-edges, add soft penalty for distance < MIN_DIST
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    # Total possible pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [(i, j) for i, j in all_pairs if (i, j) not in edge_set]
    
    def residuals(pos):
        pts = pos.reshape(n, 2)
        res = []
        # Hard constraints: touching pairs at exact distance
        for i, j in edges:
            d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
            res.append(d - MIN_DIST)
        # Soft constraints: non-touching pairs should be at least MIN_DIST
        for i, j in non_edges:
            d = np.sqrt((pts[i,0]-pts[j,0])**2 + (pts[i,1]-pts[j,1])**2)
            if d < MIN_DIST:
                res.append((MIN_DIST - d) * 5.0)  # scaled penalty
        return np.array(res)
    
    # Initial guess: place hexagons on a hexagonal lattice perturbed
    np.random.seed(42)
    # Try multiple initializations
    best_pos = None
    best_cost = np.inf
    
    for attempt in range(20):
        if attempt == 0:
            # Lattice initialization
            pos0 = np.zeros(n * 2)
            # Simple grid layout
            for k in range(n):
                q = k % 4
                r = k // 4
                pos0[k*2] = 1.5 * q
                pos0[k*2+1] = SQRT3 * (r + q/2.0)
        else:
            pos0 = np.random.normal(0, 2.0, n * 2)
        
        try:
            result = least_squares(residuals, pos0, method='lm', max_nfev=2000)
            cost = result.cost
            if cost < best_cost:
                best_cost = cost
                best_pos = result.x
        except Exception:
            pass
    
    if best_pos is None:
        return None
    
    pts = best_pos.reshape(n, 2)
    
    # Verify: check that all distances are >= MIN_DIST - tolerance
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
            if d < MIN_DIST - 0.01:
                return None
    
    return pts

def generate_contact_graphs():
    """
    Generate plausible contact graphs for 11 hexagons.
    Strategy: start from known good packings and enumerate variations.
    """
    graphs = []
    
    # Graph 1: Central hexagon + ring of 6 + 4 more (like a larger hex)
    # Center: 0, Ring: 1-6, Extra: 7-10
    g1 = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
          (1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
          (1,7),(2,8),(3,9),(4,10),
          (7,8),(8,9),(9,10)]
    graphs.append(g1)
    
    # Graph 2: Two rows of 5-6
    g2 = [(0,1),(1,2),(2,3),(3,4),(4,5),
          (5,6),(6,7),(7,8),(8,9),(9,10),
          (0,5),(1,6),(2,7),(3,8),(4,9),
          (1,5),(2,6),(3,7),(4,8),(5,9)]
    graphs.append(g2)
    
    # Graph 3: Star pattern - center with 6, then 4 in gaps
    g3 = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
          (1,2),(2,3),(3,4),(4,5),(5,6),(6,1),
          (1,7),(3,8),(5,9),(7,8),(8,9)]
    graphs.append(g3)
    
    # Graph 4: Linear chain with branches
    g4 = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),
          (1,11) if N > 11 else None]
    # Filter None
    g4 = [e for e in g4 if e is not None]
    graphs.append(g4)
    
    # Graph 5: Compact cluster
    g5 = [(0,1),(0,2),(0,3),(1,2),(1,4),(2,5),(3,4),(3,5),(4,5),
          (0,6),(1,7),(2,8),(3,9),(4,10),
          (6,7),(7,8),(8,9),(9,10)]
    graphs.append(g5)
    
    # Graph 6: Honeycomb fragment
    g6 = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,5),(4,6),(5,6),
          (0,7),(1,8),(2,9),(3,10),
          (7,8),(8,9),(9,10)]
    graphs.append(g6)
    
    return graphs

def optimize_container_for_positions(pts, deadline):
    """
    Given fixed positions (n, 2) of hexagon centers (all angle 0),
    optimize outer hexagon center and angle to minimize required side.
    """
    inner = [[float(pts[i,0]), float(pts[i,1]), 0.0] for i in range(N)]
    
    cx0 = float(np.mean(pts[:, 0]))
    cy0 = float(np.mean(pts[:, 1]))
    
    def obj_container(p):
        ox, oy, oa = p[0], p[1], p[2]
        return compute_required_side_smooth(inner, (ox, oy), oa, beta=50.0)
    
    def obj_container_exact(p):
        ox, oy, oa = p[0], p[1], p[2]
        return compute_required_side_fast(inner, (ox, oy), oa)
    
    best_val = np.inf
    best_p = None
    
    # Multi-start over outer angle
    for oa0 in range(0, 60, 10):
        if time.time() > deadline - 10:
            break
        p0 = np.array([cx0, cy0, float(oa0)])
        try:
            r1 = minimize(obj_container, p0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 300})
            r2 = minimize(obj_container_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 500})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    
    if best_p is None:
        return inner, (cx0, cy0), 0.0, compute_required_side_fast(inner, (cx0, cy0), 0.0)
    
    return inner, (float(best_p[0]), float(best_p[1])), float(best_p[2]), best_val

def joint_refine(inner, oc, oa, deadline):
    """
    Full joint refinement: optimize all positions, angles, and container simultaneously.
    Uses differential evolution for global search, then Nelder-Mead polish.
    """
    def obj_joint(params, beta=40.0):
        il = []
        for i in range(N):
            il.append([params[i*3], params[i*3+1], params[i*3+2]])
        ocx, ocy, oaa = params[N*3], params[N*3+1], params[N*3+2]
        side = compute_required_side_smooth(il, (ocx, ocy), oaa, beta=beta)
        pen = overlap_penalty(il)
        return side + 100.0 * pen
    
    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i*3], params[i*3+1], params[i*3+2]])
        ocx, ocy, oaa = params[N*3], params[N*3+1], params[N*3+2]
        side = compute_required_side_fast(il, (ocx, ocy), oaa)
        pen = overlap_penalty(il)
        return side + 500.0 * pen
    
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(inner):
        p0[i*3] = x
        p0[i*3+1] = y
        p0[i*3+2] = a
    p0[N*3] = oc[0]
    p0[N*3+1] = oc[1]
    p0[N*3+2] = oa
    
    # Differential evolution for global exploration
    bounds = []
    for i in range(N):
        bounds.append((inner[i][0]-3, inner[i][0]+3))
        bounds.append((inner[i][1]-3, inner[i][1]+3))
        bounds.append((-30, 30))
    bounds.append((oc[0]-2, oc[0]+2))
    bounds.append((oc[1]-2, oc[1]+2))
    bounds.append((-30, 30))
    
    try:
        de_result = differential_evolution(
            obj_joint, bounds, 
            maxiter=50, popsize=15, 
            tol=1e-6, seed=42,
            update_interval=1, polish=False
        )
        if de_result.fun < obj_joint(p0):
            p0 = de_result.x
    except Exception:
        pass
    
    # Nelder-Mead polish
    try:
        nm_result = minimize(obj_joint_exact, p0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
        p_final = nm_result.x
    except Exception:
        p_final = p0
    
    final_inner = []
    for i in range(N):
        final_inner.append([float(p_final[i*3]), float(p_final[i*3+1]), float(p_final[i*3+2])])
    final_oc = (float(p_final[N*3]), float(p_final[N*3+1]))
    final_oa = float(p_final[N*3+2])
    
    return final_inner, final_oc, final_oa

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    best_side = float('inf')
    best_inner = None
    best_oc = None
    best_oa = 0.0
    
    # Phase 1: Try multiple contact graph topologies
    graphs = generate_contact_graphs()
    
    for gi, graph in enumerate(graphs):
        if time.time() > deadline - 60:
            break
        
        pts = solve_contact_graph(graph)
        if pts is None:
            continue
        
        # Optimize container for this configuration
        inner, oc, oa, side_val = optimize_container_for_positions(pts, deadline)
        
        if check_disjoint(inner, eps=1e-5) and side_val < best_side:
            best_side = side_val
            best_inner = inner
            best_oc = oc
            best_oa = oa
    
    # Phase 2: If we have a good candidate, do joint refinement
    if best_inner is not None and time.time() < deadline - 30:
        refined_inner, refined_oc, refined_oa = joint_refine(
            best_inner, best_oc, best_oa, deadline)
        
        if check_disjoint(refined_inner, eps=1e-5):
            refined_side = compute_required_side_fast(refined_inner, refined_oc, refined_oa)
            if refined_side < best_side:
                best_inner = refined_inner
                best_oc = refined_oc
                best_oa = refined_oa
                best_side = refined_side
    
    # Phase 3: Fallback if nothing worked - use a known good lattice
    if best_inner is None or not check_disjoint(best_inner, eps=1e-5):
        # Standard hexagonal lattice: center + 6 ring + 4 more
        axial = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
                 (2,-1),(1,1),(-2,1),(-1,-1)]
        s0 = 1.0 + 1e-8
        centers = [(1.5*s0*q, SQRT3*s0*(r+q/2.0)) for q,r in axial]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner)/len(best_inner)
        cy0 = sum(c[1] for c in best_inner)/len(best_inner)
        best_oc = (cx0, cy0)
        best_oa = 0.0
        best_side = compute_required_side_fast(best_inner, best_oc, best_oa)
        
        # Quick refinement of fallback
        if time.time() < deadline - 15:
            try:
                def obj_fb(p):
                    il = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
                    return compute_required_side_smooth(il, (p[N*3], p[N*3+1]), p[N*3+2], beta=40.0) + 100*overlap_penalty(il)
                
                p0 = np.zeros(N*3+3)
                for i,(x,y,a) in enumerate(best_inner):
                    p0[i*3]=x; p0[i*3+1]=y; p0[i*3+2]=a
                p0[N*3]=cx0; p0[N*3+1]=cy0; p0[N*3+2]=0.0
                
                res = minimize(obj_fb, p0, method='Nelder-Mead',
                               options={'maxiter':500, 'xatol':1e-9, 'fatol':1e-11, 'maxfev':1000})
                if res.fun < best_side:
                    fi = [[float(res.x[i*3]), float(res.x[i*3+1]), float(res.x[i*3+2])] for i in range(N)]
                    foc = (float(res.x[N*3]), float(res.x[N*3+1]))
                    foa = float(res.x[N*3+2])
                    if check_disjoint(fi, eps=1e-5):
                        fs = compute_required_side_fast(fi, foc, foa)
                        if fs < best_side:
                            best_inner = fi; best_oc = foc; best_oa = foa; best_side = fs
            except Exception:
                pass
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        axial = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),
                 (2,-1),(1,1),(-2,1),(-1,-1)]
        s0 = 1.0 + 1e-8
        centers = [(1.5*s0*q, SQRT3*s0*(r+q/2.0)) for q,r in axial]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner)/len(best_inner)
        cy0 = sum(c[1] for c in best_inner)/len(best_inner)
        best_oc = (cx0, cy0)
        best_oa = 0.0
        best_side = compute_required_side_fast(best_inner, best_oc, best_oa)
    
    return best_inner, best_oc, best_side, best_oa

# EVOLVE-BLOCK-END
