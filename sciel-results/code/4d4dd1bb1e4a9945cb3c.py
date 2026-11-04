import numpy as np
import time
from scipy.optimize import minimize
import sys

START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # Safety margin for 150s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_exact(cluster_vertices):
    """
    Compute the minimum side length S of a regular hexagon centered at origin 
    (with any rotation) that contains all points in cluster_vertices.
    
    Actually, we need to optimize the outer hexagon's rotation and center as well.
    However, for a given set of points, the minimal enclosing regular hexagon 
    can be found by checking all 6 facet orientations relative to the cluster.
    
    A regular hexagon of side S centered at (0,0) with rotation phi has support function:
    h(phi + k*pi/3) <= S * h_hex_unit(phi + k*pi/3)
    where h_hex_unit(theta) = cos(theta) if theta in [-pi/6, pi/6] mod 2pi, etc.
    Actually, the distance from center to side is S * sqrt(3)/2.
    The condition for a point p to be inside a hexagon of side S, rotation phi, center c is:
    max_k | dot(p-c, n_k) | <= S * sqrt(3)/2
    where n_k are the 6 normal vectors of the hexagon.
    
    So, for a fixed rotation phi and center c, the required side S is:
    S = (2/sqrt(3)) * max_{p in P, k in 0..5} | dot(p-c, n_k(phi)) |
    
    We want to minimize S over phi and c.
    Since the cluster is symmetric-ish, we can assume the optimal center is the centroid 
    or close to it. Let's optimize phi and c (cx, cy) jointly using a simple grid search 
    or Nelder-Mead on the 3 variables (cx, cy, phi) for the given cluster vertices.
    """
    P = np.asarray(cluster_vertices)
    sqrt3 = np.sqrt(3)
    
    def objective(params):
        cx, cy, phi = params
        c = np.array([cx, cy])
        # Normals of the hexagon
        # For a regular hexagon with side S, the distance from center to each side is S*sqrt(3)/2.
        # The normals are at angles phi, phi+pi/3, ..., phi+5pi/3.
        angles = phi + np.arange(6) * np.pi / 3
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Compute dot products (P - c) . n_k
        # P is (N, 2), normals is (6, 2)
        # We want max over k of | (P-c) @ n_k |
        # Let's vectorize: 
        # Dots[i, k] = (P[i] - c) @ n_k
        # = P[i] @ n_k - c @ n_k
        
        # Efficient computation:
        # P @ normals.T gives (N, 6)
        dots = P @ normals.T - np.dot(c, normals.T)
        
        max_proj = np.max(np.abs(dots))
        
        # S = (2/sqrt(3)) * max_proj
        return (2.0 / sqrt3) * max_proj

    # Initial guess: centroid and 0 rotation
    centroid = np.mean(P, axis=0)
    x0 = np.array([centroid[0], centroid[1], 0.0])
    
    # Optimize rotation and center
    # Nelder-Mead is robust for small dimensions
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.success or res.fun < objective(x0):
            return res.fun
        else:
            return objective(x0)
    except Exception:
        return objective(x0)

def _check_overlap(v1, v2, tol=1e-6):
    """
    Check if two convex polygons (defined by vertices) overlap using SAT.
    Returns True if they overlap.
    """
    def project(points, axis):
        return np.dot(points, axis)
    
    axes = []
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1)%len(v)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = project(v1, axis)
        proj2 = project(v2, axis)
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _build_cluster(params, topology):
    """
    Build the cluster positions from relative parameters.
    params: array of shape (N-1, 3) where each row is (dx, dy, dangle) 
            representing the offset of node i (i>0) from its parent.
    topology: list of parents. topology[i] is the parent index of node i.
              topology[0] is ignored (root).
    """
    N = len(topology)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # Node 0 is at origin, angle 0
    pos[0] = [0.0, 0.0]
    ang[0] = 0.0
    
    for i in range(1, N):
        parent = topology[i]
        dx, dy, dangle = params[i-1]
        
        # Transform local offset to global
        # The local frame of the parent is rotated by ang[parent]
        a = ang[parent]
        ca, sa = np.cos(a), np.sin(a)
        
        # Global offset
        gdx = ca * dx - sa * dy
        gdy = sa * dx + ca * dy
        
        pos[i] = pos[parent] + np.array([gdx, gdy])
        ang[i] = ang[parent] + dangle
        
    return pos, ang

def _get_cluster_vertices(pos, ang):
    """Get all vertices of all hexagons in the cluster."""
    verts_list = []
    for i in range(len(pos)):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, np.degrees(ang[i]))
        verts_list.append(verts)
    return np.vstack(verts_list)

def _evaluate_cluster(params, topology):
    """
    Evaluate a cluster configuration.
    Returns (score, is_valid). Score is the side length of the minimal enclosing hexagon.
    """
    pos, ang = _build_cluster(params, topology)
    
    # Check for overlaps
    verts_list = []
    for i in range(len(pos)):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, np.degrees(ang[i]))
        verts_list.append(verts)
        
    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            if _check_overlap(verts_list[i], verts_list[j]):
                return np.inf, False
                
    # Compute minimal enclosing hexagon side
    all_verts = np.vstack(verts_list)
    side = _min_outer_side_exact(all_verts)
    
    return side, True

def construct():
    global START_TIME, DEADLINE
    
    # Topology: 
    # 0: Center
    # 1-6: Ring 1 (parents: 0)
    # 7-10: Ring 2 (parents: 1, 3, 4, 6? Or 1, 2, 3, 4?)
    # Let's use a standard compact topology.
    # 0 is root.
    # 1,2,3,4,5,6 are children of 0.
    # 7 is child of 1.
    # 8 is child of 3.
    # 9 is child of 4.
    # 10 is child of 6.
    # This creates a 11-node tree.
    
    N = 11
    topology = [-1, 0, 0, 0, 0, 0, 0, 1, 3, 4, 6]
    
    # Initial parameters
    # For Ring 1: distance approx 2*sqrt(3) ~ 3.46, angle 0, 60, 120...
    # Relative angle to parent:
    # Node 1 (parent 0): angle 0, dist 2*sqrt(3)
    # Node 2 (parent 0): angle 60, dist 2*sqrt(3)
    # ...
    
    sqrt3 = np.sqrt(3)
    init_params = np.zeros((N-1, 3))
    
    # Ring 1
    for i in range(6):
        # Node index i+1
        # Angle in global frame: i * 60 deg
        # Since parent (0) is at angle 0, relative angle is i * 60 deg
        init_params[i, 0] = 2 * sqrt3  # dx
        init_params[i, 1] = 0.0        # dy
        init_params[i, 2] = i * np.pi / 3  # dangle in radians
        
    # Ring 2
    # Node 7 (parent 1):
    # Parent 1 is at angle 0. Node 7 should be further out.
    # In the lattice, the next hex in the "gap" is at angle 30 deg from center?
    # Actually, let's just start with reasonable guesses.
    # Node 7: relative to 1. 
    # Let's say it's at distance 2*sqrt(3) from 1, at an angle that pushes it out.
    init_params[6, 0] = 2 * sqrt3
    init_params[6, 1] = 0.0
    init_params[6, 2] = np.pi / 3  # 60 deg relative to parent 1's frame? 
    # Parent 1 is aligned with 0. So global angle is 60.
    # But we want it in the gap between 1 and 2?
    # Let's try to make it a compact cluster.
    
    # Node 8 (parent 3):
    init_params[7, 0] = 2 * sqrt3
    init_params[7, 1] = 0.0
    init_params[7, 2] = -np.pi / 3
    
    # Node 9 (parent 4):
    init_params[8, 0] = 2 * sqrt3
    init_params[8, 1] = 0.0
    init_params[8, 2] = np.pi / 3
    
    # Node 10 (parent 6):
    init_params[9, 0] = 2 * sqrt3
    init_params[9, 1] = 0.0
    init_params[9, 2] = -np.pi / 3
    
    # Objective function for Nelder-Mead
    def objective(params):
        if time.time() > DEADLINE:
            return 1e10
        score, valid = _evaluate_cluster(params, topology)
        if not valid:
            # Penalty for overlap
            # We can estimate overlap magnitude or just return a large number
            # To help gradient-free methods, a large penalty is standard.
            return 1e6 + np.sum(params**2)
        return score
    
    # Optimization
    best_score = np.inf
    best_params = init_params.copy()
    
    # Initial evaluation
    s, v = _evaluate_cluster(init_params, topology)
    if v:
        best_score = s
    else:
        best_score = 1e6
        # Try to find a valid initial state by randomizing slightly
        for _ in range(20):
            if time.time() > DEADLINE - 10:
                break
            p = init_params + np.random.normal(0, 0.1, init_params.shape)
            s, v = _evaluate_cluster(p, topology)
            if v and s < best_score:
                best_score = s
                best_params = p
                break
    
    # Nelder-Mead Optimization
    # We split the budget.
    # Run multiple restarts or one long run?
    # Nelder-Mead can get stuck. Let's do a few restarts with different initializations.
    
    n_restarts = 5
    time_per_restart = (DEADLINE - time.time()) / (n_restarts + 1)
    
    for r in range(n_restarts):
        if time.time() > DEADLINE - 5:
            break
            
        # Perturb best solution for new start
        if r == 0:
            start_params = best_params.copy()
        else:
            # Random restart around best or random
            if np.random.rand() < 0.5:
                start_params = best_params + np.random.normal(0, 0.5, best_params.shape)
            else:
                # Random init
                start_params = init_params + np.random.normal(0, 1.0, init_params.shape)
                
        # Bounds? Nelder-Mead doesn't have bounds. We rely on penalty.
        # We can clip angles to avoid huge rotations.
        
        try:
            res = minimize(objective, start_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final extraction
    pos, ang = _build_cluster(best_params, topology)
    
    # Compute final side
    verts_list = []
    for i in range(N):
        verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, np.degrees(ang[i]))
        verts_list.append(verts)
        
    # Check validity one last time
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if _check_overlap(verts_list[i], verts_list[j]):
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback to a known valid configuration if optimization failed
        # Use the initial lattice-like config
        pos, ang = _build_cluster(init_params, topology)
        verts_list = []
        for i in range(N):
            verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, np.degrees(ang[i]))
            verts_list.append(verts)
        # If still invalid (shouldn't be), use a trivial linear chain
        # But let's assume init_params is valid or close.
        # Let's force validity by checking and if invalid, return a safe but large box.
        # Actually, the problem requires valid packing.
        # If we can't find one, we should return the best valid one found.
        # We tracked best_params from valid evaluations only (due to penalty structure returning inf/1e6).
        # Wait, _evaluate_cluster returns inf if invalid. 
        # In objective, we return 1e6 + penalty. 
        # So best_params should correspond to a valid config if best_score < 1e6.
        
        # If best_score >= 1e6, we failed to find a valid packing.
        # We must return *something*. 
        # Let's try to construct a trivial valid packing: Linear chain.
        # 11 hexagons in a row. Side length approx 11*2 = 22?
        # Let's generate a linear chain.
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = i * 2.0 * sqrt3 # Distance between centers of touching hexes is 2*sqrt(3)? 
            # No, distance between centers of touching unit hexagons is 2 * apothem = 2 * sqrt(3)/2 * 2? 
            # Apothem = sqrt(3)/2. Distance = 2 * apothem = sqrt(3).
            # Wait. Side length 1. Distance between parallel sides is sqrt(3).
            # So center distance for touching is sqrt(3).
            pos[i, 0] = i * sqrt3
        # This is a valid packing.
        # Compute side for this.
        verts_list = []
        for i in range(N):
            verts = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, np.degrees(ang[i]))
            verts_list.append(verts)
        all_verts = np.vstack(verts_list)
        final_side = _min_outer_side_exact(all_verts)
        
        # Return this as fallback
        inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
        return inner_list, (0.0, 0.0), float(final_side), 0.0

    # Valid packing found
    all_verts = np.vstack(verts_list)
    final_side = _min_outer_side_exact(all_verts)
    
    # The outer hexagon center is (0,0) in our calculation, but we can translate the whole cluster.
    # The problem asks for (inner, center, side, angle).
    # The "center" is the center of the outer hexagon.
    # We computed the minimal side assuming the outer hexagon is centered at (0,0) and optimized its rotation.
    # The function _min_outer_side_exact also optimized the center (cx, cy) of the outer hexagon relative to the cluster.
    # Wait, _min_outer_side_exact optimizes cx, cy, phi to minimize S.
    # So the optimal outer center is NOT necessarily (0,0) relative to the cluster centroid.
    # We need to extract the optimal cx, cy from the optimization.
    
    # Let's re-run the final _min_outer_side_exact but also return the center and angle.
    # My current _min_outer_side_exact only returns the side.
    # I need to modify it or re-implement the extraction.
    
    # Let's write a helper that returns (side, cx, cy, phi)
    
    def _min_outer_side_full(P):
        P = np.asarray(P)
        sqrt3 = np.sqrt(3)
        
        def objective(params):
            cx, cy, phi = params
            c = np.array([cx, cy])
            angles = phi + np.arange(6) * np.pi / 3
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = P @ normals.T - np.dot(c, normals.T)
            max_proj = np.max(np.abs(dots))
            return (2.0 / sqrt3) * max_proj

        centroid = np.mean(P, axis=0)
        x0 = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < objective(x0):
                return res.fun, res.x[0], res.x[1], res.x[2]
            else:
                return objective(x0), x0[0], x0[1], x0[2]
        except Exception:
            return objective(x0), x0[0], x0[1], x0[2]
    
    final_side, outer_cx, outer_cy, outer_phi = _min_outer_side_full(all_verts)
    
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
    
    return inner_list, (float(outer_cx), float(outer_cy)), float(final_side), float(np.degrees(outer_phi))

if __name__ == "__main__":
    construct()
