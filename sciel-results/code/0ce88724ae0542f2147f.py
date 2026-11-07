# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _icosahedron_points():
    """Return unit vectors to vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    pts = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _cuboctahedron_points():
    """Return unit vectors to vertices of a cuboctahedron (12 vertices)."""
    # Permutations of (0, 1, 1)
    pts = []
    for perm in [(0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                 (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                 (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]:
        pts.append(perm)
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _build_symmetric_cluster(n, kind, rng):
    """
    Build a cluster using symmetric shells.
    For 12-6 and 9-6: Icosahedral shells (12, 42, 92, ...)
    For 10-8: Cuboctahedral/FCC shells (12, 30, 55, 91, ...)
    
    Returns:
        params: array of shell radii (and possibly center)
        shell_defs: list of (unit_vectors, shell_index)
    """
    if kind in ['12-6', '9-6']:
        # Icosahedral clusters: 1, 13, 55, 147...
        # Shells: 12, 42, 92...
        shell_sizes = [12, 42, 92, 140] # Approx sizes
        base_pts = _icosahedron_points()
        
        # Generate points for multiple shells
        # Shell 1: 12 pts (icosahedron)
        # Shell 2: 42 pts (3x icosahedron + 3x face centers? No, just scaled icosahedron + extra)
        # For simplicity, we use concentric icosahedra with varying density.
        # Actually, for 55 atoms: 1 center + 12 shell1 + 42 shell2.
        # For 38 atoms: 1 center + 12 shell1 + 25 shell2 (partial)
        
        # Strategy: Use icosahedral symmetry.
        # We define shells by scaling the icosahedron vertices.
        # To get more points, we add "face centers" or "edge centers" of the icosahedron.
        
        # Let's generate a pool of symmetric points:
        # 12 vertices (V)
        # 30 edge midpoints (E)
        # 20 face centers (F)
        
        # Edge midpoints of icosahedron:
        # Icosahedron has 30 edges.
        # We can compute edges by finding pairs of vertices with distance < threshold.
        v = base_pts
        d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        # Nearest neighbors in icosahedron are 5 per vertex.
        # Distance between V is approx 1.051 (normalized).
        # Let's find edges:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if d[i, j] < 1.2: # Threshold for edge
                    edges.append((i, j))
        
        edge_mids = []
        for i, j in edges:
            mid = (v[i] + v[j]) / 2
            mid /= np.linalg.norm(mid)
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids)
        
        # Face centers:
        # 20 faces. Triangles of vertices.
        # We can find faces by looking at triplets of vertices that form a triangle.
        # Or simpler: icosahedron faces correspond to (1,1,phi) type permutations?
        # Let's just generate more points by scaling V and adding noise, but that breaks symmetry.
        
        # Better approach for symmetric optimization:
        # Use the 12 V points, 30 E points, 20 F points.
        # Total 62 points per "layer".
        
        # We need n points.
        # Start with center (0,0,0).
        # Then add shells.
        
        # For 12-6, 9-6:
        # n=38: 1 + 12 + 25. 25 is not symmetric. 
        # Maybe 1 + 12 + 12 + 13? 
        # Let's just use a "spherical" symmetric approach with icosahedral shells.
        # Shell 1: 12 V
        # Shell 2: 30 E (scaled)
        # Shell 3: 20 F (scaled)
        
        # Let's define a list of candidate symmetric points:
        # 0: Center
        # 1-12: V
        # 13-42: E
        # 43-62: F
        
        # We will select the first n points from this list, ordered by radius.
        # But we want to optimize the radii of the shells.
        
        # Let's group by type:
        # Group 0: Center (1 pt)
        # Group 1: V (12 pts)
        # Group 2: E (30 pts)
        # Group 3: F (20 pts)
        
        # For n=38:
        # Use Group 0 (1), Group 1 (12), and 25 from Group 2? 
        # This breaks symmetry if we pick arbitrary 25.
        # Instead, let's allow the "radius" of each group to vary, and if we need partial groups, 
        # we assume the cluster is "full" up to some shell.
        
        # Actually, for n=38, 55, 75, 80, 59, 79, 55...
        # 55 is a magic number (1+12+42). 
        # 38 is not. 
        # 75 is not.
        
        # Alternative: Use a "spherical cap" or just concentric spheres with icosahedral symmetry.
        # We will define the cluster as a set of points on icosahedral orbits.
        # Orbit 1: 12 V
        # Orbit 2: 30 E
        # Orbit 3: 20 F
        # Orbit 4: 12 V (scaled)
        # ...
        
        # We will optimize the scale of each orbit.
        # If n is not a sum of orbit sizes, we take the first n points from the sorted list of orbits by energy?
        # No, we must return exactly n points.
        
        # Let's use a simpler symmetric construction:
        # 1 Center
        # K shells of 12 Icosahedron vertices.
        # If n > 1 + 12*K, we add the next shell of 30 Edge midpoints?
        
        # Let's just generate a large set of symmetric points and pick the n closest to the center, 
        # then optimize the radial distance of each *shell* (group of points at similar initial radius).
        
        all_pts = [np.zeros(3)] # Center
        all_types = [0]
        
        # Shell 1: V
        for p in v:
            all_pts.append(p)
            all_types.append(1)
            
        # Shell 2: E
        for p in edge_mids:
            all_pts.append(p)
            all_types.append(2)
            
        # Shell 3: F (approximate)
        # Face centers of icosahedron are along the normals of the faces.
        # We can generate them by taking the cross product of edges?
        # Or just use the 20 directions of a dodecahedron (dual of icosahedron).
        # Dodecahedron vertices:
        # (±1, ±1, ±1), (0, ±1/φ, ±φ), (±1/φ, ±φ, 0), (±φ, 0, ±1/φ)
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1/phi
        dode_pts = []
        for sx in [1, -1]:
            for sy in [1, -1]:
                for sz in [1, -1]:
                    dode_pts.append([sx, sy, sz])
        for sx in [1, -1]:
            for sy in [1, -1]:
                dode_pts.append([0, sx*inv_phi, sy*phi])
        for sx in [1, -1]:
            for sy in [1, -1]:
                dode_pts.append([sx*inv_phi, sy*phi, 0])
        for sx in [1, -1]:
            for sy in [1, -1]:
                dode_pts.append([sx*phi, 0, sy*inv_phi])
        
        dode_pts = np.array(dode_pts)
        dode_pts /= np.linalg.norm(dode_pts, axis=1, keepdims=True)
        
        for p in dode_pts:
            all_pts.append(p)
            all_types.append(3)
            
        # Repeat shells with scaling
        # Shell 4: V * 1.5
        for p in v:
            all_pts.append(p * 1.5)
            all_types.append(4)
        # Shell 5: E * 1.5
        for p in edge_mids:
            all_pts.append(p * 1.5)
            all_types.append(5)
        # Shell 6: F * 1.5
        for p in dode_pts:
            all_pts.append(p * 1.5)
            all_types.append(6)
        # Shell 7: V * 2.0
        for p in v:
            all_pts.append(p * 2.0)
            all_types.append(7)
        # Shell 8: E * 2.0
        for p in edge_mids:
            all_pts.append(p * 2.0)
            all_types.append(8)
        # Shell 9: F * 2.0
        for p in dode_pts:
            all_pts.append(p * 2.0)
            all_types.append(9)
            
        all_pts = np.array(all_pts)
        all_types = np.array(all_types)
        
        # Sort by radius
        radii = np.linalg.norm(all_pts, axis=1)
        sort_idx = np.argsort(radii)
        all_pts = all_pts[sort_idx]
        all_types = all_types[sort_idx]
        
        # Select first n points
        pts = all_pts[:n]
        types = all_types[:n]
        
        # Define shells: unique types
        unique_types = np.unique(types)
        # Map type to shell index
        type_to_shell = {t: i for i, t in enumerate(unique_types)}
        shell_indices = np.array([type_to_shell[t] for t in types])
        
        # Initial radii for each shell
        # Group points by shell and take mean radius
        shell_radii = []
        for s in unique_types:
            mask = types == s
            if np.any(mask):
                r = np.mean(np.linalg.norm(all_pts[mask], axis=1))
                shell_radii.append(r)
            else:
                shell_radii.append(1.0)
        shell_radii = np.array(shell_radii)
        
        # Store unit vectors for each point in the shell
        # We need to be able to reconstruct pts from shell_radii
        # For each point i, its direction is all_pts[i] / ||all_pts[i]||
        # But if we change the radius, we just scale the direction.
        # Wait, the directions are fixed by the symmetry.
        # So we optimize shell_radii.
        
        return shell_radii, all_pts[:n] / np.linalg.norm(all_pts[:n], axis=1, keepdims=True), shell_indices

    else:
        # 10-8: Cuboctahedral / FCC
        # Similar logic but with FCC symmetry
        # 12 vertices of cuboctahedron
        # 24 edge midpoints?
        # Let's just use the icosahedral one for now as a fallback, or generate FCC shells.
        # FCC shells: 12, 30, 55, 91...
        # Let's use the same generic symmetric shell builder but with FCC points.
        
        base_pts = _cuboctahedron_points()
        # Generate more shells
        all_pts = [np.zeros(3)]
        all_types = [0]
        
        for p in base_pts:
            all_pts.append(p)
            all_types.append(1)
            
        # Next shell: 30 points? 
        # FCC (111) plane has 30 points in the next shell?
        # Let's just scale the base points and add noise for now, or use a denser set.
        # For 10-8, 55 atoms is magic (1+12+42? No, FCC is 1+12+42? No, FCC is 1+12+30+... no)
        # FCC magic numbers: 1, 13, 55, 147.
        # 13 = 1 + 12.
        # 55 = 1 + 12 + 42. (42 is not FCC shell, FCC shells are 12, 30, 55? No.)
        # FCC coordination: 12.
        # Shell 1: 12.
        # Shell 2: 30? (Face centers of cube? No.)
        # Actually, 55-atom FCC cluster is a cuboctahedron of radius 2?
        # Let's just use the icosahedral builder for 10-8 as well, it's a valid starting guess.
        
        # Re-use the icosahedral logic for simplicity, it's a good general start.
        pass

def _relax_symmetric(args):
    """
    Optimize shell radii for a symmetric cluster.
    """
    shell_radii, unit_vectors, shell_indices, kind = args
    
    n = len(unit_vectors)
    
    def map_params(radii):
        # radii: array of shell radii
        # Construct full positions
        pts = np.zeros((n, 3))
        for i in range(n):
            s = shell_indices[i]
            pts[i] = unit_vectors[i] * radii[s]
        return pts
    
    def energy_and_grad(radii):
        pts = map_params(radii)
        # Check validity
        if np.any(~np.isfinite(pts)):
            return np.inf, None
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.35:
            return np.inf, None
        
        e, grad_pts = sutton_chen_energy_gradient(pts, kind)
        if not np.isfinite(e):
            return np.inf, None
            
        # Chain rule: dE/dradii = sum_i (dE/dp_i) * (dp_i/dradii)
        # dp_i/dradii_s = unit_vectors[i] if shell_indices[i] == s, else 0
        grad_radii = np.zeros_like(radii)
        for s in range(len(radii)):
            mask = shell_indices == s
            if np.any(mask):
                # Sum gradients of all atoms in shell s, projected onto their unit vectors
                # dE/dp_i is a vector. dE/dr_s = sum_{i in s} dE/dp_i . u_i
                grad_s = np.sum(np.sum(grad_pts[mask] * unit_vectors[mask], axis=1))
                grad_radii[s] = grad_s
                
        return e, grad_radii
        
    # Initial radii
    x0 = shell_radii.copy()
    
    # Bounds: radii must be positive
    bounds = [(0.5, 5.0) for _ in range(len(x0))]
    
    try:
        res = minimize(
            energy_and_grad, x0, jac=True, method='L-BFGS-B', 
            bounds=bounds, options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success and np.isfinite(res.fun):
            best_radii = res.x
            best_e = res.fun
        else:
            # Fallback: try a few random perturbations of x0
            best_e = np.inf
            best_radii = x0
            for _ in range(5):
                pert = x0 + np.random.normal(0, 0.1, size=x0.shape)
                pert = np.clip(pert, 0.5, 5.0)
                res_p = minimize(
                    energy_and_grad, pert, jac=True, method='L-BFGS-B', 
                    bounds=bounds, options={'maxiter': 20, 'ftol': 1e-10}
                )
                if res_p.success and res_p.fun < best_e:
                    best_e = res_p.fun
                    best_radii = res_p.x
                    
        pts = map_params(best_radii)
        # Final relaxation with full DOF to remove any symmetry constraints that might be suboptimal
        # This is a "symmetry breaking" step
        e_final, p_final = relax(pts, 'sutton_chen', kind)
        
        if np.isfinite(e_final) and np.all(np.isfinite(p_final)):
            d = np.sqrt(np.sum((p_final[:, None, :] - p_final[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() >= 0.35:
                return (e_final, p_final)
                
    except Exception:
        pass
        
    return (np.inf, None)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    rng = np.random.default_rng(20240613 ^ (n * 101) ^ (hash(kind) % 100000))
    deadline = time.time() + 285.0
    
    # Generate symmetric starts
    starts = []
    
    # 1. Icosahedral symmetric start
    shell_radii, unit_vectors, shell_indices = _build_symmetric_cluster(n, kind, rng)
    starts.append((shell_radii, unit_vectors, shell_indices, kind))
    
    # 2. FCC symmetric start (if 10-8)
    if kind == '10-8':
        # We can try a different symmetry or just perturb the icosahedral one
        # Let's add a few random perturbations of the symmetric start
        for _ in range(3):
            pert_radii = shell_radii + rng.normal(0, 0.1, size=shell_radii.shape)
            starts.append((pert_radii, unit_vectors, shell_indices, kind))
            
    # 3. Random starts (fallback)
    for _ in range(5):
        radius = 0.72 * n ** (1.0 / 3.0) * 0.8
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 20000:
            p = rng.uniform(-radius, radius, 3)
            attempts += 1
            if np.linalg.norm(p) > radius:
                continue
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.4:
                    ok = False
                    break
            if ok:
                pts.append(p)
        if len(pts) == n:
            pts = np.array(pts)
            # Convert to symmetric format? No, just run standard relax on these.
            # We'll handle them separately.
            starts.append(('random', pts, kind))
            
    best_e = np.inf
    best_x = None
    
    # Parallel relaxation
    tasks = []
    for start in starts:
        if isinstance(start, tuple) and len(start) == 4 and not isinstance(start[0], str):
            tasks.append(start)
        else:
            # Random start
            tasks.append(('random', start[1], kind))
            
    def _worker(task):
        if task[0] == 'random':
            _, pts, k = task
            try:
                e, p = relax(pts, 'sutton_chen', k)
                if np.isfinite(e) and np.all(np.isfinite(p)):
                    d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
                    np.fill_diagonal(d, np.inf)
                    if d.min() >= 0.35:
                        return (e, p)
            except Exception:
                pass
            return (np.inf, None)
        else:
            return _relax_symmetric(task)
            
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            if time.time() > deadline:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_x = p
            except Exception:
                pass
                
    # If we have time, do a final polish on the best
    if best_x is not None and time.time() < deadline - 10.0:
        try:
            e, p = relax(best_x, 'sutton_chen', kind)
            if np.isfinite(e) and np.all(np.isfinite(p)):
                d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
                np.fill_diagonal(d, np.inf)
                if d.min() >= 0.35 and e < best_e:
                    best_e = e
                    best_x = p
        except Exception:
            pass

    if best_x is None:
        # Fallback
        return np.random.normal(0, 1, (n, 3))
        
    return best_x
# EVOLVE-BLOCK-END
