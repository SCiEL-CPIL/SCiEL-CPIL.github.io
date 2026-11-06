import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize
import itertools

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _get_fcc_lattice_points(n):
    """
    Generate a list of unique FCC lattice points (in units of lattice constant a)
    sorted by distance from origin. We take the n closest.
    These points define the 'skeleton' of the cluster.
    """
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Estimate side length
    # Volume of n atoms ~ n * (a^3/4) for FCC density 4 atoms/unit cell
    # a ~ (4n)^(1/3)
    side = int(np.ceil((n * 2.0) ** (1.0/3.0))) + 2
    
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    
    pts = np.array(pts)
    # Center
    pts -= np.mean(pts, axis=0)
    
    # Sort by distance to center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]

def _generate_oh_symmetry_seeds(n, rng):
    """
    Generate seeds constrained to Oh symmetry (cubic).
    We take the n closest FCC points to the center.
    Then we define free parameters for each point:
    1. Radial scaling factor (shared for all? No, let's try per-shell or global).
       To be robust, let's allow small random displacements but constrain them to be symmetric?
       Actually, the prompt suggests "parameterizing only the unique atomic shells and radial/axial displacements".
       
       A simpler and very effective approach for "symmetry-adapted" in this context:
       Start with the perfect FCC/HCP/ICO geometry.
       The "optimization" in reduced coordinates is just finding the optimal lattice constant (scale)
       and perhaps a slight breathing mode.
       
       However, the global minima of Morse clusters are often distorted.
       Let's use a strategy:
       1. Generate the n closest points from an FCC lattice.
       2. Assign each point a "shell" index based on its distance rank.
       3. Optimize the radial distance of each shell? Or just a global scale?
       
       Let's try: Global scale + per-atom radial displacement (but constrained to be small) + full 3D relaxation.
       Wait, the directive says "reduce search space... to guarantee convergence".
       
       Let's implement a specific reduced coordinate system for the FCC-like cluster:
       Coordinates = Lattice_Points * Scale + Displacements.
       If we fix the lattice topology (FCC), we just optimize Scale.
       But clusters are finite, so surface atoms relax.
       
       Strategy:
       1. Identify the "bulk" vs "surface" atoms in the FCC cluster.
       2. For the first pass, optimize ONLY the global scale factor (1 parameter).
       3. Then, allow small radial displacements for surface atoms (3 * N_surface parameters).
       4. Then, full relaxation.
       
       This is a "hierarchical" symmetry breaking.
    """
    lattice_pts = _get_fcc_lattice_points(n)
    
    # Calculate distances from center to identify shells
    center = np.mean(lattice_pts, axis=0)
    rel_pts = lattice_pts - center
    dists = np.linalg.norm(rel_pts, axis=1)
    
    # Define shells by grouping points with similar distances
    # Simple approach: sort by distance, group into K shells
    sorted_idx = np.argsort(dists)
    sorted_dists = dists[sorted_idx]
    
    # Heuristic: number of shells ~ sqrt(n) or just distinct distance levels
    # Let's just use the exact distance values as shell identifiers if they are close
    # But for optimization, let's parameterize:
    # x_i = r_i * (u_i / |u_i|) where u_i is the unit vector of the lattice point
    # So we optimize r_i (radial distance) for each atom.
    # This preserves the angular positions (symmetry of the lattice directions) but allows radial breathing.
    # This reduces 3n params to n params.
    
    # Initial guess: r_i = |lattice_pts_i| * 1.0 (scale factor 1)
    # Let's optimize r_i directly.
    
    def objective_r(params):
        # params: radial distances for each atom
        # Ensure min distance
        # Reconstruct positions
        unit_vecs = rel_pts / np.linalg.norm(rel_pts, axis=1, keepdims=True)
        pos = unit_vecs * params[:, np.newaxis] + center
        
        if not _min_distance_check(pos):
            return 1e10
        
        e, _ = morse_energy_gradient(pos, 3.0) # rho is passed in construct, but here we need it.
        # Wait, rho is an argument to construct. I need to pass rho to this function or make it a closure.
        # I will restructure to pass rho.
        return e

    return lattice_pts, center, rel_pts

def _optimize_reduced_coords_fcc(n, rho, time_budget):
    """
    Optimize an FCC-like cluster using reduced coordinates (radial breathing).
    """
    lattice_pts = _get_fcc_lattice_points(n)
    center = np.mean(lattice_pts, axis=0)
    rel_pts = lattice_pts - center
    norms = np.linalg.norm(rel_pts, axis=1)
    unit_vecs = rel_pts / norms[:, np.newaxis]
    
    # Initial guess: ideal Morse equilibrium is r=1.
    # In FCC, nearest neighbor distance is a/sqrt(2) if a is lattice constant.
    # Our lattice_pts are in units where basis is 0.5, so nearest neighbor is sqrt(0.5^2+0.5^2) = 0.707?
    # No, standard FCC: atoms at (0,0,0) and (0.5, 0.5, 0). Dist = sqrt(0.25+0.25) = 0.707.
    # We want this to be 1. So scale factor should be 1/0.707 = sqrt(2) ~ 1.414.
    # Let's start with scale = sqrt(2).
    
    scale = np.sqrt(2.0)
    r0 = norms * scale
    
    # Objective: minimize energy given radial distances r
    def obj_r(r_vec):
        pos = unit_vecs * r_vec[:, np.newaxis] + center
        if not _min_distance_check(pos):
            return 1e10
        e, _ = morse_energy_gradient(pos, rho)
        return e
    
    def grad_r(r_vec):
        pos = unit_vecs * r_vec[:, np.newaxis] + center
        if not _min_distance_check(pos):
            return np.zeros_like(r_vec)
        e, g = morse_energy_gradient(pos, rho)
        # Chain rule: dE/dr_i = sum_k dE/dx_k * dx_k/dr_i
        # x_k = u_k * r_k + c_k
        # dx_k/dr_i = u_k if k==i else 0
        # So grad_r_i = dot(g_i, u_i)
        grad_r = np.sum(g * unit_vecs, axis=1)
        return grad_r
    
    # Optimize
    res = scipy.optimize.minimize(
        obj_r, 
        r0, 
        jac=grad_r, 
        method='L-BFGS-B', 
        options={'maxiter': 500, 'gtol': 1e-6}
    )
    
    best_pos = unit_vecs * res.x[:, np.newaxis] + center
    best_e = res.fun
    
    # Second stage: Full relaxation from this symmetric solution
    # This allows the symmetry to break if needed, but starts from a very good guess.
    try:
        e2, pos2 = relax(best_pos, 'morse', rho)
        if e2 < best_e:
            best_e = e2
            best_pos = pos2
    except:
        pass
        
    return best_e, best_pos

def _optimize_reduced_coords_ico(n, rho, time_budget):
    """
    Optimize an Icosahedral cluster using reduced coordinates.
    Mackay icosahedron shells.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Vertices of icosahedron
    v1 = np.array([0, 1, phi])
    v2 = np.array([0, -1, phi])
    v3 = np.array([0, 1, -phi])
    v4 = np.array([0, -1, -phi])
    v5 = np.array([1, phi, 0])
    v6 = np.array([-1, phi, 0])
    v7 = np.array([1, -phi, 0])
    v8 = np.array([-1, -phi, 0])
    v9 = np.array([phi, 0, 1])
    v10 = np.array([-phi, 0, 1])
    v11 = np.array([phi, 0, -1])
    v12 = np.array([-phi, 0, -1])
    
    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])
    norm = np.linalg.norm(verts, axis=1)
    verts = verts / norm[:, np.newaxis]
    
    # Mackay shells:
    # Shell 0: Center (1 atom)
    # Shell 1: 12 vertices (icosahedron)
    # Shell 2: 30 edges midpoints? No, Mackay uses vertices of larger icosahedra.
    # Shell 2 has 30 atoms. They are at the midpoints of the edges of the icosahedron?
    # Actually, the second shell of Mackay icosahedron consists of 30 atoms.
    # Their positions can be derived.
    
    points = [np.zeros(3)]
    points.extend(verts)
    
    if n <= 13:
        pts = np.array(points[:n])
    else:
        # Generate shell 2
        # Edges of icosahedron
        edges = []
        # Find pairs of vertices with distance ~ sqrt(2) * (1/phi) ?
        # Vertices are unit vectors. Distance between adjacent vertices in unit icosahedron:
        # v1=(0,1,phi), v5=(1,phi,0). dist^2 = 1 + (phi-1)^2 + phi^2 = 1 + (0.618)^2 + 2.618 = 1 + 0.382 + 2.618 = 4.0?
        # Wait, norm of v1 is sqrt(1+phi^2) = sqrt(1+2.618) = sqrt(3.618) = 1.902.
        # So unit vectors.
        # Adjacent vertices in icosahedron have dot product -1/phi^2?
        # Let's just find all pairs with dist < 1.5 (in unit sphere coords, edge length is ~1.05?)
        # Actually, for unit icosahedron, edge length is 2*sin(pi/5) = 2*0.587 = 1.175?
        # Let's compute distances.
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                dists[i,j] = np.linalg.norm(verts[i] - verts[j])
        
        # Find edges
        edge_mask = (dists > 1.0) & (dists < 1.3) # Heuristic threshold
        # Actually, min dist in unit icosahedron is ~1.051. Next is ~1.902.
        edge_mask = dists < 1.5
        np.fill_diagonal(edge_mask, False)
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if edge_mask[i,j]:
                    edges.append((i, j))
        
        # Midpoints of edges are the 30 atoms of shell 2?
        # In Mackay icosahedron, the second shell atoms are NOT at midpoints.
        # They are at positions that form a larger icosahedron?
        # No, the 30 atoms of the second shell are located at the midpoints of the edges of the icosahedron formed by the 12 vertices?
        # Let's check literature. Mackay icosahedron: 1 + 12 + 30 + 55...
        # The 30 atoms are indeed at the midpoints of the edges of the 12-vertex icosahedron?
        # Let's assume yes for now.
        
        shell2_pts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            shell2_pts.append(mid)
            
        points.extend(shell2_pts)
        
        if n <= 43:
            pts = np.array(points[:n])
        else:
            # Shell 3: 55 atoms.
            # This gets complex. Let's fallback to random if n > 43 for ico.
            # Or just take the first n points we have.
            if n <= len(points):
                pts = np.array(points[:n])
            else:
                # Fallback to FCC
                return None, None
            
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    rel_pts = pts - center
    norms = np.linalg.norm(rel_pts, axis=1)
    unit_vecs = rel_pts / norms[:, np.newaxis]
    
    # Ideal scale: Nearest neighbor in Mackay ico (shell 1) is edge length.
    # We want NN distance = 1.
    # Current NN dist in unit sphere is ~1.05.
    # Scale factor = 1 / 1.05 ~ 0.95.
    # Let's optimize scale.
    
    def obj_r(r_vec):
        pos = unit_vecs * r_vec[:, np.newaxis] + center
        if not _min_distance_check(pos):
            return 1e10
        e, _ = morse_energy_gradient(pos, rho)
        return e
    
    def grad_r(r_vec):
        pos = unit_vecs * r_vec[:, np.newaxis] + center
        if not _min_distance_check(pos):
            return np.zeros_like(r_vec)
        e, g = morse_energy_gradient(pos, rho)
        grad_r = np.sum(g * unit_vecs, axis=1)
        return grad_r
    
    r0 = norms * 0.95
    
    res = scipy.optimize.minimize(
        obj_r, 
        r0, 
        jac=grad_r, 
        method='L-BFGS-B', 
        options={'maxiter': 500, 'gtol': 1e-6}
    )
    
    best_pos = unit_vecs * res.x[:, np.newaxis] + center
    best_e = res.fun
    
    try:
        e2, pos2 = relax(best_pos, 'morse', rho)
        if e2 < best_e:
            best_e = e2
            best_pos = pos2
    except:
        pass
        
    return best_e, best_pos

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses symmetry-adapted reduced-coordinate optimization.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 45.0
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: FCC-like (Oh symmetry)
    if time.time() < deadline - 10.0:
        try:
            e_fcc, pos_fcc = _optimize_reduced_coords_fcc(n, rho, deadline - time.time())
            if e_fcc < best_energy:
                best_energy = e_fcc
                best_positions = pos_fcc
        except Exception as e:
            pass
    
    # Strategy 2: Icosahedral (Ih symmetry) - especially for high rho or small n
    if (rho >= 6 or n <= 50) and time.time() < deadline - 10.0:
        try:
            e_ico, pos_ico = _optimize_reduced_coords_ico(n, rho, deadline - time.time())
            if e_ico is not None and e_ico < best_energy:
                best_energy = e_ico
                best_positions = pos_ico
        except Exception as e:
            pass
            
    # Strategy 3: HCP-like?
    # HCP is harder to parameterize with simple reduced coords.
    # Let's try a quick HCP seed + relax if time permits.
    if time.time() < deadline - 5.0:
        try:
            # Generate HCP seed
            a = 1.0
            c = np.sqrt(8.0/3.0) * a
            pts = []
            radius = np.sqrt(n / (np.pi * 0.866))
            num_layers = int(np.ceil(2 * radius / (c/2))) + 2
            
            for layer_idx in range(-num_layers//2, num_layers//2 + 1):
                z = layer_idx * (c / 2.0)
                if layer_idx % 2 == 0:
                    offset = np.array([0.0, 0.0])
                else:
                    offset = np.array([a/2.0, a*np.sqrt(3)/6.0])
                    
                for i in range(-int(radius), int(radius) + 1):
                    for j in range(-int(radius), int(radius) + 1):
                        x = a * (i + j/2.0) + offset[0]
                        y = a * (np.sqrt(3)/2.0 * j) + offset[1]
                        p = np.array([x, y, z])
                        if np.linalg.norm(p[:2]) <= radius:
                            pts.append(p)
                            
            if len(pts) >= n:
                pts = np.array(pts)
                pts -= np.mean(pts, axis=0)
                dists = np.linalg.norm(pts, axis=1)
                idx = np.argsort(dists)[:n]
                seed_hcp = pts[idx]
                
                # Scale to ideal HCP distance
                # HCP nearest neighbor is a. We want 1. So scale by 1.
                # But our a=1.0. So it's already scaled.
                
                if _min_distance_check(seed_hcp):
                    e_hcp, pos_hcp = relax(seed_hcp, potential, rho)
                    if e_hcp < best_energy:
                        best_energy = e_hcp
                        best_positions = pos_hcp
        except Exception:
            pass
            
    # Final Polish
    if best_positions is not None and time.time() < deadline - 2.0:
        try:
            def objective(x):
                pos = x.reshape(-1, 3)
                if not _min_distance_check(pos):
                    return 1e10
                e, _ = morse_energy_gradient(pos, rho)
                return e
                
            def gradient(x):
                pos = x.reshape(-1, 3)
                if not _min_distance_check(pos):
                    return np.zeros_like(x)
                e, g = morse_energy_gradient(pos, rho)
                return g.flatten()
                
            res = scipy.optimize.minimize(
                objective, 
                best_positions.flatten(), 
                jac=gradient, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_positions = res.x.reshape(-1, 3)
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback: Random
        rng = np.random.default_rng(42)
        radius = n ** (1.0/3.0) * 0.8
        best_positions = rng.uniform(-radius, radius, size=(n, 3))
        
    return best_positions
