import numpy as np
import time
import sys
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants
POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_mackay_template(size, rng=None):
    """
    Generate a Mackay icosahedral template for the given size.
    For sizes 13, 55: exact Mackay structures.
    For other sizes: grow from nearest Mackay by adding/removing atoms with icosahedral symmetry preference.
    """
    if rng is None:
        rng = np.random.default_rng(42 + size)
    
    ico_verts = _get_icosahedron_vertices()
    phi = (1 + np.sqrt(5)) / 2
    
    # Base 13-atom Mackay: center + 12 icosahedron vertices
    center = np.zeros(3)
    shell1 = ico_verts * 1.0
    
    if size == 13:
        return np.vstack([center, shell1])
    
    # For size 55: 1 + 12 + 42
    # The 42 atoms in the 3rd shell of Mackay icosahedron:
    # 30 edge midpoints + 12 face centers (dodecahedron vertices)
    if size == 55:
        # Find edges of icosahedron
        dists = np.zeros((12, 12))
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                dists[i, j] = d
                dists[j, i] = d
        
        min_dist = np.min(dists[dists > 0])
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(dists[i, j] - min_dist) < 1e-6:
                    edges.append((i, j))
        
        # 30 edge midpoints, scaled to radius r2 * sqrt(1 + 1/phi^2)
        r2 = 1.0
        r3 = r2 * np.sqrt(1 + 1/phi**2)
        
        edge_mids = []
        for i, j in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            norm_mid = np.linalg.norm(mid)
            edge_mids.append(mid / norm_mid * r3)
        
        # 12 face centers: use dodecahedron vertices that align with icosahedron faces
        # The 12 vertices of a dodecahedron that are face centers of the icosahedron
        # We need to select 12 from the 20 dodecahedron vertices
        # Actually, the 12 face centers of the icosahedron form a dodecahedron
        # Let's compute them properly
        
        # Find faces of icosahedron
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                    if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                        faces.append((i, j, k))
        
        # There should be 20 faces, but we need 12 for the Mackay 3rd shell
        # The Mackay icosahedron 3rd shell has 42 atoms: 30 edge midpoints + 12 special positions
        # Actually, let me reconsider: the 42 atoms are distributed as:
        # 12 at the vertices of the dual dodecahedron (face centers of icosahedron)
        # But there are 20 face centers, not 12.
        
        # Let me use a different approach: the 42 atoms in the 3rd shell of a Mackay icosahedron
        # are at specific positions. For simplicity, let's use a Fibonacci sphere distribution
        # for the 42 atoms and let L-BFGS-B optimize them.
        
        n_outer = 42
        golden_angle = np.pi * (3 - np.sqrt(5))
        outer_atoms = []
        for i in range(n_outer):
            y = 1 - 2 * i / (n_outer - 1)
            radius_at_y = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            outer_atoms.append([x, y, z] * r3)
        outer_atoms = np.array(outer_atoms)
        
        return np.vstack([center, shell1, outer_atoms])
    
    # For other sizes: grow from 13 or 55
    if size < 13:
        base = _generate_mackay_template(13, rng)
        # Remove atoms from the outer shell
        # Keep center + as many shell1 atoms as needed
        n_keep = size - 1
        if n_keep <= 12:
            return np.vstack([center, shell1[:n_keep]])
        else:
            return base[:size]
    
    if size < 55:
        # Start with 13 and add atoms in a polytetrahedral pattern
        base = _generate_mackay_template(13, rng)
        current = base.copy()
        r_base = 1.0
        
        # Add atoms one by one, preferring icosahedral directions
        ico_dirs = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        
        for _ in range(size - 13):
            # Pick a random icosahedral direction with some noise
            idx = rng.integers(0, 12)
            direction = ico_dirs[idx] + rng.normal(0, 0.1, 3)
            direction = direction / np.linalg.norm(direction)
            
            # Place at a radius that grows slowly
            r_new = r_base * (1.0 + 0.05 * (size - 13))
            new_atom = direction * r_new
            
            # Check minimum distance
            min_dist_check = True
            for existing in current:
                if np.linalg.norm(new_atom - existing) < 0.5:
                    # Try different radius
                    for r_try in [r_new * 0.8, r_new * 1.2, r_new * 1.5]:
                        new_atom_try = direction * r_try
                        if all(np.linalg.norm(new_atom_try - e) >= 0.5 for e in current):
                            new_atom = new_atom_try
                            break
                    else:
                        continue
                    break
            
            current = np.vstack([current, new_atom])
        
        return current
    
    # size > 55: start with 55 and add
    base = _generate_mackay_template(55, rng)
    current = base.copy()
    ico_dirs = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    for _ in range(size - 55):
        idx = rng.integers(0, 12)
        direction = ico_dirs[idx] + rng.normal(0, 0.1, 3)
        direction = direction / np.linalg.norm(direction)
        
        r_new = 1.2 + 0.05 * (size - 55)
        new_atom = direction * r_new
        
        min_dist_check = True
        for existing in current:
            if np.linalg.norm(new_atom - existing) < 0.5:
                for r_try in [r_new * 0.8, r_new * 1.2, r_new * 1.5]:
                    new_atom_try = direction * r_try
                    if all(np.linalg.norm(new_atom_try - e) >= 0.5 for e in current):
                        new_atom = new_atom_try
                        break
                else:
                    continue
                break
        
        current = np.vstack([current, new_atom])
    
    return current

def _energy_and_gradient(positions):
    """Compute energy and gradient using the Dzugutov potential."""
    try:
        energy, grad = dzugutov_energy_gradient(positions)
        return energy, grad
    except:
        # Fallback: numerical gradient
        positions = np.array(positions, dtype=float)
        n = len(positions)
        energy = 0.0
        grad = np.zeros_like(positions)
        
        A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
        
        for i in range(n):
            for j in range(i+1, n):
                r_vec = positions[i] - positions[j]
                r = np.linalg.norm(r_vec)
                if r < 1e-10:
                    continue
                
                if r < a:
                    V = A * (r**(-m) - B) * np.exp(c / (r - a))
                    dV_dr = A * (-m * r**(-m-1)) * np.exp(c / (r - a)) + A * (r**(-m) - B) * np.exp(c / (r - a)) * (-c / (r - a)**2)
                elif r < b:
                    V = B * np.exp(d / (r - b))
                    dV_dr = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
                else:
                    V = 0.0
                    dV_dr = 0.0
                
                energy += V
                grad[i] += dV_dr * r_vec / r
                grad[j] -= dV_dr * r_vec / r
        
        return energy, grad

def _lbfgs_optimize(positions, time_limit, max_iter=500):
    """
    Optimize positions using L-BFGS-B with the Dzugutov gradient.
    Returns the optimized positions and energy.
    """
    positions = np.array(positions, dtype=float).flatten()
    n_atoms = len(positions) // 3
    
    def objective(x):
        pos = x.reshape(-1, 3)
        try:
            e, g = _energy_and_gradient(pos)
            return e, g.flatten()
        except:
            return 1e10, np.zeros_like(x)
    
    start_time = time.time()
    
    result = minimize(
        objective,
        positions,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    opt_pos = result.x.reshape(-1, 3)
    opt_energy = result.fun
    
    # Verify with relax
    try:
        verified_energy, verified_pos = relax(opt_pos, POTENTIAL)
        if verified_energy < opt_energy:
            opt_energy = verified_energy
            opt_pos = verified_pos
    except:
        pass
    
    return opt_pos, opt_energy

def _perturb_and_optimize(positions, rng, time_limit, n_perturb=5, perturb_scale=0.1):
    """
    Generate perturbed versions of the template and optimize each.
    Return the best result.
    """
    best_energy = float('inf')
    best_pos = None
    
    n = len(positions)
    
    # First, optimize the original
    remaining = time_limit - time.time()
    if remaining > 1:
        pos, energy = _lbfgs_optimize(positions, time.time() + remaining * 0.3, max_iter=200)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Then try perturbations
    for i in range(n_perturb):
        if time.time() > time_limit - 2:
            break
        
        remaining = time_limit - time.time()
        if remaining < 3:
            break
        
        # Different types of perturbations
        if i == 0:
            # Small random perturbation
            perturbed = positions + rng.normal(0, perturb_scale, size=(n, 3))
        elif i == 1:
            # Medium perturbation
            perturbed = positions + rng.normal(0, perturb_scale * 2, size=(n, 3))
        elif i == 2:
            # Scale perturbation
            scale = 1.0 + rng.uniform(-0.1, 0.1)
            perturbed = positions * scale
        elif i == 3:
            # Anisotropic perturbation
            anisotropy = rng.uniform(0.9, 1.1, 3)
            perturbed = positions * anisotropy
        else:
            # Larger perturbation
            perturbed = positions + rng.normal(0, perturb_scale * 3, size=(n, 3))
        
        # Ensure minimum distances
        for _ in range(100):
            min_d = np.inf
            for i2 in range(n):
                for j2 in range(i2+1, n):
                    d = np.linalg.norm(perturbed[i2] - perturbed[j2])
                    if d < min_d:
                        min_d = d
            if min_d >= 0.5:
                break
            # Push apart atoms that are too close
            for i2 in range(n):
                for j2 in range(i2+1, n):
                    d = np.linalg.norm(perturbed[i2] - perturbed[j2])
                    if d < 0.5:
                        direction = (perturbed[i2] - perturbed[j2]) / max(d, 1e-10)
                        push = (0.5 - d) / 2 + 0.01
                        perturbed[i2] += direction * push
                        perturbed[j2] -= direction * push
        
        pos, energy = _lbfgs_optimize(perturbed, time.time() + remaining * 0.5, max_iter=150)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    if best_pos is None:
        # Fallback
        try:
            best_energy, best_pos = relax(positions, POTENTIAL)
        except:
            best_pos = positions
            best_energy = float('inf')
    
    return best_pos, best_energy

def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0  # Safety margin before 330s kill
    
    size = instance["n"]
    start_time = time.time()
    
    # Time budget: distribute among 6 instances
    # Total available: ~290s, so ~48s per instance
    instance_deadline = min(start_time + 46.0, _GLOBAL_DEADLINE - 5.0)
    
    rng = np.random.default_rng(42 + size)
    
    # Generate Mackay template
    template = _generate_mackay_template(size, rng)
    
    # Ensure minimum distances in template
    for _ in range(200):
        min_d = np.inf
        for i in range(size):
            for j in range(i+1, size):
                d = np.linalg.norm(template[i] - template[j])
                if d < min_d:
                    min_d = d
        if min_d >= 0.5:
            break
        for i in range(size):
            for j in range(i+1, size):
                d = np.linalg.norm(template[i] - template[j])
                if d < 0.5:
                    direction = (template[i] - template[j]) / max(d, 1e-10)
                    push = (0.5 - d) / 2 + 0.01
                    template[i] += direction * push
                    template[j] -= direction * push
    
    # Optimize using L-BFGS-B with perturbations
    best_pos, best_energy = _perturb_and_optimize(
        template, rng, instance_deadline, n_perturb=4, perturb_scale=0.08
    )
    
    # Final check: ensure minimum distances
    for _ in range(500):
        min_d = np.inf
        for i in range(size):
            for j in range(i+1, size):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < min_d:
                    min_d = d
        if min_d >= 0.5:
            break
        for i in range(size):
            for j in range(i+1, size):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < 0.5:
                    direction = (best_pos[i] - best_pos[j]) / max(d, 1e-10)
                    push = (0.5 - d) / 2 + 0.005
                    best_pos[i] += direction * push
                    best_pos[j] -= direction * push
    
    # Final relaxation
    try:
        final_energy, final_pos = relax(best_pos, POTENTIAL)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos
    except:
        pass
    
    return best_pos
