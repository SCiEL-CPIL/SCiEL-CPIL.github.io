# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _get_icosahedron_vertices():
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _generate_mackay_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Structure: 1 center + 12 inner shell + 12 outer vertices + 30 outer edge midpoints.
    """
    v = _get_icosahedron_vertices()
    
    # Inner shell (radius ~1.0)
    inner = v * 1.05
    
    # Outer shell (radius ~1.8)
    outer_verts = v * 1.8
    
    # Find edges of the icosahedron to generate midpoints
    dists = np.linalg.norm(v[:, np.newaxis] - v[np.newaxis, :], axis=2)
    min_dist = np.min(dists[dists > 1e-6])
    
    outer_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if dists[i, j] < 1.1 * min_dist:
                mid = (outer_verts[i] + outer_verts[j]) / 2.0
                outer_mids.append(mid)
                
    points = [np.zeros(3)]
    points.extend(list(inner))
    points.extend(list(outer_verts))
    points.extend(outer_mids)
    
    return np.array(points)

def _generate_defect_templates(n):
    """
    Generate diverse templates by removing or adding atoms to the 55-atom Mackay structure.
    For n < 55: Remove atoms from the outer shell. To break symmetry and find different minima,
    we remove atoms in different patterns (e.g., removing a vertex removes its 5 connected midpoints).
    For n > 55: Add atoms near low-coordination sites or on a larger shell.
    """
    base = _generate_mackay_55()
    templates = []
    
    if n == 55:
        return [base]
        
    if n < 55:
        # Strategy: Remove atoms from the outer shell (indices 25 to 54).
        # Outer verts: 13-24. Outer mids: 25-54.
        # We want to generate distinct defects.
        
        # Pattern 1: Remove random midpoints (least impact on symmetry)
        # Pattern 2: Remove a vertex and its connected midpoints (bigger defect)
        
        # Identify connections: Vertex i connects to 5 midpoints.
        # We can pre-compute which midpoints connect to which vertices.
        # Midpoint k is between vertex u and v.
        
        v = _get_icosahedron_vertices()
        dists = np.linalg.norm(v[:, np.newaxis] - v[np.newaxis, :], axis=2)
        min_dist = np.min(dists[dists > 1e-6])
        
        # Map midpoint index (in the list of 30) to vertex indices
        # The order in _generate_mackay_55 is i=0..11, j=i+1..11
        mid_to_verts = {}
        mid_idx = 0
        for i in range(12):
            for j in range(i + 1, 12):
                if dists[i, j] < 1.1 * min_dist:
                    mid_to_verts[mid_idx] = (i, j)
                    mid_idx += 1
        
        # Generate a few specific defects
        # Defect A: Remove 5 midpoints connected to vertex 0 (and vertex 0 itself? No, just mids first to keep count precise)
        # Actually, let's just remove specific sets of indices from the outer shell.
        
        outer_indices = list(range(13, 55))
        
        # Template 1: Remove last k atoms (arbitrary, breaks symmetry)
        k_remove = 55 - n
        t1 = np.delete(base, outer_indices[-k_remove:], axis=0)
        templates.append(t1)
        
        # Template 2: Remove a "star" of midpoints around a vertex
        # Remove vertex 0 (index 13) and its 5 connected midpoints
        # This removes 6 atoms. If k_remove is not 6, we adjust.
        # For now, let's just generate 3 diverse random removals from the outer shell.
        rng = np.random.default_rng(42)
        for _ in range(2):
            remove_idx = rng.choice(outer_indices, size=k_remove, replace=False)
            t = np.delete(base, remove_idx, axis=0)
            templates.append(t)
            
        # Ensure we have at least one template
        if not templates:
            templates.append(base[:n])

    else: # n > 55
        # Add atoms. 
        # Strategy: Add atoms on a sphere of radius ~2.2, or near the surface.
        # To be physically plausible, add them near the "holes" or just randomly on a shell.
        rng = np.random.default_rng(123)
        for _ in range(3): # Generate 3 variants
            extra = []
            for _ in range(n - 55):
                # Random direction
                u = rng.normal(size=3)
                u = u / np.linalg.norm(u)
                # Place at radius 2.0
                extra.append(u * 2.0)
            t = np.vstack([base, np.array(extra)])
            templates.append(t)
            
        if not templates:
            # Fallback
            extra = np.random.default_rng(1).uniform(-2.5, 2.5, (n-55, 3))
            templates.append(np.vstack([base, extra]))

    return templates

def _energy_and_grad(x, n):
    """Flattened energy and gradient for L-BFGS-B."""
    positions = x.reshape(-1, 3)
    energy, grad = dzugutov_energy_gradient(positions)
    
    if grad.shape == (n, 3):
        grad_flat = grad.flatten()
    else:
        grad_flat = grad
        
    return energy, grad_flat

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin (300s limit, 10s buffer)
    
    # 1. Generate diverse templates
    try:
        templates = _generate_defect_templates(n)
    except Exception:
        # Fallback to random compact
        rng = np.random.default_rng(42)
        radius = 1.5 * n**(1.0/3.0)
        init_pos = rng.uniform(-radius, radius, (n, 3))
        templates = [init_pos]
        
    best_energy = np.inf
    best_pos = templates[0].copy()
    
    # 2. Multi-start Optimization
    # We have limited time. Let's allocate time per template.
    # If we have 3-4 templates, we can spend ~60s each.
    num_templates = len(templates)
    time_per_template = 250.0 / num_templates  # Total budget 250s
    
    for i, template in enumerate(templates):
        if time.time() > deadline:
            break
            
        # Center the template
        pos = template - np.mean(template, axis=0)
        
        # Check validity (min distance)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.5:
            continue
            
        x0 = pos.flatten()
        
        # Phase 1: Coarse L-BFGS
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                args=(n,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-3}
            )
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x.reshape(-1, 3)
                
                # Phase 2: Fine L-BFGS
                if time.time() < deadline - 10:
                    res2 = minimize(
                        _energy_and_grad,
                        res.x,
                        args=(n,),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.fun < best_energy:
                        best_energy = res2.fun
                        best_pos = res2.x.reshape(-1, 3)
                        
        except Exception:
            pass
            
        # Phase 3: Try built-in relax if time permits
        if time.time() < deadline - 5:
            try:
                # Use the current best or the template?
                # Let's try relaxing the template directly if it wasn't optimized yet, 
                # or the result of L-BFGS.
                # The provided relax is robust. Let's try it on the best_pos found so far 
                # if it's different, or on the template if L-BFGS failed.
                
                # Actually, let's just try relax on the initial template for this specific seed
                # to see if it finds a different basin.
                test_pos = pos.copy()
                relaxed_pos = relax(test_pos, potential)
                
                e_relax, _ = dzugutov_energy_gradient(relaxed_pos)
                
                if e_relax < best_energy:
                    best_energy = e_relax
                    best_pos = relaxed_pos
                    
            except Exception:
                pass

    # Final Safety Check
    if time.time() < deadline - 2:
        # One last global relaxation of the best candidate
        try:
            final_pos = best_pos.copy()
            relaxed_final = relax(final_pos, potential)
            e_final, _ = dzugutov_energy_gradient(relaxed_final)
            if e_final < best_energy:
                best_pos = relaxed_final
        except Exception:
            pass

    # Verify constraints
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Fallback to a safe random structure if constraints are violated
        rng = np.random.default_rng(999)
        radius = 2.0 * n**(1.0/3.0)
        safe_pos = rng.uniform(-radius, radius, (n, 3))
        safe_pos = safe_pos - np.mean(safe_pos, axis=0)
        try:
            relaxed_safe = relax(safe_pos, potential)
            best_pos = relaxed_safe
        except Exception:
            best_pos = safe_pos

    return best_pos
# EVOLVE-BLOCK-END
