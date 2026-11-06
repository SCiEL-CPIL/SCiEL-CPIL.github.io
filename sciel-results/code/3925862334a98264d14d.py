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

def _generate_icosahedron_fragment():
    """
    Generate a 13-atom icosahedral fragment (1 center + 12 vertices).
    This is the fundamental building block for the Dzugutov potential's polytetrahedral order.
    """
    v = _get_icosahedron_vertices()
    # Scale to optimal bond length. 
    # The Dzugutov potential minimum is around r ~ 1.8-1.9? 
    # Let's check the potential: V(r) = A(r^-m - B)exp(c/(r-a)) for r < a.
    # a=1.87. The term (r^-16 - 1.28) is zero at r = (1/1.28)^(1/16) ~ 0.97.
    # Wait, B is positive. A(r^-m - B). If r is large, r^-m is small, so -B dominates -> negative energy.
    # If r is small, r^-m is large -> positive energy.
    # The minimum is where dV/dr = 0.
    # Let's just use a reasonable guess for the fragment size, say radius 1.0 for the icosahedron shell.
    # The bond length between center and vertex is 1.0.
    # The bond length between vertices is 2*sin(pi/5) ~ 1.17.
    # Let's try to relax this fragment to find its intrinsic minimum size.
    
    frag = np.vstack([np.zeros(3), v]) # 13 atoms
    
    return frag

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix."""
    # Use QR decomposition of a random Gaussian matrix
    M = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    return Q

def _assemble_cluster_fragments(n, rng):
    """
    Assemble a cluster of n atoms using 13-atom icosahedral fragments.
    1. Determine number of fragments.
    2. Place fragments on a coarse lattice or random positions.
    3. Relax the rigid bodies (positions + orientations) to minimize energy.
    4. Return the full atomic coordinates.
    """
    num_frags = n // 13
    remainder = n % 13
    
    # Base fragment
    base_frag = _generate_icosahedron_fragment()
    
    # Initial placement of fragment centers
    # We want them to be close enough to interact but not overlap.
    # Optimal distance between icosahedron centers?
    # If radius is 1, surface is at 1. If we place centers at distance 2.5, there is a gap.
    # Let's place them in a compact cluster.
    
    centers = []
    orientations = []
    
    # Place first fragment at origin
    centers.append(np.zeros(3))
    orientations.append(np.eye(3))
    
    # Place subsequent fragments
    for i in range(1, num_frags):
        # Random direction from previous center? Or just random position in a shell?
        # Let's try to pack them.
        # Simple heuristic: place on a sphere of radius ~ 2.5 * i^(1/3)?
        # Or just random positions in a box and let relaxation fix it.
        
        # Random position in a sphere of radius 5
        u = rng.normal(size=3)
        u = u / np.linalg.norm(u)
        r = rng.uniform(2.0, 5.0)
        centers.append(u * r)
        
        # Random orientation
        orientations.append(_random_rotation_matrix(rng))
        
    # Handle remainder atoms
    # If remainder > 0, we can add a partial fragment or just random atoms.
    # Adding a partial fragment (e.g., just the center atom of the next icosahedron) might be unstable.
    # Better: Add random atoms near the surface of the existing cluster.
    
    all_positions = []
    all_rotations = []
    
    for i in range(num_frags):
        R = orientations[i]
        pos = base_frag @ R.T + centers[i]
        all_positions.append(pos)
        
    if remainder > 0:
        # Add 'remainder' atoms.
        # Let's add them as a "cap" or just random atoms near the center of mass.
        # To be safe, let's just add them as random atoms in a small sphere around the centroid.
        centroid = np.mean(np.vstack(all_positions), axis=0)
        for _ in range(remainder):
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            r = rng.uniform(1.0, 2.0)
            all_positions.append(centroid + u * r)
            
    # Flatten all positions
    # We need to optimize the positions of ALL atoms?
    # The "Fragment-Merger" idea implies optimizing the rigid body parameters first.
    # However, implementing a rigid body constraint in scipy.optimize is complex.
    # Alternative: 
    # 1. Initialize with rigid fragments.
    # 2. Run a "soft" relaxation where we penalize deviations from the fragment shape?
    # 3. Or simply: Initialize with fragments, then run standard L-BFGS on all atoms.
    # The key is the INITIAL CONFIGURATION. The fragment assembly provides a better starting point than random.
    
    # Let's refine the initial configuration slightly before returning.
    # We can do a few steps of gradient descent on the full system to "unstick" overlapping fragments.
    
    full_pos = np.vstack(all_positions)
    
    return full_pos

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
    
    # 1. Generate diverse initial configurations using Fragment-Merger
    rng = np.random.default_rng(42)
    
    # We will generate 3-4 different assemblies
    num_starts = 4
    time_per_start = 250.0 / num_starts
    
    best_energy = np.inf
    best_pos = None
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        try:
            # Generate a new assembly
            pos0 = _assemble_cluster_fragments(n, rng)
            
            # Center it
            pos0 = pos0 - np.mean(pos0, axis=0)
            
            # Check validity (min distance)
            diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # If there are overlaps, we might need to nudge them apart.
            # For now, if min dist < 0.5, we skip this start or try to fix it.
            if np.min(dists) < 0.5:
                # Try to fix by random perturbation
                pos0 += rng.normal(0, 0.1, pos0.shape)
                # Re-center
                pos0 = pos0 - np.mean(pos0, axis=0)
                # Re-check
                diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < 0.5:
                    continue # Skip if still invalid
            
            x0 = pos0.flatten()
            
            # Phase 1: Coarse L-BFGS
            # Use a loose tolerance to find the general basin
            res = minimize(
                _energy_and_grad,
                x0,
                args=(n,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-3}
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
                        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.fun < best_energy:
                        best_energy = res2.fun
                        best_pos = res2.x.reshape(-1, 3)
                        
        except Exception as e:
            # Ignore errors and try next start
            pass
            
        # Phase 3: Try built-in relax if time permits
        if time.time() < deadline - 5 and best_pos is not None:
            try:
                # Use the provided relax function on the current best
                # Note: relax might be expensive, so only do it once or twice
                if i == num_starts - 1: # Only on the last iteration to save time? Or on every improvement?
                    # Let's try relaxing the best candidate found so far
                    test_pos = best_pos.copy()
                    relaxed_pos = relax(test_pos, potential)
                    
                    e_relax, _ = dzugutov_energy_gradient(relaxed_pos)
                    
                    if e_relax < best_energy:
                        best_energy = e_relax
                        best_pos = relaxed_pos
                        
            except Exception:
                pass

    # Final Safety Check
    if best_pos is None:
        # Fallback to random if all fragment assemblies failed
        rng = np.random.default_rng(999)
        radius = 1.5 * n**(1.0/3.0)
        best_pos = rng.uniform(-radius, radius, (n, 3))
        best_pos = best_pos - np.mean(best_pos, axis=0)
        
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
