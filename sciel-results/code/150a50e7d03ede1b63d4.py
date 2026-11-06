"""Morse cluster global minima search with diverse inits and multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

# Global time tracking
_start_time = None
_time_budget_per_instance = {}

def _init_time_tracking():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    _init_time_tracking()
    return time.time() - _start_time

def _remaining_time():
    _init_time_tracking()
    return max(0.0, 320.0 - _elapsed())  # 10s safety margin

def _morse_energy(pos, rho):
    """Compute Morse energy."""
    n = len(pos)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            x = rho * (1.0 - r)
            e = np.exp(x)
            energy += e * (e - 2.0)
    return energy

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like cluster."""
    points = []
    # FCC lattice vectors
    a = spacing * 0.7071  # half diagonal for close packing
    basis = [
        [0, 0, 0], [0, a, a], [a, 0, a], [a, a, 0],
        [a, a, a], [0, 0, 2*a], [0, 2*a, 0], [2*a, 0, 0],
        [2*a, 2*a, 0], [2*a, 0, 2*a], [0, 2*a, 2*a],
        [2*a, 2*a, 2*a], [a, a, 2*a], [a, 2*a, a], [2*a, a, a]
    ]
    # Add more FCC points in expanding shells
    for shell in range(1, 20):
        for i in range(-shell, shell+1):
            for j in range(-shell, shell+1):
                for k in range(-shell, shell+1):
                    if (i + j + k) % 2 == 0:
                        p = np.array([i, j, k]) * a
                        if np.linalg.norm(p) <= spacing * n**(1/3) * 1.2:
                            points.append(p)
        if len(points) >= n:
            break
    # If not enough, add random points
    rng = np.random.default_rng(42)
    radius = spacing * n**(1/3) * 0.8
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    points = np.array(points[:n])
    # Center
    points -= points.mean(axis=0)
    return points

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like cluster."""
    points = []
    a = spacing * 0.5774  # BCC lattice constant for close packing
    for shell in range(1, 20):
        for i in range(-shell, shell+1):
            for j in range(-shell, shell+1):
                for k in range(-shell, shell+1):
                    p = np.array([i, j, k]) * a
                    if np.linalg.norm(p) <= spacing * n**(1/3) * 1.2:
                        points.append(p)
                    # BCC body center
                    if i % 2 == 0 and j % 2 == 0 and k % 2 == 0:
                        p_center = (np.array([i, j, k]) + 0.5) * a
                        if np.linalg.norm(p_center) <= spacing * n**(1/3) * 1.2:
                            points.append(p_center)
        if len(points) >= n:
            break
    rng = np.random.default_rng(123)
    radius = spacing * n**(1/3) * 0.8
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points

def _build_hcp_cluster(n, spacing=1.0):
    """Build an HCP-like cluster."""
    points = []
    a = spacing * 0.7071
    c = a * np.sqrt(8.0/3.0)
    # HCP layers
    for layer in range(-10, 11):
        z = layer * c / 2.0
        if abs(z) > spacing * n**(1/3) * 1.2:
            continue
        if layer % 2 == 0:
            # A layer
            for i in range(-10, 11):
                for j in range(-10, 11):
                    x = i * a + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                    p = np.array([x, y, z])
                    if np.linalg.norm(p) <= spacing * n**(1/3) * 1.2:
                        points.append(p)
        else:
            # B layer
            for i in range(-10, 11):
                for j in range(-10, 11):
                    x = i * a + a / 2.0 + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0 + a / 2.0
                    p = np.array([x, y, z])
                    if np.linalg.norm(p) <= spacing * n**(1/3) * 1.2:
                        points.append(p)
    rng = np.random.default_rng(777)
    radius = spacing * n**(1/3) * 0.8
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points

def _build_icosahedral_cluster(n, spacing=1.0):
    """Build an icosahedral shell cluster."""
    points = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, -phi, 0], [1, -phi, 0], [-1, phi, 0]
    ])
    # Deduplicate
    ico_verts = np.array([v for i, v in enumerate(ico_verts) if all(not np.allclose(v, ico_verts[j]) for j in range(i))])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add shells
    for shell_r in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        r = shell_r * spacing
        for v in ico_verts:
            points.append(v * r)
        if len(points) >= n:
            break
    # Add center
    points.insert(0, np.array([0.0, 0.0, 0.0]))
    
    rng = np.random.default_rng(999)
    radius = spacing * n**(1/3) * 0.8
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points

def _build_random_compact(n, spacing=1.0, seed=0):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n**(1/3) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    return points

def _optimize_from_init(pos, rho, max_time):
    """Multi-stage optimization from a starting position."""
    n = len(pos)
    best_energy = _morse_energy(pos, rho)
    best_pos = pos.copy()
    
    t0 = time.time()
    
    # Stage 1: Quick relax
    try:
        energy, pos_relaxed = relax(pos, 'morse', rho)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos_relaxed.copy()
    except:
        pass
    
    if time.time() - t0 > max_time * 0.3:
        return best_energy, best_pos
    
    # Stage 2: L-BFGS-B
    try:
        def fun_and_grad(x):
            pos = x.reshape(n, 3)
            e = _morse_energy(pos, rho)
            grad, _ = morse_energy_gradient(pos, rho)
            return e, grad.flatten()
        
        res = minimize(fun_and_grad, best_pos.flatten(), method='L-BFGS-B', 
                       jac=True, options={'maxiter': 500, 'gtol': 1e-8})
        if res.success or res.fun < best_energy:
            pos_new = res.x.reshape(n, 3)
            e_new = _morse_energy(pos_new, rho)
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
    except:
        pass
    
    if time.time() - t0 > max_time * 0.7:
        return best_energy, best_pos
    
    # Stage 3: BFGS polish
    try:
        def fun_and_grad_bfgs(x):
            pos = x.reshape(n, 3)
            e = _morse_energy(pos, rho)
            grad, _ = morse_energy_gradient(pos, rho)
            return e, grad.flatten()
        
        res = minimize(fun_and_grad_bfgs, best_pos.flatten(), method='BFGS',
                       jac=True, options={'maxiter': 200, 'gtol': 1e-10})
        if res.success or res.fun < best_energy:
            pos_new = res.x.reshape(n, 3)
            e_new = _morse_energy(pos_new, rho)
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
    except:
        pass
    
    return best_energy, best_pos

def construct(instance):
    """Find global minimum for Morse cluster."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    remaining = _remaining_time()
    if remaining <= 5.0:
        # Not enough time, return a quick valid structure
        pos = _build_random_compact(n, 1.0, seed=n)
        return pos
    
    # Allocate time for this instance
    time_for_this = min(remaining, 50.0)  # Cap per instance
    
    # Generate diverse initial structures
    inits = []
    inits.append(_build_fcc_cluster(n, 1.0))
    inits.append(_build_bcc_cluster(n, 1.0))
    inits.append(_build_hcp_cluster(n, 1.0))
    inits.append(_build_icosahedral_cluster(n, 1.0))
    inits.append(_build_random_compact(n, 1.0, seed=n))
    inits.append(_build_random_compact(n, 1.0, seed=n+1000))
    
    # Add perturbed versions of best lattice structures
    rng = np.random.default_rng(n + 5000)
    for base in inits[:3]:
        perturbed = base + rng.normal(0, 0.1, base.shape)
        inits.append(perturbed)
    
    best_energy = np.inf
    best_pos = inits[0]
    
    t_start = time.time()
    per_init_time = time_for_this / len(inits)
    
    for i, init_pos in enumerate(inits):
        if time.time() - t_start > time_for_this - 2.0:
            break
        
        # Ensure minimum distance constraint
        # Check and fix any close pairs
        pos = init_pos.copy()
        for _ in range(10):  # Simple distance fixing
            fixed = False
            for a in range(n):
                for b in range(a+1, n):
                    d = np.linalg.norm(pos[a] - pos[b])
                    if d < 0.51:
                        direction = (pos[a] - pos[b]) / (d + 1e-10)
                        shift = (0.51 - d) / 2.0 + 0.01
                        pos[a] += direction * shift
                        pos[b] -= direction * shift
                        fixed = True
            if not fixed:
                break
        
        energy, optimized_pos = _optimize_from_init(pos, rho, per_init_time)
        if energy < best_energy:
            best_energy = energy
            best_pos = optimized_pos
    
    # Final verification and distance enforcement
    final_pos = best_pos.copy()
    for _ in range(20):
        fixed = False
        for a in range(n):
            for b in range(a+1, n):
                d = np.linalg.norm(final_pos[a] - final_pos[b])
                if d < 0.501:
                    direction = (final_pos[a] - final_pos[b]) / (d + 1e-10)
                    shift = (0.501 - d) / 2.0 + 0.005
                    final_pos[a] += direction * shift
                    final_pos[b] -= direction * shift
                    fixed = True
        if not fixed:
            break
    
    return final_pos
# EVOLVE-BLOCK-END
