"""Rewritten program for LJ38 global minimum search."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 38
DEADLINE = 140.0  # Safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

_START_TIME = time.time()

def _lj_energy_and_grad(positions):
    """Compute LJ energy and gradient analytically."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) over pairs (each pair counted twice, so /2)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Gradient: dE/dri = 4 * sum_j (12*r^-13 - 6*r^-7) * (ri - rj) / r
    # = 4 * sum_j (12*inv_r12 - 6*inv_r6) * (ri - rj) * inv_r2 * inv_r2 * inv_r2 * inv_r
    # Actually: d/dr [4*(r^-12 - r^-6)] = 4*(-12*r^-13 + 6*r^-7)
    # dE/dri = sum_j 4*(-12*r^-13 + 6*r^-7) * (ri - rj)/r
    # = sum_j 4*(-12*inv_r12 + 6*inv_r6) * (ri - rj) * inv_r2
    # Wait, let me be more careful:
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dri = 4 * sum_{j != i} (-12 * r_ij^-13 * (ri - rj)/r_ij + 6 * r_ij^-7 * (ri - rj)/r_ij)
    # = 4 * sum_{j != i} (-12 * r_ij^-14 + 6 * r_ij^-8) * (ri - rj)
    # = 4 * sum_{j != i} (-12 * inv_r12 * inv_r2 + 6 * inv_r6 * inv_r2) * (ri - rj)
    
    # Using the full matrix (diagonal is 1.0 so contributes 0 to gradient since diff is 0)
    coeff = 4.0 * (-12.0 * inv_r12 * inv_r2 + 6.0 * inv_r6 * inv_r2)
    # Zero out diagonal to avoid self-interaction
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _lj_energy_only(positions):
    """Compute LJ energy only."""
    e, _ = _lj_energy_and_grad(positions)
    return e


def _check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    return np.min(r2) >= min_dist ** 2


def _make_fcc_truncated_octahedron():
    """
    Create an FCC-based truncated octahedron for LJ38.
    
    The global minimum for LJ38 is reported as an FCC truncated octahedron.
    A truncated octahedron in FCC has layers:
    - Layer 0: 1 atom (center)
    - Layer 1: 6 atoms (octahedral directions)  
    - Layer 2: 12 atoms (cubic directions)
    - Layer 3: 8 atoms (body diagonal directions)
    - Layer 4: 11 atoms (additional surface atoms)
    
    Total: 1 + 6 + 12 + 8 + 11 = 38
    
    Let's construct this from FCC lattice points.
    """
    # FCC lattice: points (i,j,k) where i+j+k is even, scaled so NN distance = 1
    # In FCC with NN=1, the lattice constant a satisfies a/sqrt(2) = 1, so a = sqrt(2)
    # Points are at (i*a, j*a, k*a) where i+j+k is even
    # Actually, standard FCC: atoms at corners and face centers of cubic cell with side a
    # NN distance = a/sqrt(2) = 1 => a = sqrt(2)
    
    a = math.sqrt(2.0)
    
    # Generate FCC points within a radius
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k even (for simple cubic representation)
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda x: x[3])
    
    # Take the 38 closest
    selected = np.array([p[:3] for p in points[:38]], dtype=float)
    
    # Center the structure
    selected -= np.mean(selected, axis=0)
    
    return selected


def _make_icosahedral_structure():
    """Create an icosahedral cluster for LJ38."""
    phi = (1 + math.sqrt(5)) / 2.0
    
    # 12 vertices of regular icosahedron
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    
    # Layer 1: 12 vertices at radius r1
    r1 = 1.12
    layer1 = ico_verts * r1
    
    # Layer 2: 8 cube vertices at radius r2
    cube_verts = np.array([
        [1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
        [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    ], dtype=float)
    cube_verts /= np.linalg.norm(cube_verts[0])
    r2 = 1.65
    layer2 = cube_verts * r2
    
    # Layer 3: 6 octahedral directions at radius r3
    octa_verts = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    r3 = 1.45
    layer3 = octa_verts * r3
    
    # Layer 4: 11 additional points - use edge midpoints of icosahedron
    # Icosahedron has 30 edges; we need 11 more points
    # Let's use some face centers and edge midpoints
    
    # 20 face centers of icosahedron
    # Each face is a triangle of 3 icosahedron vertices
    # For now, let's add points along the directions between icosahedron vertices
    
    # Generate 11 more points by interpolating between pairs of icosahedron vertices
    additional = []
    for i in range(11):
        # Pick pairs of nearby icosahedron vertices
        vi = ico_verts[i % 12]
        vj = ico_verts[(i + 3) % 12]
        mid = (vi + vj) / 2.0
        mid /= np.linalg.norm(mid)
        additional.append(mid * 1.35)
    
    additional = np.array(additional)
    
    # Combine: 1 center + 12 + 8 + 6 + 11 = 38
    center = np.array([[0.0, 0.0, 0.0]])
    all_points = np.vstack([center, layer1, layer2, layer3, additional])
    
    return all_points


def _relax(positions):
    """Relax using L-BFGS with analytic gradient."""
    def obj(x):
        pos = x.reshape(N, 3)
        return _lj_energy_only(pos)
    
    def jac(x):
        pos = x.reshape(N, 3)
        _, grad = _lj_energy_and_grad(pos)
        return grad.flatten()
    
    try:
        result = minimize(
            obj, positions.flatten(), jac=jac, method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def _relax_with_constraint(positions):
    """Relax while ensuring minimum distance is maintained."""
    # Use a penalty-based approach
    def penalized_obj(x):
        pos = x.reshape(N, 3)
        e = _lj_energy_only(pos)
        # Penalty for violating min distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # 0.5^2
            penalty = 1000.0 * (0.25 - min_r2)
            e += penalty
        return e
    
    def penalized_jac(x):
        pos = x.reshape(N, 3)
        _, grad = _lj_energy_and_grad(pos)
        # Approximate penalty gradient (numerical for simplicity)
        eps = 1e-6
        for d in range(N * 3):
            xp = x.copy()
            xp[d] += eps
            fp = penalized_obj(xp)
            xm = x.copy()
            xm[d] -= eps
            fm = penalized_obj(xm)
            grad[d] += (fp - fm) / (2 * eps)
        return grad.flatten()
    
    try:
        result = minimize(
            penalized_obj, positions.flatten(), jac=penalized_jac, method='L-BFGS-B',
            options={'maxiter': 150, 'gtol': 1e-6}
        )
        relaxed = result.x.reshape(N, 3)
        if _check_validity(relaxed):
            return relaxed
        return positions
    except Exception:
        return positions


def _optimize_scale(positions, n_iter=10):
    """Optimize the overall scale to minimize energy."""
    best_scale = 1.0
    best_energy = _lj_energy_only(positions)
    
    for i in range(n_iter):
        # Try scaling up and down
        for scale_factor in [0.95, 0.98, 1.02, 1.05]:
            scaled = positions * scale_factor
            if _check_validity(scaled):
                e = _lj_energy_only(scaled)
                if e < best_energy:
                    best_energy = e
                    best_scale = scale_factor * best_scale
    
    return positions * best_scale


def _basin_hopping(positions, max_iters=50):
    """Simple basin hopping with random perturbations."""
    current = positions.copy()
    current_e = _lj_energy_only(current)
    best = current.copy()
    best_e = current_e
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if _time_left() < 15:
            break
        
        # Perturb
        perturbation = rng.normal(0, 0.1, size=(N, 3))
        perturbed = current + perturbation
        
        if not _check_validity(perturbed):
            continue
        
        # Relax
        relaxed = _relax(perturbed)
        if not _check_validity(relaxed):
            continue
        
        relaxed_e = _lj_energy_only(relaxed)
        
        # Metropolis acceptance
        if relaxed_e < current_e:
            current = relaxed
            current_e = relaxed_e
        else:
            # Accept with probability exp(-(E_new - E_old)/T)
            T = 0.1
            if rng.random() < math.exp(-(relaxed_e - current_e) / T):
                current = relaxed
                current_e = relaxed_e
        
        if current_e < best_e:
            best_e = current_e
            best = current.copy()
    
    return best


def construct():
    """Main entry point: generate candidates, relax, and return the best."""
    global _START_TIME
    _START_TIME = time.time()
    
    best_positions = None
    best_energy = float('inf')
    
    def update_best(pos):
        nonlocal best_positions, best_energy
        if pos is None or not _check_validity(pos):
            return
        e = _lj_energy_only(pos)
        if e < best_energy:
            best_energy = e
            best_positions = pos
    
    # Generate initial candidates
    candidates = []
    
    # 1. FCC truncated octahedron
    try:
        fcc = _make_fcc_truncated_octahedron()
        if _check_validity(fcc):
            candidates.append(fcc)
    except Exception:
        pass
    
    # 2. Icosahedral structure
    try:
        ico = _make_icosahedral_structure()
        if _check_validity(ico):
            candidates.append(ico)
    except Exception:
        pass
    
    # 3. Simple cubic lattice (original seed)
    side = int(np.ceil(N ** (1/3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
    cubic = np.array(points[:N], dtype=float) * 1.12
    if _check_validity(cubic):
        candidates.append(cubic)
    
    # 4. FCC lattice fragment with different scales
    for scale in [1.05, 1.10, 1.15, 1.20]:
        try:
            fcc_frag = _make_fcc_truncated_octahedron() * scale
            if _check_validity(fcc_frag):
                candidates.append(fcc_frag)
        except Exception:
            pass
    
    # 5. Random clusters
    for seed in range(15):
        rng = np.random.default_rng(seed)
        radius = rng.uniform(1.8, 3.0)
        points = []
        attempts = 0
        while len(points) < N and attempts < 5000:
            theta = rng.uniform(0, 2 * math.pi)
            phi_angle = rng.uniform(0, math.pi)
            r = radius * rng.uniform(0, 1) ** (1/3)
            x = r * math.sin(phi_angle) * math.cos(theta)
            y = r * math.sin(phi_angle) * math.sin(theta)
            z = r * math.cos(phi_angle)
            p = np.array([x, y, z])
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            attempts += 1
        if len(points) == N:
            rand_pos = np.array(points, dtype=float)
            if _check_validity(rand_pos):
                candidates.append(rand_pos)
    
    # Relax all candidates and find the best
    for i, cand in enumerate(candidates):
        if _time_left() < 20:
            break
        
        try:
            relaxed = _relax(cand)
            if _check_validity(relaxed):
                # Try scale optimization
                optimized = _optimize_scale(relaxed)
                if _check_validity(optimized):
                    update_best(optimized)
                else:
                    update_best(relaxed)
            else:
                update_best(cand)
        except Exception:
            update_best(cand)
    
    # If we have time, do basin hopping from the best solution
    if best_positions is not None and _time_left() > 30:
        try:
            bh_result = _basin_hopping(best_positions, max_iters=30)
            if _check_validity(bh_result):
                # Final polish
                polished = _relax(bh_result)
                if _check_validity(polished):
                    update_best(polished)
                else:
                    update_best(bh_result)
        except Exception:
            pass
    
    # More random restarts if time permits
    if best_positions is not None and _time_left() > 40:
        for seed in range(50):
            if _time_left() < 20:
                break
            rng = np.random.default_rng(seed + 1000)
            # Perturb the best solution
            perturbation = rng.normal(0, 0.3, size=(N, 3))
            perturbed = best_positions + perturbation
            if not _check_validity(perturbed):
                continue
            try:
                relaxed = _relax(perturbed)
                if _check_validity(relaxed):
                    update_best(relaxed)
            except Exception:
                pass
    
    # Final high-precision polish
    if best_positions is not None and _time_left() > 10:
        try:
            def final_obj(x):
                return _lj_energy_only(x.reshape(N, 3))
            
            def final_jac(x):
                _, grad = _lj_energy_and_grad(x.reshape(N, 3))
                return grad.flatten()
            
            result = minimize(
                final_obj, best_positions.flatten(), jac=final_jac, method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            final_pos = result.x.reshape(N, 3)
            if _check_validity(final_pos):
                update_best(final_pos)
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = cubic
    
    return best_positions
