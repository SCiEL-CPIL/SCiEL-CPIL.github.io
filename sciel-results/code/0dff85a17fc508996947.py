"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

START_TIME = time.time()
TIME_LIMIT = 115.0

def _lj_energy_and_grad(x):
    """Compute Lennard-Jones energy and gradient efficiently."""
    pos = x.reshape(-1, 3)
    n = pos.shape[0]
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_r2 = 1.0 / dists_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    coeff = 24.0 * inv_r8 - 48.0 * inv_r14
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def _relax(pos, max_iter=200):
    """Relax a structure using L-BFGS-B with time check."""
    if time.time() - START_TIME > TIME_LIMIT:
        return pos
    
    pos = pos.copy()
    try:
        result = minimize(
            _lj_energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        relaxed = result.x.reshape(-1, 3)
        
        dists = pdist(relaxed, 'euclidean')
        if np.min(dists) < 0.5:
            return pos
        return relaxed
    except Exception:
        return pos

def _valid(pos):
    """Check if positions are valid (min distance >= 0.5)."""
    try:
        dists = pdist(pos, 'euclidean')
        return np.min(dists) >= 0.5
    except:
        return False

def _make_mackay_icosahedron():
    """
    Construct a proper 55-atom Mackay icosahedron.
    Center: 1 atom
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    Total: 55 atoms
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Find edges (pairs at minimum distance)
    dists = pdist(verts)
    min_dist = np.min(dists)
    
    edges = []
    idx = 0
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[idx] - min_dist) < 1e-6:
                edges.append((i, j))
            idx += 1
    
    # 30 edge midpoints (normalized to unit sphere)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # 20 face centers (triangles formed by 3 mutually adjacent vertices)
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                if (abs(d_ij - min_dist) < 1e-6 and 
                    abs(d_ik - min_dist) < 1e-6 and 
                    abs(d_jk - min_dist) < 1e-6):
                    center = verts[i] + verts[j] + verts[k]
                    center = center / np.linalg.norm(center)
                    face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Select 12 face centers (dodecahedron vertices)
    # For Mackay icosahedron, we need specific 12 of the 20 face centers
    # These form a dodecahedron
    # Use a simple selection: take every other one based on index
    selected_face_centers = face_centers[::2][:12]
    if len(selected_face_centers) < 12:
        # Fallback: take first 12
        selected_face_centers = face_centers[:12]
    
    # LJ optimized radii for Mackay icosahedron
    # Shell 1 radius ~1.12, Shell 2 radius ~2.24
    R1 = 1.12
    R2 = 2.24
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * R1
    shell2 = np.vstack([edge_mids * R2, selected_face_centers * R2])
    
    all_points = np.vstack([center, shell1, shell2])
    
    # Verify we have exactly 55 atoms
    assert all_points.shape[0] == 55, f"Expected 55 atoms, got {all_points.shape[0]}"
    
    return all_points

def _make_fcc_seed():
    """Generate an FCC fragment seed with 55 atoms."""
    pts = []
    # Generate FCC lattice points
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    fcc_pts = np.array(pts)
    
    # Scale so nearest neighbor distance ~1.12
    fcc_pts *= 1.12 / np.sqrt(0.5)
    
    # Select 55 closest to center
    dists = np.linalg.norm(fcc_pts, axis=1)
    idxs = np.argsort(dists)[:55]
    return fcc_pts[idxs]

def _make_bcc_seed():
    """Generate a BCC fragment seed with 55 atoms."""
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
    bcc_pts = np.array(pts)
    
    # Scale so nearest neighbor distance ~1.12
    fcc_pts = bcc_pts * 1.12 / np.sqrt(0.75)
    
    dists = np.linalg.norm(fcc_pts, axis=1)
    idxs = np.argsort(dists)[:55]
    return fcc_pts[idxs]

def _make_random_seed(scale=1.5):
    """Generate a valid random seed."""
    for _ in range(100):
        pts = np.random.randn(N, 3) * scale
        if _valid(pts):
            return pts
    # Fallback: spread out more
    pts = np.random.randn(N, 3) * 3.0
    return pts

def _distort_seed(base, noise=0.1):
    """Create a distorted version of a seed."""
    return base + np.random.randn(N, 3) * noise

def construct():
    """Genetic Algorithm for LJ cluster optimization."""
    np.random.seed(42)
    
    pop_size = 30
    
    # Generate diverse initial population
    population = []
    energies = []
    
    # Add key seeds
    seeds = [
        _make_mackay_icosahedron(),
        _distort_seed(_make_mackay_icosahedron(), 0.05),
        _distort_seed(_make_mackay_icosahedron(), 0.15),
        _make_fcc_seed(),
        _distort_seed(_make_fcc_seed(), 0.1),
        _make_bcc_seed(),
        _distort_seed(_make_bcc_seed(), 0.1),
        _make_random_seed(1.0),
        _make_random_seed(1.5),
        _make_random_seed(2.0),
    ]
    
    for s in seeds:
        population.append(s.copy())
        e, _ = _lj_energy_and_grad(s.flatten())
        energies.append(e)
    
    # Fill rest with random
    while len(population) < pop_size:
        s = _make_random_seed(np.random.uniform(0.8, 2.0))
        population.append(s)
        e, _ = _lj_energy_and_grad(s.flatten())
        energies.append(e)
    
    # Relax initial population
    for i in range(len(population)):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        population[i] = _relax(population[i])
        if _valid(population[i]):
            energies[i], _ = _lj_energy_and_grad(population[i].flatten())
    
    best_idx = np.argmin(energies)
    best_energy = energies[best_idx]
    best_pos = population[best_idx].copy()
    
    max_generations = 80
    gen = 0
    stagnation_count = 0
    current_sigma = 0.3
    
    while gen < max_generations:
        if time.time() - START_TIME > TIME_LIMIT - 25:
            break
            
        gen += 1
        
        # Sort by energy
        sorted_indices = np.argsort(energies)
        sorted_pop = [population[i].copy() for i in sorted_indices]
        sorted_energies = [energies[i] for i in sorted_indices]
        
        # Elitism: Keep top 4
        elitism_count = 4
        new_population = [sorted_pop[i] for i in range(elitism_count)]
        new_energies = [sorted_energies[i] for i in range(elitism_count)]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            
            # Tournament selection (size 5)
            def tournament_select():
                idxs = np.random.choice(len(population), 5, replace=False)
                best_i = idxs[np.argmin([energies[i] for i in idxs])]
                return population[best_i].copy()
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover: exchange random subsets of atoms
            if np.random.rand() < 0.75:
                # Uniform crossover at atom level
                mask = np.random.rand(N) < 0.5
                child = np.where(mask[:, np.newaxis], parent1, parent2)
            else:
                # Keep better parent
                e1, _ = _lj_energy_and_grad(parent1.flatten())
                e2, _ = _lj_energy_and_grad(parent2.flatten())
                child = parent1.copy() if e1 <= e2 else parent2.copy()
            
            # Adaptive mutation
            if stagnation_count > 8:
                current_sigma = min(0.8, current_sigma * 1.15)
            else:
                current_sigma = max(0.05, current_sigma * 0.97)
                
            child += np.random.randn(N, 3) * current_sigma
            
            # Relax
            relaxed_child = _relax(child, max_iter=150)
            if _valid(relaxed_child):
                new_population.append(relaxed_child)
                e, _ = _lj_energy_and_grad(relaxed_child.flatten())
                new_energies.append(e)
            else:
                new_population.append(child)
                e, _ = _lj_energy_and_grad(child.flatten())
                new_energies.append(e)
                
        population = new_population
        energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy:
            best_energy = energies[gen_best_idx]
            best_pos = population[gen_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
            
        # Stagnation: inject diversity
        if stagnation_count > 12:
            num_replace = max(5, pop_size // 6)
            for _ in range(num_replace):
                idx = np.random.randint(len(population))
                # Mix of random and distorted seeds
                if np.random.rand() < 0.5:
                    population[idx] = _make_random_seed(np.random.uniform(0.8, 2.5))
                else:
                    base_seed = np.random.choice([
                        _make_mackay_icosahedron(),
                        _make_fcc_seed(),
                        _make_bcc_seed()
                    ])
                    population[idx] = _distort_seed(base_seed, np.random.uniform(0.1, 0.4))
                population[idx] = _relax(population[idx], max_iter=100)
                if _valid(population[idx]):
                    energies[idx], _ = _lj_energy_and_grad(population[idx].flatten())
            stagnation_count = 0
            
    # Final polish: relax top candidates with more iterations
    if time.time() - START_TIME < TIME_LIMIT - 5:
        sorted_indices = np.argsort(energies)
        top_candidates = [population[i].copy() for i in sorted_indices[:8]]
        
        for i, cand in enumerate(top_candidates):
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            polished = _relax(cand, max_iter=400)
            if _valid(polished):
                e, _ = _lj_energy_and_grad(polished.flatten())
                if e < best_energy:
                    best_energy = e
                    best_pos = polished
    
    # Final relaxation on absolute best
    if time.time() - START_TIME < TIME_LIMIT:
        final_pos = _relax(best_pos, max_iter=600)
        if _valid(final_pos):
            best_pos = final_pos
            
    return best_pos
# EVOLVE-BLOCK-END
