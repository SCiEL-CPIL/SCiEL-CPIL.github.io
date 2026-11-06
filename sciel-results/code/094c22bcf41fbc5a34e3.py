# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

def _dzugutov_energy(positions):
    """Compute Dzugutov energy for a set of positions."""
    n = len(positions)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 0.5:
                return 1e10
            if r < a:
                total += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total += B * np.exp(d / (r - b))
    return total

def _dzugutov_gradient(positions):
    """Compute Dzugutov gradient for a set of positions."""
    n = len(positions)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 0.5:
                return np.ones_like(positions) * 1e10
            if r < a:
                # dV/dr for r < a
                dVdr = A * (-m * r ** (-m - 1)) * np.exp(c / (r - a)) + \
                       A * (r ** (-m) - B) * np.exp(c / (r - a)) * (-c / (r - a) ** 2)
                # Force on i is -dV/dr * rij/r
                fij = -dVdr * rij / r
                grad[i] += fij
                grad[j] -= fij
            elif r < b:
                # dV/dr for r < b
                dVdr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
                fij = -dVdr * rij / r
                grad[i] += fij
                grad[j] -= fij
    return grad

def _repulsive_kick(positions, kick_strength):
    """Add repulsive kick to positions to push atoms apart."""
    n = len(positions)
    kicked = positions.copy()
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 0.5:
                continue
            if r < 2.0:
                # Repulsive force that decays with distance
                force = kick_strength * np.exp(-2.0 * r) / (r + 1e-6)
                kicked[i] += force * rij / r
                kicked[j] -= force * rij / r
    return kicked

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts

def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with exactly n_target atoms.
    Shell structure:
      - Core: 1 atom at origin
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
    For n up to 60, we use core + shell1 + part of shell2.
    """
    # Core atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms
    ico_verts = _get_icosahedron_vertices()
    r1 = spacing * 1.0
    for v in ico_verts:
        points.append(r1 * v)
    
    # Shell 2: 42 atoms (if needed)
    if n_target > 13:
        # Generate second shell points
        # The second shell of a Mackay icosahedron has 42 atoms
        # These are at the centers of the faces and edges of the icosahedron
        # For simplicity, we'll generate points on a sphere at radius r2
        # and select the closest n_target - len(points) points
        
        r2 = spacing * 1.8  # Approximate second shell radius
        
        # Generate candidate points on a sphere
        # Use a Fibonacci sphere for even distribution
        n_candidates = 100
        phi_angle = np.arccos(1 - 2 * np.arange(n_candidates) / n_candidates)
        theta_angle = np.pi * (1 + np.sqrt(5)) * np.arange(n_candidates)
        
        candidates = np.zeros((n_candidates, 3))
        for k in range(n_candidates):
            candidates[k, 0] = r2 * np.sin(phi_angle[k]) * np.cos(theta_angle[k])
            candidates[k, 1] = r2 * np.sin(phi_angle[k]) * np.sin(theta_angle[k])
            candidates[k, 2] = r2 * np.cos(phi_angle[k])
        
        # Select the closest n_target - len(points) candidates
        n_needed = n_target - len(points)
        # Sort by distance to origin (should all be similar) and take first n_needed
        # Better: select points that are well-separated
        selected = []
        for cand in candidates:
            # Check if this candidate is far enough from all existing points
            min_dist = np.min([np.linalg.norm(cand - p) for p in points])
            if min_dist > 0.5 * spacing:
                selected.append(cand)
                if len(selected) == n_needed:
                    break
        
        # If we didn't get enough, add remaining candidates
        if len(selected) < n_needed:
            for cand in candidates:
                if cand not in selected:
                    selected.append(cand)
                    if len(selected) == n_needed:
                        break
        
        points.extend(selected)
    
    # If we still don't have enough, add more points
    while len(points) < n_target:
        # Add a random point near the surface
        angle1 = np.random.uniform(0, 2 * np.pi)
        angle2 = np.random.uniform(0, np.pi)
        r = spacing * 1.5
        p = np.array([
            r * np.sin(angle2) * np.cos(angle1),
            r * np.sin(angle2) * np.sin(angle1),
            r * np.cos(angle2)
        ])
        points.append(p)
    
    return np.array(points[:n_target])

def _local_minimize(positions, max_iter=500):
    """Minimize energy using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    return 1e10
        return _dzugutov_energy(pos)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    return np.ones_like(x) * 1e10
        return _dzugutov_gradient(pos).reshape(-1)
    
    result = minimize(objective, positions.reshape(-1), jac=gradient, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun

def construct(instance):
    """Repulsive Annealing: start from icosahedral template, iteratively relax with decaying kicks."""
    size = instance["n"]
    potential = instance["potential"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate initial icosahedral template
    initial_positions = _generate_mackay_icosahedron(size)
    
    # Ensure minimum distance
    # If any atoms are too close, push them apart
    for _ in range(100):
        min_dist = np.inf
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(initial_positions[i] - initial_positions[j])
                if d < min_dist:
                    min_dist = d
        if min_dist > 0.5:
            break
        # Push apart the closest pair
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(initial_positions[i] - initial_positions[j])
                if d < 0.5:
                    direction = initial_positions[i] - initial_positions[j]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                    initial_positions[i] += 0.1 * direction
                    initial_positions[j] -= 0.1 * direction
    
    # Repulsive Annealing loop
    best_positions = initial_positions.copy()
    best_energy = _dzugutov_energy(initial_positions)
    
    n_kicks = 50
    kick_decay = 0.95
    
    for kick_iter in range(n_kicks):
        if time.time() > deadline:
            break
        
        # Relax the current configuration
        relaxed_pos, relaxed_energy = _local_minimize(best_positions, max_iter=200)
        
        # Update best if improved
        if relaxed_energy < best_energy:
            best_energy = relaxed_energy
            best_positions = relaxed_pos.copy()
        
        # Apply repulsive kick
        kick_strength = 0.5 * (kick_decay ** kick_iter)
        kicked_pos = _repulsive_kick(best_positions, kick_strength)
        
        # Ensure minimum distance after kick
        for _ in range(50):
            min_dist = np.inf
            for i in range(size):
                for j in range(i + 1, size):
                    d = np.linalg.norm(kicked_pos[i] - kicked_pos[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist > 0.5:
                break
            for i in range(size):
                for j in range(i + 1, size):
                    d = np.linalg.norm(kicked_pos[i] - kicked_pos[j])
                    if d < 0.5:
                        direction = kicked_pos[i] - kicked_pos[j]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)
                        kicked_pos[i] += 0.05 * direction
                        kicked_pos[j] -= 0.05 * direction
        
        # Use kicked position as new starting point
        best_positions = kicked_pos
    
    # Final relaxation
    if time.time() < deadline - 10:
        final_pos, final_energy = _local_minimize(best_positions, max_iter=500)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END
