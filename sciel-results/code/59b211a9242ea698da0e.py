import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

TOTAL_TIME_LIMIT = 330.0
SAFETY_MARGIN = 10.0
INSTANCE_TIME_LIMIT = (TOTAL_TIME_LIMIT - SAFETY_MARGIN) / 6.0

def _generate_fcc(n, scale=1.0, noise=0.0, seed=None):
    if n <= 0:
        return np.empty((0, 3))
    rng = np.random.default_rng(seed) if seed is not None else None
    
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * scale
    
    max_shell = int(np.ceil(n ** (1.0/3.0) * 1.5)) + 2
    points = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * scale + b
                    points.append(p)
                    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    if noise > 0:
        if rng is None:
            rng = np.random.default_rng()
        points += rng.normal(0, noise, points.shape)
        
    points -= np.mean(points, axis=0)
    return points

def _generate_hcp(n, scale=1.0, noise=0.0, seed=None):
    if n <= 0:
        return np.empty((0, 3))
    rng = np.random.default_rng(seed) if seed is not None else None
    
    a = scale
    c = a * np.sqrt(8.0/3.0)
    a1 = np.array([a, 0, 0])
    a2 = np.array([-a/2, a * np.sqrt(3)/2, 0])
    
    points = []
    max_layer = int(np.ceil(n ** (1.0/3.0) * 2.0)) + 2
    for l in range(-max_layer, max_layer + 1):
        z = l * c
        offset = np.array([0, 0, 0]) if l % 2 == 0 else (a1 + a2) / 3.0
        max_r = int(np.ceil(n ** (1.0/3.0))) + 1
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                p = i * a1 + j * a2 + offset + np.array([0, 0, z])
                points.append(p)
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    
    if noise > 0:
        if rng is None:
            rng = np.random.default_rng()
        points += rng.normal(0, noise, points.shape)
        
    points -= np.mean(points, axis=0)
    return points

def _generate_random(n, seed=None, min_dist=0.7):
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    points = []
    max_radius = 1.0 * n ** (1.0/3.0)
    
    for _ in range(20000):
        if len(points) >= n:
            break
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) > max_radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
            
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
            
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    start_time = time.time()
    deadline = start_time + INSTANCE_TIME_LIMIT
    
    best_energy = np.inf
    best_positions = None
    
    # Define a set of initial configurations
    # We will try to generate them lazily or in a loop
    
    # 1. Structured
    # 2. Structured + Noise
    # 3. Random
    
    tasks = []
    # FCC variants
    for scale in [0.9, 1.0, 1.1]:
        for noise in [0.0, 0.05]:
            tasks.append(lambda s=scale, no=noise: _generate_fcc(n, s, no, seed=42))
            
    # HCP variants
    for scale in [0.9, 1.0, 1.1]:
        for noise in [0.0, 0.05]:
            tasks.append(lambda s=scale, no=noise: _generate_hcp(n, s, no, seed=42))
            
    # Random variants
    for seed in range(10):
        tasks.append(lambda s=seed: _generate_random(n, seed=s))
        
    # Add more randoms if needed
    for seed in range(10, 30):
        tasks.append(lambda s=seed: _generate_random(n, seed=s))

    for task in tasks:
        if time.time() > deadline - 2.0: # Leave 2s margin
            break
            
        try:
            positions = task()
            if positions.shape != (n, 3):
                continue
                
            # Quick sanity check: min distance
            # If min dist is very small, relax might fail or be slow
            # But let's try
            
            energy, relaxed_pos = relax(positions, 'sutton_chen', kind)
            
            if not np.all(np.isfinite(relaxed_pos)):
                continue
                
            # Check validity
            dists = np.linalg.norm(relaxed_pos[:, np.newaxis] - relaxed_pos[np.newaxis, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(dists)
            
            if min_d < 0.35:
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
                
        except Exception:
            pass
            
    if best_positions is None:
        # Fallback
        positions = _generate_random(n, seed=0)
        try:
            _, positions = relax(positions, 'sutton_chen', kind)
        except:
            pass
        best_positions = positions
        
    return best_positions
