"""Sutton-Chen transition-metal cluster global minima: iterative multi-scale perturbation + L-BFGS-B relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k]) * spacing + b * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)[:n]
    return sites[idx]

def _validate_and_fix(positions, min_dist=0.35, rng=None):
    """Check for close pairs and fix them if necessary."""
    if not np.all(np.isfinite(positions)):
        return None
    n = len(positions)
    positions = positions.copy()
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Simple O(n^2) check and fix
    changed = True
    max_iter = 100
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    # Move j away from i
                    if d < 1e-6:
                        vec = rng.normal(size=3)
                    else:
                        vec = (positions[j] - positions[i]) / d
                    positions[j] = positions[i] + vec * min_dist * 1.1
                    changed = True
    return positions

def construct(instance):
    """Iterative multi-scale perturbation + L-BFGS-B relaxation with time budget."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # 1. Generate initial high-quality seed (FCC is usually good for TM clusters)
    seed_pos = _fcc_cluster(size)
    seed_pos = _validate_and_fix(seed_pos, rng=np.random.default_rng(123))
    
    # Initial relaxation
    try:
        best_energy, best_positions = relax(seed_pos, potential, kind)
    except Exception:
        # Fallback if initial relax fails
        best_positions = _validate_and_fix(seed_pos, rng=np.random.default_rng(456))
        best_energy, _ = sutton_chen_energy_gradient(best_positions, kind)
        if not np.isfinite(best_energy):
            # Last resort: random
            rng = np.random.default_rng(789)
            best_positions = rng.normal(size=(size, 3)) * 1.0
            best_positions = _validate_and_fix(best_positions, rng=rng)
            best_energy, _ = sutton_chen_energy_gradient(best_positions, kind)
    
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        return _validate_and_fix(_fcc_cluster(size), rng=np.random.default_rng(0))
    
    # 2. Iterative perturbation loop
    # Start with small perturbations and increase
    # We want to explore local minima and hopefully find the global one
    current_positions = best_positions.copy()
    current_energy = best_energy
    
    # Perturbation scales: small to large
    # For n~30-80, typical bond length ~1.0. 
    # Small: 0.1, Med: 0.3, Large: 0.6, Very Large: 1.0
    scales = [0.1, 0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0]
    
    # Number of attempts per scale
    # Adaptive: if time is running out, do fewer
    # Estimate time per relax: depends on n. For n=75, maybe 0.1-0.5s?
    # Let's try to fit as many as possible.
    
    rng = np.random.default_rng(2024)
    
    # We will iterate through scales multiple times if time permits, 
    # or just go through the list once if time is tight.
    # To be safe and effective, let's do a few passes if fast.
    
    pass_count = 0
    max_passes = 5 # Safety cap on passes
    
    while time.time() < deadline:
        if pass_count >= max_passes:
            break
            
        for scale in scales:
            if time.time() > deadline - 5.0: # Stop if less than 5s left
                break
                
            # Generate perturbation
            # Gaussian noise
            noise = rng.normal(size=(size, 3)) * scale
            
            # Apply perturbation
            perturbed = current_positions + noise
            
            # Validate and fix
            perturbed = _validate_and_fix(perturbed, rng=rng)
            
            if perturbed is None or not np.all(np.isfinite(perturbed)):
                continue
                
            # Relax
            try:
                energy, positions = relax(perturbed, potential, kind)
            except Exception:
                continue
                
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
                current_positions = positions.copy()
                # If we found a significantly better minimum, we might want to 
                # reset the scale or continue. Continuing is fine.
                
    # Final validation
    if not np.all(np.isfinite(best_positions)):
        best_positions = _validate_and_fix(_fcc_cluster(size), rng=np.random.default_rng(1))
        
    return best_positions
# EVOLVE-BLOCK-END
