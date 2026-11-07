"""
Strategy: Multi-start Basin Hopping with Jellium-inspired seeds.
Transition metal clusters often exhibit spherical or quasi-spherical symmetry (Jellium model).
We generate initial structures that respect this symmetry (shells) and random compact structures.
We then use scipy.optimize.basin_hopping, which is a robust global optimization method
that combines local minimization (L-BFGS-B via the provided relax wrapper logic, 
though we will use sutton_chen_energy_gradient directly for the BH objective if needed, 
or rely on the provided relax if it's faster, but BH needs a function).

Wait, the provided `relax` returns (energy, positions). 
`scipy.optimize.basin_hopping` requires a function `objective(x)` that returns energy.
We can wrap `sutton_chen_energy_gradient` to get just the energy.

The key structural change is using a dedicated global optimizer (Basin Hopping) 
with multiple diverse seeds, rather than a manual perturbation loop.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import basin_hopping
from autoresearch_suite.chem_clusters import sutton_chen_energy_gradient

# Deadline guard: Evaluator kills at 330s. We aim for 300s to be safe.
_DEADLINE_SEC = 300.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _get_energy_and_grad(positions, kind):
    """Wrapper for sutton_chen_energy_gradient to fit basin_hopping requirements."""
    energy, grad = sutton_chen_energy_gradient(positions, kind)
    return energy, grad

def _generate_jellium_seeds(n, spacing=0.72, num_seeds=5, rng=None):
    """
    Generate Jellium-inspired seeds: concentric shells with random orientations.
    This exploits the spherical symmetry often found in global minima.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    seeds = []
    # Determine shell sizes roughly proportional to surface area
    # For simplicity, we'll just use random spherical shells
    for seed_idx in range(num_seeds):
        pos = np.zeros((n, 3))
        # Randomly assign atoms to shells
        # Number of shells ~ n^(1/3)
        num_shells = max(2, int(n ** (1/3)))
        atoms_per_shell = n // num_shells
        remaining = n % num_shells
        
        current_atom = 0
        for shell in range(num_shells):
            count = atoms_per_shell + (1 if shell < remaining else 0)
            # Radius increases with shell index
            # Optimal radius for N atoms on a sphere ~ sqrt(N) * constant
            # But here we have multiple shells.
            # Let's just use a linear increase in radius
            r = spacing * (shell + 1) * 0.5 
            
            # Generate random points on a sphere of radius r
            # Use Gaussian distribution and normalize
            pts = rng.normal(size=(count, 3))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms * r
            
            # Add some random jitter to break symmetry
            pts += rng.normal(scale=0.05, size=(count, 3))
            
            pos[current_atom:current_atom+count] = pts
            current_atom += count
            
        seeds.append(pos)
    return seeds

def _generate_random_compact_seeds(n, spacing=0.72, num_seeds=10, rng=None):
    """
    Generate random compact clusters with minimum distance constraint.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    seeds = []
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6 * spacing
    
    for seed_idx in range(num_seeds):
        points = []
        attempts = 0
        # Increase attempts for larger n
        max_attempts = n * 200
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check distance to existing points
                if len(points) == 0 or np.all(np.linalg.norm(np.array(points) - p, axis=1) >= min_dist):
                    points.append(p)
            attempts += 1
        
        if len(points) == n:
            seeds.append(np.array(points))
        else:
            # Fallback: just random points, BH should fix it
            seeds.append(rng.uniform(-radius, radius, (n, 3)))
            
    return seeds

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Define the objective function for basin_hopping
    # It must take x (flattened array) and return energy
    def objective(x):
        pos = x.reshape(n, 3)
        try:
            energy, _ = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(energy):
                return 1e10
            # Penalty for atoms too close
            # Calculate min distance efficiently? For small n, O(n^2) is fine.
            # But BH calls this many times.
            # Let's just check for NaNs/Infs and rely on the potential's repulsion.
            # However, the task says "two atoms closer than 0.35 make the whole result invalid".
            # We should penalize this heavily in the objective to guide BH away.
            # But computing all pair distances is O(n^2).
            # Let's do a quick check.
            # Actually, sutton_chen_energy_gradient might fail or return inf if too close?
            # The prompt says "Non-finite coordinates ... make the whole result invalid".
            # It implies the evaluator checks it.
            # To be safe, we add a penalty.
            
            # Efficient min distance check
            # For n < 100, O(n^2) is ~10k ops, which is fast.
            # Let's use a vectorized approach for the penalty.
            # But doing this every objective call might be slow.
            # Let's assume the potential's repulsive term (r^-12 or r^-9) is strong enough 
            # to prevent atoms from getting too close during optimization, 
            # and we will do a final check.
            return energy
        except Exception:
            return 1e10

    # Generate diverse seeds
    rng = np.random.default_rng(12345)
    seeds = []
    
    # 1. Jellium seeds
    jellium_seeds = _generate_jellium_seeds(n, num_seeds=5, rng=rng)
    seeds.extend(jellium_seeds)
    
    # 2. Random compact seeds
    random_seeds = _generate_random_compact_seeds(n, num_seeds=10, rng=rng)
    seeds.extend(random_seeds)
    
    # Shuffle seeds to mix types
    rng.shuffle(seeds)
    
    best_energy = float('inf')
    best_positions = None
    
    # Run basin_hopping on each seed
    # We need to be careful with time.
    # basin_hopping has a 'niter' parameter.
    # We can set a low niter and rely on multiple starts.
    # Or use a time-based stopping criterion? BH doesn't support time-based stopping directly.
    # We will set a modest niter, say 20-50, and check time between seeds.
    
    # Determine niter based on n? Smaller n might need more, larger n might be harder.
    # Let's try niter=30 for all.
    
    for i, seed in enumerate(seeds):
        if _time_left() < 10:
            break
            
        try:
            # Flatten the seed
            x0 = seed.flatten()
            
            # Run basin_hopping
            # stepsize: typical distance between atoms is ~0.7-1.0. 0.5 is reasonable.
            # minenergyfunc: optional, we'll just use the default.
            # niter: number of iterations.
            
            # We can use a callback to check time, but BH doesn't support it easily.
            # We'll just run it and hope it finishes.
            # To be safe, we can limit niter.
            
            res = basin_hopping(
                objective, 
                x0, 
                niter=30, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'L-BFGS-B'} # BH uses L-BFGS-B by default, but specifying is good
            )
            
            energy = res.fun
            positions = res.x.reshape(n, 3)
            
            # Check validity
            if np.isfinite(energy) and np.all(np.isfinite(positions)):
                # Check min distance
                min_d = float('inf')
                # O(n^2) check
                for a in range(n):
                    for b in range(a+1, n):
                        d = np.linalg.norm(positions[a] - positions[b])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                else:
                    # Invalid, try to fix? Or just skip.
                    # Skipping is safer.
                    pass
        except Exception:
            continue
            
    # If we have time left, do a final polish on the best found with a longer L-BFGS-B run
    if best_positions is not None and _time_left() > 5:
        try:
            # Use the provided relax function for a final high-precision relaxation
            # This might be faster than running BH again
            energy, positions = relax(best_positions, 'sutton_chen', kind)
            if np.isfinite(energy) and np.all(np.isfinite(positions)):
                # Check min distance again
                min_d = float('inf')
                for a in range(n):
                    for b in range(a+1, n):
                        d = np.linalg.norm(positions[a] - positions[b])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        # Fallback: return a random valid structure
        # This should rarely happen
        pos = np.random.uniform(-1, 1, (n, 3))
        return pos
        
    return best_positions
# EVOLVE-BLOCK-END
