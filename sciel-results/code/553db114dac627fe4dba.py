"""LJ38 cluster optimization using parametric geometric construction."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 145.0
START_TIME = time.time()


def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist


def make_fcc_truncated_octahedron_parametric(params):
    """
    Construct the FCC truncated octahedron cluster of 38 atoms 
    based on 4 inter-shell distances: r1 (inner cube), r2 (inner octahedron), 
    r3 (outer cube), r4 (outer octahedron).
    
    The structure consists of:
    1. Central atom (0,0,0)
    2. 6 atoms at (±r1, 0, 0), (0, ±r1, 0), (0, 0, ±r1)
    3. 8 atoms at (±r2, ±r2, 0), (±r2, 0, ±r2), (0, ±r2, ±r2)
    4. 12 atoms at (±r3, ±r3, ±r3)
    5. 11 atoms at (±r4, 0, 0), (0, ±r4, 0), (0, 0, ±r4) [Wait, 1+6+8+12 = 27. We need 38.]
    
    Let's re-verify the shell composition of the 38-atom FCC cluster.
    The standard 38-atom cluster is indeed a truncated octahedron.
    Layer 0: 1 atom (center)
    Layer 1: 6 atoms (face centers of inner cube) -> distance d1
    Layer 2: 12 atoms (edge centers of inner cube) -> distance d2
    Layer 3: 8 atoms (corners of inner cube) -> distance d3
    Layer 4: 12 atoms (face centers of outer cube) -> distance d4
    Layer 5: 6 atoms (edge centers of outer cube? No, this is getting complicated)
    
    Actually, the standard FCC truncated octahedron for N=38 is formed by 
    taking the first 4 shells of an FCC lattice centered at the origin.
    
    Let's list FCC lattice points (i,j,k) where i+j+k is even, scaled by a/√2.
    Distance squared (in units of a^2/2) is i^2+j^2+k^2.
    
    Shell 1 (d^2=1): (1,0,0), (0,1,0), (0,0,1) and negatives. 6 points.
    Shell 2 (d^2=2): (1,1,0), (1,0,1), (0,1,1) and permutations/signs. 12 points.
    Shell 3 (d^2=3): (1,1,1) and signs. 8 points.
    Shell 4 (d^2=4): (2,0,0), (0,2,0), (0,0,2) and negatives. 6 points.
    Shell 5 (d^2=5): (2,1,0) and perms/signs. 24 points.
    
    Total so far: 1 (center) + 6 + 12 + 8 + 6 = 33.
    We need 5 more atoms to reach 38.
    The next shell is d^2=5 (24 atoms). We take the first 5? No, symmetry is broken.
    
    Wait, the "truncated octahedron" for 38 atoms is often described as:
    1 central atom
    6 atoms at distance r1
    12 atoms at distance r2
    8 atoms at distance r3
    11 atoms at distance r4?
    
    Actually, let's look at the Cambridge Cluster Database description for LJ38.
    The global minimum is a truncated octahedron.
    The atoms are arranged in shells.
    Shell 1: 1 atom (center)
    Shell 2: 6 atoms (octahedral)
    Shell 3: 12 atoms (cubic)
    Shell 4: 8 atoms (cubic)
    Shell 5: 11 atoms?
    
    Let's check the count: 1+6+12+8 = 27.
    Remaining 11 atoms.
    In the FCC lattice, the next shell after d^2=3 (8 atoms) is d^2=4 (6 atoms) and d^2=5 (24 atoms).
    If we take the 6 atoms from d^2=4, we have 33 atoms.
    We need 5 more.
    The d^2=5 shell has 24 atoms. Taking 5 breaks symmetry.
    
    However, the "truncated octahedron" shape suggests a different packing.
    A truncated octahedron has 8 hexagonal faces and 6 square faces.
    The vertices of a truncated octahedron correspond to the (1,1,0) type vectors?
    
    Let's try a different parametrization.
    The 38-atom cluster is often considered as a "magic number" for icosahedral symmetry, but the global minimum is FCC.
    The FCC structure for 38 atoms is slightly distorted.
    
    Let's use the coordinates from the known global minimum as a starting template 
    and allow small perturbations, or better, construct it from the FCC lattice 
    with a variable lattice constant and allow the outer shell to relax.
    
    Actually, the most robust "parametric" approach for this specific problem 
    is to define the 38 positions based on the FCC lattice vectors 
    with a variable scale factor, and then let the optimizer adjust 
    the relative positions if needed. But the prompt asks for a structural change 
    to a "rigid body defined by symmetry operations".
    
    Let's define the 38 atoms as follows:
    1. Center: (0,0,0)
    2. 6 atoms: (±a, 0, 0), (0, ±a, 0), (0, 0, ±a)
    3. 12 atoms: (±b, ±b, 0), (±b, 0, ±b), (0, ±b, ±b)
    4. 8 atoms: (±c, ±c, ±c)
    5. 6 atoms: (±d, 0, 0), (0, ±d, 0), (0, 0, ±d)  <- These are collinear with shell 2.
    6. 5 atoms: From the next shell (2,1,0) type?
    
    Wait, 1+6+12+8+6 = 33.
    The remaining 5 atoms in the standard 38-atom FCC cluster are from the (2,1,0) shell?
    No, the 38-atom cluster is usually defined as the first 38 atoms of the FCC lattice 
    sorted by distance from the center.
    
    Distances squared (in units of a^2/2):
    0: 1 atom
    1: 6 atoms
    2: 12 atoms
    3: 8 atoms
    4: 6 atoms
    5: 24 atoms
    
    Sum: 1+6+12+8+6 = 33.
    We need 5 more from the 24 atoms in shell 5.
    This breaks the symmetry of the truncated octahedron.
    
    However, the "truncated octahedron" description might refer to a slightly different 
    selection or a distorted FCC.
    
    Let's assume the question implies the standard "magic" cluster which is often 
    icosahedral, but the text says "putative global minimum is a face-centred-cubic 
    truncated octahedron".
    
    Let's try to construct the 33-atom core (shells 0-4) and add 5 atoms 
    from the next shell in a way that minimizes energy.
    Or, perhaps the 38-atom cluster is formed by a different set of shells?
    
    Let's look at the energy -173.928427.
    
    Alternative Strategy:
    Use the known coordinates of the global minimum if I can generate them 
    via a high-precision search, or use a very fast local optimizer 
    starting from a good guess.
    
    Since I must use a "different strategy", and the current one is SA, 
    I will use **Direct Search on a Parametric Model**.
    
    I will model the cluster as 4 shells with variable radii r1, r2, r3, r4 
    and fixed angular positions corresponding to the FCC lattice directions.
    
    Shell 1: 6 atoms at distance r1 along axes.
    Shell 2: 12 atoms at distance r2 along face diagonals.
    Shell 3: 8 atoms at distance r3 along body diagonals.
    Shell 4: 6 atoms at distance r4 along axes (collinear with Shell 1).
    Shell 5: 5 atoms at distance r5 along specific (2,1,0) directions.
    
    To maintain symmetry and validity, I will pick the 5 atoms from the 24 available 
    in the (2,1,0) shell such that they form a symmetric set? 
    Actually, 5 atoms cannot form a highly symmetric set on that shell.
    
    Maybe the 38-atom cluster is not just the first 38 FCC atoms.
    Let's check if there is a 38-atom truncated octahedron with higher symmetry.
    A truncated octahedron has 24 vertices.
    The 38-atom cluster is often described as having a "truncated octahedral" shape 
    but it is an FCC fragment.
    
    Let's try a different approach: **Basin Hopping with a very aggressive 
    initial guess and fast local relaxation**.
    
    Wait, the directive says "Make a STRUCTURAL change: a different design or 
    modelling family...".
    
    Let's try **Genetic Algorithm** or **Particle Swarm Optimization**?
    Or **Nelder-Mead** on a reduced parameter space?
    
    Let's stick to the **Parametric FCC** idea but refine the atom selection.
    If I take the 33 atoms (shells 0-4) and then add 5 atoms from shell 5, 
    the energy will be higher than the true global minimum if the true GM 
    involves relaxation of these atoms.
    
    However, the current program (SA) is already finding a good solution (score 1.45).
    The goal is to beat 1.4867 (the inspiration best).
    
    Let's try **Simulated Annealing with a much better initial guess 
    and a different perturbation scheme**.
    
    Actually, the most effective strategy for LJ clusters is often 
    **Basin Hopping** with a robust local optimizer.
    The current program uses SA.
    
    Let's implement **Basin Hopping**.
    1. Start with the 33-atom FCC core + 5 random atoms from shell 5.
    2. Relax with L-BFGS.
    3. Perturb a random atom by a large amount.
    4. Relax.
    5. Accept if energy is lower.
    
    This is structurally different from SA (which accepts uphill moves probabilistically) 
    because Basin Hopping is a Monte Carlo method on the landscape of local minima.
    
    Let's implement Basin Hopping.
    
    """
    
    # Generate the 33-atom core
    core_points = []
    # Shell 0
    core_points.append((0,0,0))
    # Shell 1 (6)
    for sign in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
        core_points.append(sign)
    # Shell 2 (12)
    for perm in [(1,1,0), (1,0,1), (0,1,1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                p = list(perm)
                p[0] *= s1
                p[1] *= s2
                core_points.append(tuple(p))
    # Shell 3 (8)
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                core_points.append((s1, s2, s3))
    # Shell 4 (6) - (2,0,0) type
    for sign in [(2,0,0), (-2,0,0), (0,2,0), (0,-2,0), (0,0,2), (0,0,-2)]:
        core_points.append(sign)
        
    # We have 33 points. Scale them.
    # The FCC lattice constant a. The distance between neighbors is a/sqrt(2).
    # In reduced units, sigma=1. The equilibrium distance is 2^(1/6) * sigma.
    # So a/sqrt(2) = 2^(1/6) => a = sqrt(2) * 2^(1/6).
    # The coordinates above are in units of a/2? No.
    # (1,0,0) in FCC lattice (where lattice points are at integer coords with even sum)
    # distance from origin is 1.
    # Actual distance = 1 * (a/sqrt(2))? No.
    # FCC basis vectors are (0, 1/2, 1/2), (1/2, 0, 1/2), (1/2, 1/2, 0).
    # The point (1,0,0) is a lattice point. Distance from origin is 1 * a.
    # Wait, the nearest neighbors in FCC are at distance a/sqrt(2).
    # The point (1,0,0) is a nearest neighbor?
    # Distance between (0,0,0) and (1,0,0) is 1.
    # Distance between (0,0,0) and (1/2, 1/2, 0) is sqrt(1/4+1/4) = 1/sqrt(2).
    # So (1/2, 1/2, 0) is a nearest neighbor.
    # My "Shell 1" points (1,0,0) are actually second-nearest neighbors in standard FCC?
    # No, in the standard definition, the lattice points are (0,0,0), (1,0,0)...
    # The nearest neighbors of (0,0,0) are the 12 points (1/2, 1/2, 0) etc.
    # So my "Shell 2" points are the nearest neighbors.
    # My "Shell 1" points (1,0,0) are at distance 1.
    # My "Shell 2" points (1,1,0) are at distance sqrt(2).
    # This scaling is confusing.
    
    # Let's use the explicit coordinates of the nearest neighbors.
    # Nearest neighbors (12): (±0.5, ±0.5, 0), (±0.5, 0, ±0.5), (0, ±0.5, ±0.5)
    # Second nearest (6): (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    # Third nearest (8): (±0.5, ±0.5, ±0.5) -> dist sqrt(0.75) = 0.866
    # Wait, (0.5, 0.5, 0.5) dist is sqrt(0.75).
    # (1, 0, 0) dist is 1.
    # (0.5, 0.5, 0) dist is 0.707.
    
    # So the order of distance from origin in FCC (lattice constant a=1):
    # 1. (0,0,0)
    # 2. 12 atoms at dist 1/sqrt(2) ~ 0.707. Points: (±0.5, ±0.5, 0) etc.
    # 3. 6 atoms at dist 1. Points: (±1, 0, 0) etc.
    # 4. 8 atoms at dist sqrt(3/4) ~ 0.866. Points: (±0.5, ±0.5, ±0.5).
    #    Wait, 0.866 < 1. So the 8 body-center-like atoms are closer than the 6 face-center-like atoms?
    #    Let's check.
    #    (0.5, 0.5, 0.5) -> r2 = 0.25+0.25+0.25 = 0.75.
    #    (1, 0, 0) -> r2 = 1.
    #    So the 8 atoms are closer than the 6 atoms.
    
    # Order:
    # 1. Center
    # 2. 12 atoms (dist 0.707)
    # 3. 8 atoms (dist 0.866)
    # 4. 6 atoms (dist 1.0)
    # Total: 1+12+8+6 = 27 atoms.
    
    # Next shell:
    # 24 atoms at dist sqrt(1.25) ~ 1.118. Points: (±1, ±0.5, 0) etc.
    # We need 11 more atoms to reach 38.
    # 27 + 11 = 38.
    
    # So the cluster consists of the first 27 atoms of the FCC lattice 
    # plus 11 atoms from the next shell.
    
    # Let's generate these 27 atoms.
    atoms = []
    atoms.append(np.array([0,0,0]))
    
    # Shell 1: 12 atoms
    for s1 in [0.5, -0.5]:
        for s2 in [0.5, -0.5]:
            atoms.append(np.array([s1, s2, 0]))
            atoms.append(np.array([s1, 0, s2]))
            atoms.append(np.array([0, s1, s2]))
            
    # Shell 2: 8 atoms
    for s1 in [0.5, -0.5]:
        for s2 in [0.5, -0.5]:
            for s3 in [0.5, -0.5]:
                atoms.append(np.array([s1, s2, s3]))
                
    # Shell 3: 6 atoms
    for s in [1.0, -1.0]:
        atoms.append(np.array([s, 0, 0]))
        atoms.append(np.array([0, s, 0]))
        atoms.append(np.array([0, 0, s]))
        
    # We have 27 atoms.
    # Now add 11 atoms from the next shell (dist sqrt(1.25)).
    # The 24 atoms are permutations of (±1, ±0.5, 0).
    # To maintain some symmetry, we can pick them carefully, 
    # but for Basin Hopping, we can just pick 11 random ones 
    # or all 24 and let the optimizer remove/relax them? 
    # No, we must have exactly 38 atoms.
    
    # Let's pick the first 11 from a list of all 24.
    shell_4_points = []
    for perm in [(1, 0.5, 0), (1, 0, 0.5), (0.5, 1, 0), (0, 1, 0.5), (0.5, 0, 1), (0, 0.5, 1)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                p = list(perm)
                p[0] *= s1
                p[1] *= s2
                shell_4_points.append(np.array(p))
                
    # We need 11. Let's take the first 11.
    for i in range(11):
        atoms.append(shell_4_points[i])
        
    # Scale the whole cluster.
    # The equilibrium distance in LJ is 2^(1/6).
    # The nearest neighbor distance in our FCC (lattice constant 1) is 1/sqrt(2).
    # So we want 1/sqrt(2) * scale = 2^(1/6).
    # scale = 2^(1/6) * sqrt(2) = 2^(1/6 + 1/2) = 2^(4/6) = 2^(2/3).
    
    scale = 2.0 ** (2.0/3.0)
    atoms = np.array(atoms) * scale
    
    # This gives us a 38-atom cluster.
    # Now, use Basin Hopping to optimize.
    
    return atoms


def relax_ljfg(positions):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy_vectorized(pos)
        
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 200, 'gtol': 1e-8})
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def basin_hopping(initial_positions, n_iterations=100, step_size=0.5, seed=42):
    """
    Basin Hopping algorithm.
    1. Relax current structure.
    2. Perturb a random atom.
    3. Relax new structure.
    4. Accept if energy is lower.
    """
    rng = np.random.default_rng(seed)
    current = initial_positions.copy()
    current_energy = lj_energy_vectorized(current)
    
    best = current.copy()
    best_energy = current_energy
    
    for i in range(n_iterations):
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        # Perturb
        new_pos = current.copy()
        idx = rng.integers(0, N)
        displacement = rng.normal(0, step_size, size=3)
        new_pos[idx] += displacement
        
        if not check_validity(new_pos):
            continue
            
        # Relax
        relaxed_new = relax_ljfg(new_pos)
        if not check_validity(relaxed_new):
            continue
            
        new_energy = lj_energy_vectorized(relaxed_new)
        
        # Accept if better
        if new_energy < current_energy:
            current = relaxed_new
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
                
    return best, best_energy


def construct():
    """Main entry point: Use Basin Hopping starting from parametric FCC."""
    global START_TIME
    START_TIME = time.time()
    
    initial = make_fcc_truncated_octahedron_parametric(None)
    
    # Initial relaxation
    relaxed_initial = relax_ljfg(initial)
    if not check_validity(relaxed_initial):
        relaxed_initial = initial
        
    best_positions = relaxed_initial
    best_energy = lj_energy_vectorized(relaxed_initial)
    
    # Run Basin Hopping with a few different starting perturbations
    for restart in range(5):
        if time.time() - START_TIME > DEADLINE - 15:
            break
            
        rng = np.random.default_rng(100 + restart)
        perturbed = best_positions.copy()
        for _ in range(5):
            idx = rng.integers(0, N)
            perturbed[idx] += rng.normal(0, 0.3, size=3)
            
        if not check_validity(perturbed):
            continue
            
        bh_result, bh_energy = basin_hopping(
            perturbed, 
            n_iterations=50, 
            step_size=0.5, 
            seed=200 + restart
        )
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_result
            
    # Final relaxation
    final_relaxed = relax_ljfg(best_positions)
    if check_validity(final_relaxed):
        final_energy = lj_energy_vectorized(final_relaxed)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_relaxed
            
    if not check_validity(best_positions):
        best_positions = initial
        
    return best_positions

# EVOLVE-BLOCK-END
