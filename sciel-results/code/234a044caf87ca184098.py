"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from scipy.optimize import minimize
from scipy.spatial import cKDTree

# Constants
N_CIRCLES = 200
HALF_SIDE = 0.5
# Known best radius from Huang, Ye and Cantrell (2010)
KNOWN_BEST_R = 0.036612798904


def compute_radius(centers):
    """Compute the maximum valid radius for the given centers."""
    if len(centers) != N_CIRCLES:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-12):
        return 0.0
    
    # Distance to boundary
    min_dist_to_wall = np.min(HALF_SIDE - np.abs(centers), axis=None)
    if min_dist_to_wall <= 0:
        return 0.0
    
    # Min pairwise distance
    tree = cKDTree(centers)
    # Use a generous radius for query_pairs to ensure we catch the minimum
    # Since we expect r ~ 0.036, 2*r ~ 0.073. 0.1 is safe.
    pairs = tree.query_pairs(r=0.1) 
    if len(pairs) == 0:
        min_pair_dist = 1e9
    else:
        min_pair_dist = min(np.linalg.norm(centers[i] - centers[j]) for i, j in pairs)
    
    r = min(min_dist_to_wall, min_pair_dist / 2)
    return r


def score_config(centers):
    """Return the radius if valid, else 0."""
    if centers.shape != (N_CIRCLES, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > HALF_SIDE + 1e-10):
        return 0.0
    
    # Check for coincident centers
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=1e-9)
    if len(pairs) > 0:
        return 0.0
    
    return compute_radius(centers)


def _simple_grid():
    """Fallback simple grid."""
    m = int(np.ceil(np.sqrt(N_CIRCLES)))
    # We need at least 200 points. 15x15 = 225.
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N_CIRCLES]


def nlp_optimize(centers_init, r_target, time_limit=30.0):
    """
    Attempt to improve the packing by solving an NLP problem.
    We maximize r subject to:
    1. |c_i - c_j| >= 2r for all i < j
    2. 0.5 - |c_ik| >= r for all i, k
    
    Since SLSQP maximizes -f, we minimize -r.
    However, SLSQP works best with bounded variables.
    Variables: x = [r, c_1x, c_1y, ..., c_Nx, c_Ny]
    Bounds:
    r: [0, 0.25]
    c_ix, c_iy: [0.5 - r, 0.5 - r] is not fixed bounds because r changes.
    
    Alternative formulation:
    Fix r to a specific value and check feasibility? No, we want to maximize r.
    
    Standard approach for this problem in SLSQP:
    Variables: positions only? No, r is the objective.
    Let's define the vector v = [r, x_1, y_1, ..., x_N, y_N].
    
    Objective: minimize -v[0]
    
    Constraints:
    For each pair (i,j): (x_i - x_j)^2 + (y_i - y_j)^2 - (2*v[0])^2 >= 0
    For each point i, coord k: 0.5 - |x_ik| - v[0] >= 0
    
    This is a non-convex problem with many constraints (N=200 -> ~20k pair constraints).
    SLSQP might struggle with 20k constraints.
    
    However, the prompt suggests trying this. To make it feasible within 120s:
    1. We must reduce the number of constraints.
    2. Or use a sparse Jacobian.
    3. Or use a smaller N for NLP and interpolate? No, N is fixed.
    
    Actually, 200 points -> 19900 pairs. SLSQP with 20000 constraints is likely too slow or unstable in Python without compiled code.
    
    Let's try a different NLP strategy:
    Fix r to a value slightly higher than current best.
    Check if there is a feasible configuration for that r.
    If yes, increase r. If no, decrease r.
    This turns it into a feasibility problem (constraints only, no objective to optimize, or objective is constant).
    SLSQP can handle feasibility.
    
    But even feasibility with 20k constraints is hard.
    
    Alternative: Use the "soft" constraints or a subset of constraints?
    
    Let's look at the "inspiration 3" note: "physics-based Repulsion".
    The directive says: "Formulate the packing as a hard-constrained Nonlinear Programming (NLP) problem using the SLSQP algorithm".
    
    To make this work in 120s:
    We can't solve the full 20k constraint problem with SLSQP in pure Python easily.
    However, we can use a **local** NLP approach.
    Only enforce constraints for pairs that are "close" in the current configuration.
    As r increases, more pairs become critical.
    
    Strategy:
    1. Start with a good grid.
    2. Estimate current max r.
    3. Define a "critical set" of pairs: those with distance < 2*(r + margin).
    4. Run SLSQP on positions only, with r fixed to a target value slightly higher than current best.
    5. Objective: Minimize sum of violations of the critical constraints + boundary constraints.
       Or better: Maximize r, but constrained by the critical pairs.
    
    Let's try:
    Variables: positions (400 vars).
    Objective: Minimize -r (where r is a parameter? No, r must be a variable or fixed).
    
    If we fix r, we just need to satisfy constraints.
    Let's try to find the max r by binary search on r, checking feasibility via SLSQP.
    
    Feasibility Check for fixed r:
    Minimize sum( max(0, 2r - dist_ij)^2 ) + sum( max(0, r - dist_to_wall)^2 )
    This is smooth if we use soft violations.
    
    Let's implement a "Soft Violation" minimizer using SLSQP or L-BFGS-B.
    L-BFGS-B is faster for large problems.
    
    Objective function for fixed r:
    f(x) = sum_{i<j} max(0, 2r - ||x_i - x_j||)^2 + sum_{i} sum_{k} max(0, r - (0.5 - |x_ik|))^2
    
    If min f(x) < epsilon, then r is feasible.
    
    We can binary search r between current_best and 0.04.
    
    Let's implement this.
    """
    start_time = time.time()
    
    # Get current best r
    r_current = compute_radius(centers_init)
    
    # If r is already very high, we might not improve much.
    # Let's try to push it up.
    
    # Define the soft violation function
    def soft_violation(x_flat, r_val):
        x = x_flat.reshape(N_CIRCLES, 2)
        
        # Boundary violations
        # dist to wall: 0.5 - |x|
        # violation: max(0, r_val - (0.5 - |x|))
        dists_wall = 0.5 - np.abs(x)
        viol_wall = np.maximum(0, r_val - dists_wall)
        wall_cost = np.sum(viol_wall ** 2)
        
        # Pairwise violations
        # We can't check all 20k pairs efficiently in a pure Python loop inside the objective
        # if it's called 1000 times.
        # We need to vectorize or use cKDTree inside the objective?
        # cKDTree inside objective is expensive to build (O(N log N)).
        # N=200, so 200 log 200 ~ 1500 ops. 1000 calls -> 1.5M ops. This is fine.
        
        tree = cKDTree(x)
        # Query pairs within 2*r_val + margin
        # If dist < 2*r_val, violation is (2*r_val - dist)^2
        # If dist >= 2*r_val, violation is 0
        # We only need to check pairs with dist < 2*r_val
        pairs = tree.query_pairs(r=2 * r_val)
        
        pair_cost = 0.0
        for i, j in pairs:
            d = np.linalg.norm(x[i] - x[j])
            if d < 2 * r_val:
                viol = 2 * r_val - d
                pair_cost += viol ** 2
        
        return wall_cost + pair_cost
    
    # Binary search for the maximum feasible r
    # Range: [r_current, r_current + 0.005]
    # We want to find the largest r such that min soft_violation < 1e-6
    
    def is_feasible(r_val, x_init):
        # Minimize soft_violation
        # Use L-BFGS-B for speed
        x_flat = x_init.flatten()
        
        res = minimize(
            lambda x: soft_violation(x, r_val),
            x0=x_flat,
            method='L-BFGS-B',
            options={'maxiter': 50, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        
        return res.fun < 1e-8, res.x.reshape(N_CIRCLES, 2)
    
    best_config = centers_init.copy()
    best_r = r_current
    
    # Try a few target r values
    # We can do a linear scan or binary search.
    # Given time limits, let's try a few specific values.
    
    target_r_values = np.linspace(r_current, r_current + 0.002, 5)
    
    for r_target in target_r_values:
        if time.time() > start_time + time_limit:
            break
            
        feasible, x_new = is_feasible(r_target, best_config)
        
        if feasible:
            # Update best
            best_r = r_target
            best_config = x_new
            
    return best_config, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s
    
    best_config = _simple_grid()
    best_r = compute_radius(best_config)
    
    # Initial refinement with NLP (Soft Violation)
    # This is the structural change: using NLP/Feasibility search instead of SA
    
    # We might run multiple NLP attempts with different initializations
    # 1. Grid
    # 2. Grid + small noise
    
    inits = [
        _simple_grid(),
        _simple_grid() + np.random.uniform(-0.005, 0.005, size=(N_CIRCLES, 2))
    ]
    
    for init in inits:
        init = np.clip(init, -0.5, 0.5)
        if time.time() > deadline - 30.0:
            break
            
        # Run NLP optimization
        new_config, new_r = nlp_optimize(init, best_r, time_limit=30.0)
        
        if new_r > best_r:
            best_r = new_r
            best_config = new_config
    
    # Final check and return
    # Ensure we return a valid config
    final_r = compute_radius(best_config)
    if final_r < best_r:
        # If NLP degraded it somehow (shouldn't happen if we check feasibility), fallback
        # But we only update if new_r > best_r, so this shouldn't trigger unless compute_radius is inconsistent
        pass
        
    return best_config
# EVOLVE-BLOCK-END
