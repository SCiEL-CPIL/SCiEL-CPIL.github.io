"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import scipy.spatial as spatial

DEADLINE = 115.0  # safety margin before 120s kill


def _hex_init(n):
    """Create a hexagonal packing initialization for n circles in unit square."""
    # For 32 circles, use a hexagonal lattice pattern
    # Hexagonal packing: rows offset by half spacing
    # Let's try to fit 32 circles in a hexagonal pattern
    
    # Approximate hexagonal grid for 32 circles
    # 6x6 grid would be 36, so we can use 6 columns and ~5.3 rows
    # Let's use a systematic approach: place circles in hexagonal pattern
    
    positions = []
    r = 0.0  # will be determined by packing
    
    # Hexagonal lattice: horizontal spacing = 2r, vertical spacing = 2*r*sqrt(3)/2 = r*sqrt(3)
    # For N=32, let's try a 6x6 arrangement with some removed, or 5x7, etc.
    
    # Try different arrangements and pick the best
    best_positions = None
    best_min_dist = 0
    
    for cols in range(4, 10):
        rows = int(np.ceil(n / cols))
        if rows * cols < n:
            continue
            
        # Hexagonal spacing
        # For a hexagonal grid with 'cols' columns and 'rows' rows
        # Horizontal spacing: dx
        # Vertical spacing: dy = dx * sqrt(3)/2
        # Odd rows shifted by dx/2
        
        # We need to fit in [0,1]^2
        # Total width ≈ (cols-1)*dx + 2r = (cols-1)*dx + dx = cols*dx (if r=dx/2)
        # Actually if circle radius = dx/2, then center range is [dx/2, 1-dx/2]
        # So (cols-1)*dx ≤ 1 - dx => cols*dx ≤ 1 => dx ≤ 1/cols
        
        # For hexagonal: vertical spacing between rows = dx * sqrt(3)/2
        # Total height ≈ (rows-1)*dy + 2r = (rows-1)*dx*sqrt(3)/2 + dx
        # Need: (rows-1)*dx*sqrt(3)/2 + dx ≤ 1
        # dx * [(rows-1)*sqrt(3)/2 + 1] ≤ 1
        # dx ≤ 1 / [(rows-1)*sqrt(3)/2 + 1]
        
        dx_from_width = 1.0 / cols
        dx_from_height = 1.0 / ((rows - 1) * np.sqrt(3) / 2 + 1) if rows > 1 else 1.0
        dx = min(dx_from_width, dx_from_height)
        
        # Place circles
        pos = []
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx + (dx / 2 if row % 2 == 1 else 0)
                y = (row + 0.5) * dx * np.sqrt(3) / 2 + dx / 2
                pos.append((x, y))
                if len(pos) == n:
                    break
            if len(pos) == n:
                break
        
        if len(pos) != n:
            continue
        
        pos = np.array(pos)
        
        # Check if all in [0,1]^2
        if np.any(pos < 0) or np.any(pos > 1):
            continue
        
        # Compute minimum distance between pairs
        if len(pos) > 1:
            tree = spatial.KDTree(pos)
            pairs = tree.query_pairs(1.0)  # will find all pairs within distance 1
            if pairs:
                min_d = min(np.linalg.norm(pos[i] - pos[j]) for i, j in pairs)
            else:
                min_d = 1.0
        else:
            min_d = 1.0
        
        # Also check distance to boundaries
        min_boundary = min(np.min(pos), 1 - np.max(pos))
        
        if min_d > best_min_dist:
            best_min_dist = min_d
            best_positions = pos
    
    if best_positions is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
        best_positions = np.array(cells[:n])
    
    return best_positions


def _compute_radii(positions):
    """Compute maximum possible radii for given positions (non-overlapping, in unit square)."""
    n = len(positions)
    radii = np.ones(n) * 0.5
    
    # Distance to boundaries
    for i in range(n):
        radii[i] = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
    
    # Distance to other circles: r_i + r_j <= dist(i,j)
    # To maximize sum, we need to solve: maximize sum(r_i) subject to r_i + r_j <= d_ij, r_i >= 0
    # This is a linear program. But for speed, use a simpler approach:
    # Set r_i = min over all j of (d_ij - r_j) ... but that's circular.
    
    # Simple approach: iterate
    for _ in range(50):
        for i in range(n):
            max_r = radii[i]  # keep current as upper bound from boundaries
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(positions[i] - positions[j])
                max_r = min(max_r, d - radii[j])
            radii[i] = max(0, max_r)
    
    return radii


def _objective_and_grad(x, positions):
    """Objective: negative sum of radii. x = positions flattened."""
    n = len(positions)
    pos = x.reshape(n, 2)
    
    # Radii from boundaries
    r = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), np.minimum(pos[:, 1], 1 - pos[:, 1]))
    
    # Pairwise constraints: r_i + r_j <= d_ij
    # For each pair, the binding constraint reduces radii
    # Use a soft approach: penalty for violations
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            violation = r[i] + r[j] - d
            if violation > 0:
                penalty += violation ** 2
    
    return -(r.sum() - penalty)


def construct():
    start_time = time.time()
    
    # Initialize with hexagonal pattern
    positions = _hex_init(N).astype(float)
    
    # Initial radii
    radii = _compute_radii(positions)
    best_score = radii.sum()
    best_positions = positions.copy()
    best_radii = radii.copy()
    
    # Local search: random perturbations with acceptance
    n = N
    rng = np.random.RandomState(42)
    
    # Use a simple local search with simulated annealing-like acceptance
    current_positions = positions.copy()
    current_radii = radii.copy()
    current_score = best_score
    
    step_size = 0.05
    num_iterations = 0
    
    while time.time() - start_time < DEADLINE:
        num_iterations += 1
        
        # Random perturbation
        delta = rng.randn(n, 2) * step_size
        new_positions = current_positions + delta
        
        # Clamp to [0, 1]
        new_positions = np.clip(new_positions, 0, 1)
        
        # Compute new radii
        new_radii = _compute_radii(new_positions)
        new_score = new_radii.sum()
        
        # Accept if better or with some probability
        if new_score > current_score:
            current_positions = new_positions
            current_radii = new_radii
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_positions = current_positions.copy()
                best_radii = current_radii.copy()
        else:
            # Simulated annealing: accept with probability exp(-delta/T)
            delta = current_score - new_score
            T = 0.01 * (1 - num_iterations / 10000)  # cooling
            if T > 0 and rng.rand() < np.exp(-delta / max(T, 1e-6)):
                current_positions = new_positions
                current_radii = new_radii
                current_score = new_score
        
        # Adapt step size
        if num_iterations % 100 == 0:
            if current_score > best_score - 0.01:
                step_size *= 0.95
            else:
                step_size *= 1.05
            step_size = np.clip(step_size, 0.001, 0.2)
        
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break
    
    # Final polish: try to optimize with scipy
    if time.time() - start_time < DEADLINE - 10:
        # Use scipy minimize to polish
        x0 = best_positions.flatten()
        
        def obj(x):
            pos = x.reshape(n, 2)
            r = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), np.minimum(pos[:, 1], 1 - pos[:, 1]))
            
            # Penalty for overlaps
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    violation = r[i] + r[j] - d
                    if violation > 0:
                        penalty += violation ** 2
            
            return -(r.sum() - penalty)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 500})
            if result.success or result.fun < -best_score:
                new_pos = result.x.reshape(n, 2)
                new_pos = np.clip(new_pos, 0, 1)
                new_radii = _compute_radii(new_pos)
                new_score = new_radii.sum()
                
                if new_score > best_score:
                    best_score = new_score
                    best_positions = new_pos
                    best_radii = new_radii
        except Exception:
            pass
    
    # Construct final array
    result = np.column_stack([best_positions, best_radii])
    
    # Ensure validity: clamp radii to be non-negative and within bounds
    for i in range(n):
        result[i, 2] = max(0, min(result[i, 2], result[i, 0], 1 - result[i, 0], result[i, 1], 1 - result[i, 1]))
    
    return result
# EVOLVE-BLOCK-END
