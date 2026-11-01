"""
Strategy: Force-directed graph optimization with rigid-body cluster perturbations.
We treat the circles as nodes in a dynamic graph. Edges exist between circles
that are closer than a threshold (e.g., 4 * current_min_dist). 
Optimization alternates between:
1. A force-directed relaxation step (repulsion along edges, attraction to maintain spacing).
2. A rigid-body perturbation step where a connected component of the "contact graph" 
   is moved as a single unit to explore large-scale rearrangements without breaking 
   local lattice structures.
"""

import numpy as np
import time
import sys
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
def _compute_score(centers):
    """
    Computes the minimum radius for a given set of centers.
    Returns (min_dist_between_centers / 2, min_dist_to_boundary).
    The actual score is the minimum of these two.
    """
    if len(centers) == 0:
        return 0.0, 0.0
    
    # Distance to boundary: min(0.5 - |x|, 0.5 - |y|)
    # Since square is [-0.5, 0.5]
    dist_x = 0.5 - np.abs(centers[:, 0])
    dist_y = 0.5 - np.abs(centers[:, 1])
    min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
    
    if min_boundary_dist < 0:
        # Invalid: outside square
        return 0.0, min_boundary_dist

    # Distance between centers
    tree = cKDTree(centers)
    # We only care about distances smaller than some threshold to save time,
    # but for score we need the global min.
    # For N=200, O(N^2) is fast enough, but cKDTree query_pairs is faster.
    # Let's use query_pairs with a generous r to find all close pairs, 
    # but if no pairs are close, we need the min of all.
    # Actually, for N=200, full distance matrix is 40k ops, which is trivial.
    # But let's stick to cKDTree for consistency with potential scaling.
    # To get the global min efficiently without O(N^2) in Python loops:
    # We can just compute the full matrix for N=200.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_center_dist = np.min(dists)
    
    return min_center_dist / 2.0, min_boundary_dist

def _get_contact_graph(centers, threshold_factor=2.5):
    """
    Returns a boolean adjacency matrix for the contact graph.
    Two circles are connected if their distance is < threshold_factor * current_min_dist.
    This helps identify clusters that should move together.
    """
    # First, estimate current min dist to set threshold
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist == 0:
        # Fallback
        threshold = 0.1
    else:
        threshold = threshold_factor * min_dist
        
    # Build adjacency
    # dists < threshold and i != j
    adj = (dists < threshold)
    return adj, min_dist

def _force_relaxation(centers, iterations=50, learning_rate=0.05):
    """
    Simple force-directed relaxation.
    Repulsive force between all pairs (or close pairs).
    Attractive force towards maintaining a target distance? 
    Actually, standard packing uses repulsion only, but that expands the whole system.
    We need to keep them inside the box.
    
    Better approach: 
    1. Repulsion between overlapping or close circles.
    2. Penalty for being outside the box (or just clamp).
    3. To maximize min dist, we can use a 'repulsion' that is stronger for closer pairs.
    
    Let's use a potential energy function:
    E = sum_{i<j} max(0, r_target - dist_ij)^2 + penalty_outside
    But we don't know r_target. 
    Instead, let's just do gradient ascent on the minimum distance.
    This is non-smooth.
    
    Alternative: Simulated Annealing with a specific move set.
    The prompt asks for "rigid body perturbation".
    Let's implement a hybrid:
    1. Identify clusters based on proximity.
    2. Propose a move: 
       a) Move a single random circle.
       b) Move a random cluster (connected component in contact graph) as a rigid body.
    3. Accept if it improves the score (or with some probability if it doesn't, to escape local minima).
    """
    pass # Placeholder, we will implement the main loop in construct

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin for 150s limit
    
    # Initialization: Hexagonal packing is a good start.
    # Fit hexagonal lattice into square [-0.5, 0.5].
    # Side length of hex triangle is s.
    # Vertical spacing is s * sqrt(3)/2.
    # Horizontal spacing is s.
    # Rows alternate offset.
    
    # Let's try to fit as many as possible, or just generate N points.
    # For N=200, roughly 14x14 grid.
    # Hexagonal density is higher.
    
    # Initialize with a perturbed hexagonal grid
    def hex_init(n):
        # Estimate rows and cols
        # Area per circle ~ 1/sqrt(n)
        # Hex lattice area per point = s^2 * sqrt(3)/2
        # 1 = n * s^2 * sqrt(3)/2  => s = sqrt(2 / (n * sqrt(3)))
        s = np.sqrt(2.0 / (n * np.sqrt(3.0)))
        
        # Number of rows
        # Height = (rows - 1) * s * sqrt(3)/2 + s (approx)
        # Let's just generate a grid and take first N
        # Or generate enough to fill and crop.
        
        points = []
        # Center the lattice
        # We need to fit in [-0.5, 0.5]
        # Let's generate a large enough lattice and center it.
        
        # Try to fit rows such that height <= 1
        # height = (num_rows - 1) * s * sqrt(3)/2
        # max_rows = 1 / (s * sqrt(3)/2) + 1
        max_rows = int(1.0 / (s * np.sqrt(3.0)/2.0)) + 1
        max_cols = int(1.0 / s) + 1
        
        # Generate grid
        for r in range(max_rows):
            y = (r - max_rows/2.0 + 0.5) * s * np.sqrt(3.0)/2.0
            offset = 0.0 if r % 2 == 0 else s/2.0
            for c in range(max_cols):
                x = (c - max_cols/2.0 + 0.5) * s + offset
                points.append([x, y])
                
        points = np.array(points)
        
        # Filter points inside [-0.5, 0.5] with a small margin to ensure valid start
        # Actually, if we start outside, the score is 0. We must start inside.
        # Let's check bounds.
        mask = (points[:, 0] > -0.5) & (points[:, 0] < 0.5) & (points[:, 1] > -0.5) & (points[:, 1] < 0.5)
        points = points[mask]
        
        # If we have too many, randomly sample N. If too few, pad? 
        # With the s calculation, we should have plenty.
        if len(points) >= n:
            # Randomly select n points to break symmetry
            indices = np.random.choice(len(points), n, replace=False)
            points = points[indices]
        else:
            # Pad with random points inside
            extra = np.random.uniform(-0.4, 0.4, (n - len(points), 2))
            points = np.vstack([points, extra])
            
        return points

    centers = hex_init(N)
    
    # Initial score
    current_score, _ = _compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Optimization Loop
    # Temperature for Simulated Annealing
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.995
    
    # Step size for moves
    step_size = 0.05
    
    while time.time() < deadline:
        # 1. Identify clusters for rigid body moves
        # We do this every few iterations to save time
        if np.random.random() < 0.1: # Rebuild graph occasionally
            adj, min_dist = _get_contact_graph(centers, threshold_factor=2.5)
        else:
            # Reuse last graph or approximate
            # For simplicity, let's just compute min_dist quickly
            # Or just use single point moves if graph is stale
            pass
        
        # Propose a move
        move_type = np.random.random()
        
        new_centers = centers.copy()
        
        if move_type < 0.3 and 'adj' in locals():
            # Rigid Body Move: Pick a random connected component
            # Find connected components of adj
            # Simple BFS/DFS
            visited = np.zeros(N, dtype=bool)
            components = []
            for i in range(N):
                if not visited[i]:
                    # BFS
                    queue = [i]
                    visited[i] = True
                    comp = []
                    while queue:
                        u = queue.pop()
                        comp.append(u)
                        for v in np.where(adj[u])[0]:
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                    components.append(comp)
            
            # Pick a random component (weighted by size? No, uniform is fine)
            if components:
                comp = components[np.random.randint(len(components))]
                # Move this component by a random vector
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                new_centers[comp] += np.array([dx, dy])
        
        else:
            # Single Point Move
            i = np.random.randint(N)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            new_centers[i, 0] += dx
            new_centers[i, 1] += dy
            
        # Clamp to box? 
        # If we clamp, we might get stuck on walls. 
        # Better to allow slight violation and penalize, or just reject if invalid.
        # The score function returns 0 if invalid. 
        # So if new_centers is invalid, score is 0.
        
        new_score, _ = _compute_score(new_centers)
        
        # Acceptance Criterion
        if new_score > current_score:
            centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new - old) / T)
            # Note: new_score is likely 0 if invalid, so delta is large negative.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    centers = new_centers
                    current_score = new_score
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Adaptive step size? 
        # If we accept a lot, increase step size. If we reject a lot, decrease.
        # Simple heuristic:
        # if current_score > best_score:
        #     step_size *= 1.05
        # else:
        #     step_size *= 0.95
        # step_size = np.clip(step_size, 1e-4, 0.2)
        
    return best_centers
# EVOLVE-BLOCK-END
