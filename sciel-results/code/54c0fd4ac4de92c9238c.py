"""
Sweep-line angular clustering heuristic for CVRP.
Strategy: Pre-compute angular coordinates of all nodes relative to the depot.
At each step, select the feasible node with the smallest angular difference
from the current node (in the forward sweep direction). This encourages
spatially contiguous, non-crossing routes which typically have lower total
length than myopic nearest-neighbor routes that may zig-zag.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_start_time = time.time()
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

# Cache for angular data per instance (identified by distance_matrix shape and hash of first few elements)
_angular_cache = {}
_instance_id = None

def _get_angular_data(distance_matrix, n):
    """
    Pre-compute angles and coordinates for all nodes.
    Uses the distance matrix to infer 2D coordinates (assuming Euclidean distances).
    Note: The problem statement implies a distance matrix. If it's not Euclidean, 
    this heuristic might be less effective, but for generated instances, it's likely Euclidean.
    We reconstruct coordinates using a simple method: 
    Set depot at (0,0). 
    We can't perfectly recover coordinates from distances alone without more info, 
    but we can use a proxy: 
    Actually, the task provides `distance_matrix`. It does not provide coordinates.
    However, standard CVRP instances are usually Euclidean. 
    If we cannot recover coordinates, we cannot compute angles.
    
    Alternative: Use a "sweep" based on distance to depot? No, that's just radial.
    
    Let's look at the input: `distance_matrix`. 
    If the instances are generated on a grid or random 2D points, we can try to reconstruct 
    approximate coordinates. 
    However, a safer "angular" heuristic that doesn't require coordinates is to use 
    the relative ordering of distances to the depot? No.
    
    Wait, if we don't have coordinates, we can't compute angles. 
    Is there a way to do a "sweep" without coordinates?
    
    Let's re-read the task. "eight generated 50-customer instances". 
    Usually, these are random points in [0,1]x[0,1] or similar.
    If I only have the distance matrix, I can't reliably compute angles.
    
    HOWEVER, many CVRP benchmarks provide coordinates. This interface only provides `distance_matrix`.
    If I assume Euclidean, I can try to embed the graph in 2D using MDS (Multi-Dimensional Scaling) 
    or simple geometric reconstruction. 
    But MDS might be slow or unstable.
    
    Let's try a different "global" strategy that doesn't need coordinates:
    **Clarke's Savings Heuristic Logic implemented greedily?**
    
    Let's stick to a robust heuristic that works with just distances:
    **Nearest Neighbor with Capacity Constraint and "Return Cost" Penalty.**
    
    The current program uses NN with look-ahead. 
    A different strategy: **Greedy Insertion**? No, that's for post-processing.
    
    Let's try **Nearest Neighbor with a "Depot Proximity" bonus**.
    Nodes that are closer to the depot are "cheaper" to return from. 
    If we pick a far node early, we might be stuck far from the depot with little capacity left.
    
    Strategy: Score = distance(current, node) + alpha * distance(node, depot)
    where alpha is a small penalty for being far from the depot when capacity is low.
    
    Or better: **Angular Sweep via MDS**.
    Let's attempt a fast MDS reconstruction. If it fails or is too slow, fallback to NN.
    Given N=50, MDS is very fast.
    
    Steps:
    1. Check if we have cached coordinates for this instance.
    2. If not, perform Classical MDS to get 2D coordinates.
    3. Compute angles relative to depot (index 0).
    4. Sort nodes by angle.
    5. In select_next_node, find the current node's angle.
    6. Among feasible nodes, pick the one with the smallest positive angle difference (cyclic).
    
    This creates "sectors" of customers served by each truck, minimizing crossing.
    """
    global _instance_id
    # Create a simple ID for the matrix to cache coordinates
    # Using the shape and a hash of a few unique values
    if distance_matrix.shape != (50, 50):
        return None # Fallback if not 50 nodes
    
    # Simple hash
    key = (distance_matrix.shape, distance_matrix[1,2], distance_matrix[2,3], distance_matrix[10,11])
    
    if key in _angular_cache:
        return _angular_cache[key]
    
    try:
        # Perform MDS
        # D is distance matrix
        D = distance_matrix
        D_sq = D ** 2
        # Centering matrix J
        n = D.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J
        
        # Eigen decomposition
        # We only need top 2 eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]
        
        # Take top 2
        if eigenvalues[0] < 0 or eigenvalues[1] < 0:
             # MDS failed or non-Euclidean, fallback
             return None
             
        coords = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
        
        # Align coordinates: 
        # 1. Rotate so that node 1 (if exists) is on the x-axis? 
        #    Actually, angles are relative. We just need a consistent frame.
        #    Let's set Depot (0) at origin.
        #    The MDS coordinates are centered. We need to shift so Depot is at (0,0).
        depot_idx = 0
        coords = coords - coords[depot_idx]
        
        # Now compute angles
        angles = np.arctan2(coords[:, 1], coords[:, 0])
        
        # Store
        data = {
            'angles': angles,
            'coords': coords
        }
        _angular_cache[key] = data
        return data
        
    except Exception:
        return None

def _get_angle_index(node, angles):
    """Helper to get angle for a node."""
    return angles[node]

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using Angular Sweep (MDS-based) if possible, 
    otherwise falls back to Nearest Neighbor with Depot Penalty.
    """
    if time.time() > _DEADLINE:
        return 0

    if len(unvisited_nodes) == 0:
        return 0

    # Identify feasible candidates
    feasible_candidates = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_candidates.append(node)
            
    if not feasible_candidates:
        return 0

    if len(feasible_candidates) == 1:
        return int(feasible_candidates[0])

    # Try Angular Sweep Strategy
    angular_data = _get_angular_data(distance_matrix, len(demands))
    
    if angular_data is not None:
        angles = angular_data['angles']
        
        # Get current angle
        current_angle = angles[current_node]
        
        best_node = -1
        best_delta = float('inf')
        
        # We want the node with the smallest positive angular difference (clockwise sweep)
        # (angle - current_angle) mod 2pi
        # We also need to ensure it's feasible (already filtered)
        
        for node in feasible_candidates:
            node_angle = angles[node]
            delta = (node_angle - current_angle) % (2 * math.pi)
            
            # If delta is very small (0), it means same angle, pick closer one?
            # For now, just min delta.
            # Tie-breaker: if delta is extremely close, use distance
            if delta < best_delta:
                best_delta = delta
                best_node = node
            elif abs(delta - best_delta) < 1e-9:
                # Tie-breaker: closer node
                if distance_matrix[current_node, node] < distance_matrix[current_node, best_node]:
                    best_node = node
        
        if best_node != -1:
            return int(best_node)
            
    # Fallback: Nearest Neighbor with Depot Penalty
    # Score = dist(current, node) + 0.1 * dist(node, depot)
    # The penalty discourages going too far away if we might not have capacity to return
    # effectively.
    
    best_node = -1
    best_score = float('inf')
    
    for node in feasible_candidates:
        d_current = distance_matrix[current_node, node]
        d_depot = distance_matrix[node, depot]
        score = d_current + 0.1 * d_depot
        
        if score < best_score:
            best_score = score
            best_node = node
            
    if best_node == -1:
        return 0
        
    return int(best_node)

# EVOLVE-BLOCK-END
