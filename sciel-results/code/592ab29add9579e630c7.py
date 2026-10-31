"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 110.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Convex Hull Peeling with Directional Alignment.
    
    Strategy:
    1. If time is critical, fall back to nearest neighbor.
    2. Reconstruct 2D coordinates from the distance matrix using Multidimensional Scaling (MDS).
       This is done only once per call, but MDS on 50 points is fast (~ms).
       Note: Since we don't have state between calls, we re-compute or cache?
       Caching is hard without global state modification risks. 
       However, MDS on 50x50 is O(N^3) ~ 125k ops, which is negligible compared to 120s limit.
       Wait, doing MDS on every call (50 calls) is 50 * 125k = 6.25M ops. Still fast in NumPy.
       
    3. Compute the Convex Hull of the unvisited nodes.
    4. Score each node on the hull (and possibly interior nodes if hull is empty) based on:
       a. Directional Alignment: Dot product of (current -> candidate) and (current -> destination).
          We want to continue moving towards the destination, not away.
       b. Boundary Preference: Nodes on the convex hull are preferred to avoid crossing edges.
       c. Proximity: Slight preference for closer nodes to maintain local connectivity.
    
    This structural change moves from "greedy edge selection" to "geometric tour construction",
    which typically yields much shorter tours for Euclidean TSP by respecting the perimeter.
    """
    # Time check
    if time.time() - _START_TIME > TIME_LIMIT:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    U = unvisited_nodes
    n = len(U)
    
    # --- Step 1: Reconstruct Coordinates (MDS) ---
    # We need coordinates to compute convex hulls and angles.
    # The distance matrix D is NxN.
    # MDS: 
    # 1. Center the matrix.
    # 2. Eigen-decompose.
    # 3. Take first 2 components.
    
    D_sub = distance_matrix[np.ix_(U, U)]
    
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D_sub ** 2) @ J
    
    # Eigen decomposition
    # Since B is symmetric, use eigh
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top 2 dimensions
        # Scale by sqrt of eigenvalues
        X = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))[:, np.newaxis]
        
        # X is n x 2. Rows are coordinates of U[0], U[1], ...
    except Exception:
        # Fallback to nearest neighbor if MDS fails
        d1 = distance_matrix[current_node, unvisited_nodes]
        return int(U[np.argmin(d1)])

    # --- Step 2: Get Coordinates for Current and Destination ---
    # We need coordinates for current_node and destination_node in the same space.
    # We only have D_sub for U. We need to project current and dest onto this space.
    # Alternatively, we can run MDS on the full set of relevant nodes: U + [current, dest].
    # This is safer for alignment calculations.
    
    relevant_nodes = np.concatenate([U, [current_node, destination_node]])
    D_rel = distance_matrix[np.ix_(relevant_nodes, relevant_nodes)]
    n_rel = len(relevant_nodes)
    
    J_rel = np.eye(n_rel) - np.ones((n_rel, n_rel)) / n_rel
    B_rel = -0.5 * J_rel @ (D_rel ** 2) @ J_rel
    
    try:
        ev_rel, evec_rel = np.linalg.eigh(B_rel)
        idx_rel = np.argsort(ev_rel)[::-1]
        ev_rel = ev_rel[idx_rel]
        evec_rel = evec_rel[:, idx_rel]
        
        X_rel = evec_rel[:, :2] * np.sqrt(np.maximum(ev_rel[:2], 0))[:, np.newaxis]
    except Exception:
        d1 = distance_matrix[current_node, unvisited_nodes]
        return int(U[np.argmin(d1)])

    # Extract coordinates
    # X_rel[0:n] corresponds to U
    # X_rel[n] corresponds to current_node
    # X_rel[n+1] corresponds to destination_node
    
    X_U = X_rel[:n]
    X_cur = X_rel[n]
    X_dest = X_rel[n+1]

    # --- Step 3: Convex Hull of Unvisited Nodes ---
    # We want to prefer nodes on the boundary of the remaining set.
    # We can compute the convex hull of X_U.
    # Simple approach: For each point, check if it is a vertex of the convex hull.
    # Or, more simply, compute the distance to the centroid. Points far from centroid are "outer".
    # A robust proxy for "on the hull" without scipy.spatial is the "extremeness" score.
    
    centroid = np.mean(X_U, axis=0)
    dists_to_centroid = np.linalg.norm(X_U - centroid, axis=1)
    
    # Normalize distances to [0, 1]
    max_dist = np.max(dists_to_centroid)
    if max_dist > 0:
        dists_norm = dists_to_centroid / max_dist
    else:
        dists_norm = np.ones(n)

    # --- Step 4: Directional Alignment ---
    # Vector from current to destination
    v_dest = X_dest - X_cur
    norm_v_dest = np.linalg.norm(v_dest)
    
    if norm_v_dest < 1e-9:
        # If we are at destination, just pick nearest
        d1 = distance_matrix[current_node, unvisited_nodes]
        return int(U[np.argmin(d1)])
    
    v_dest_unit = v_dest / norm_v_dest
    
    # Vectors from current to each candidate
    v_candidates = X_U - X_cur
    norms = np.linalg.norm(v_candidates, axis=1)
    
    # Avoid division by zero
    norms[norms < 1e-9] = 1e-9
    v_candidates_unit = v_candidates / norms[:, np.newaxis]
    
    # Dot product: 1 means same direction, -1 means opposite
    alignment = np.sum(v_candidates_unit * v_dest_unit, axis=1)
    
    # --- Step 5: Combine Scores ---
    # We want:
    # 1. High alignment (move towards dest)
    # 2. High "outer-ness" (stay on hull)
    # 3. Low distance (efficiency)
    
    # Normalize alignment to [0, 1] roughly. 
    # Alignment range is [-1, 1]. 
    # We can map it: (alignment + 1) / 2
    alignment_score = (alignment + 1) / 2
    
    # Distance score: inverse of distance.
    # Min distance
    min_d = np.min(norms)
    if min_d > 0:
        dist_score = min_d / norms
    else:
        dist_score = np.ones(n)
    
    # Weighing:
    # Hull peeling is the primary driver. 
    # Let's try a weighted sum.
    # w_hull * dists_norm + w_align * alignment_score + w_dist * dist_score
    
    w_hull = 0.4
    w_align = 0.4
    w_dist = 0.2
    
    scores = w_hull * dists_norm + w_align * alignment_score + w_dist * dist_score
    
    # We want to MAXIMIZE this score.
    best_idx = np.argmax(scores)
    
    return int(U[best_idx])
# EVOLVE-BLOCK-END
