"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to maintain the "tree" context if we were doing a full Prim's, 
# but since the interface is stateless (we only get current_node and unvisited), 
# we must approximate the MST growth using only the provided inputs.
#
# The directive suggests: "minimizing the distance to the current MST (formed by visited nodes) 
# plus a penalty proportional to the increase in the MST's total weight".
#
# In a standard constructive TSP (like Nearest Neighbor), we build a path. 
# Here, we frame it as growing a tree. 
# However, we don't have the full history of visited nodes, only `current_node` and `unvisited_nodes`.
# This implies the "MST" or "structure" is implicitly defined by the path built so far.
#
# A robust structural change that fits "MST extension" in a stateless interface:
# Instead of just looking at the immediate edge (current -> candidate), we look at 
# how well the candidate integrates into the *remaining* cluster to form a low-cost 
# spanning structure, while still connecting from the current position.
#
# Let's use a "Steiner-like" or "Prim-like" step:
# 1. Calculate the cost to reach the candidate from the current node: C_reach = dist(current, c)
# 2. Calculate the cost to integrate the candidate into the rest of the unvisited nodes.
#    In Prim's algorithm, we add the node with the minimum edge to the *current tree*.
#    Here, our "current tree" is effectively the set of visited nodes, but we only know the root (current).
#    However, we can approximate the "integration cost" by looking at the minimum distance 
#    from the candidate to any OTHER unvisited node. This represents the "cheapest way to continue" 
#    from that candidate into the remaining cluster.
#    
#    Score(c) = dist(current, c) + alpha * min_{u in unvisited, u!=c} dist(c, u)
#
# This is different from the current bottleneck strategy (which used MAX distance).
# Using MIN distance encourages choosing nodes that are well-connected to the rest of the 
# remaining cluster, similar to how Prim's algorithm expands the MST by the cheapest edge.
# This avoids "dead ends" where a node is far from everyone else.

# NEW STRATEGY: Global Geometric Angle Minimization with Convexity Penalty
# 
# The previous strategies focused on local distances (NN) or local connectivity (Prim-like).
# A fundamentally different approach is to look at the global geometry of the tour being built.
# 
# Key Insight: Good TSP tours tend to follow the convex hull of the point set.
# By minimizing the "turning angle" at the current node, we encourage the tour to sweep around
# the point cloud rather than zig-zagging or collapsing inward.
# 
# We can infer the "incoming" direction from the current_node. 
# Wait, the interface gives `destination_node` (the start of the tour) and `current_node` (the last added).
# So the vector "coming in" to current_node is effectively from the previous node. 
# However, we don't have the previous node. 
# 
# Let's look at the signature: select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix)
# `destination_node` is the start city (city 0 usually).
# `current_node` is the last city added.
# 
# If we assume the tour is built sequentially, the "incoming" vector to `current_node` 
# is not directly available unless we assume the path came from `destination_node` initially 
# or we have to approximate the "direction of travel".
# 
# Actually, a simpler global geometric heuristic that doesn't require history is the 
# "Angle Minimization" relative to the centroid or the start node.
# 
# Let's try a different structural framing: **Convex Hull Peeling / Peripherality**.
# 
# The knowledge from the run states: "Shifts from local greedy path extension to a global 
# metric-geometric heuristic that penalizes 'interior' nodes (low peripherality) to mimic 
# convex hull peeling... Measured: 0.9642 -> 1.036."
# 
# This suggests that pushing the tour towards the periphery is effective.
# 
# New Strategy: **Periphery-Preserving Selection**
# 
# 1. Calculate the distance from each unvisited node to the centroid of the *unvisited* set.
#    Nodes far from the centroid are "peripheral".
# 2. Calculate the distance from the current node to each candidate.
# 3. Combine these: We want to pick a node that is reasonably close to the current node 
#    (to keep local cost low) but also has high "peripherality" (to avoid collapsing into the center).
#    
#    Score(c) = w1 * dist(current, c) - w2 * dist(c, centroid_unvisited)
#    
#    Wait, minimizing distance to centroid is bad. We want to MAXIMIZE distance to centroid?
#    If we maximize distance to centroid, we jump to the farthest point, which is bad for TSP length.
#    
#    Let's refine: The "Convex Hull Peeling" idea usually works by *inserting* nodes into the hull.
#    In a constructive path, we want to avoid "interior" nodes that force the path to backtrack.
#    
#    Alternative Global Heuristic: **Lookahead with Global Cost Estimation**
#    
#    Let's try a **2-step Lookahead** with a global penalty.
#    For each candidate c in unvisited:
#    1. Immediate cost: d(current, c)
#    2. Next-step cost: min_{u in unvisited \ {c}} d(c, u)
#    3. Global "Return" cost: Estimate the cost to close the tour. 
#       A simple approximation is the distance from c to the destination_node (start), 
#       scaled by the number of remaining nodes? No, that's not quite right.
#       
#    Let's look at the "Angle" idea again. 
#    If we don't have the previous node, we can use the vector from `destination_node` to `current_node` 
#    as a proxy for the "general direction" of the tour so far? 
#    This is weak if the tour has turned significantly.
#    
#    Let's try a different structural change: **Spectral/Graph-based Heuristic**.
#    No, too slow.
#    
#    Let's stick to the **Periphery** idea but implement it correctly.
#    
#    The "Convex Hull Peeling" finding suggests that penalizing interior nodes helps.
#    How to identify interior nodes without full history?
#    The distance to the centroid of the *remaining* points is a good proxy for "how interior" a point is 
#    relative to the remaining cluster.
#    
#    However, the most effective constructive heuristics often combine:
#    1. Local proximity (don't jump too far).
#    2. Global direction (don't reverse direction).
#    
#    Since we lack the previous node, we can approximate the "direction" by using the 
#    `destination_node` as the anchor.
#    
#    Let's try: **Weighted Sum of Distance to Current and Distance to Destination (Start)**
#    This is a "Return-aware" Nearest Neighbor.
#    Score(c) = dist(current, c) + lambda * dist(c, destination_node)
#    
#    This encourages moving towards the start node, preventing the tour from getting "stuck" 
#    far away from the start with no way back.
#    
#    Let's combine this with the **Periphery** penalty.
#    
#    Strategy: 
#    Score(c) = dist(current, c) + w_return * dist(c, destination_node) + w_periphery * (max_dist_to_centroid - dist(c, centroid_unvisited))
#    
#    Actually, let's look at the "Angle" calculation using the start node as a reference frame.
#    Vector v1 = current_node - destination_node
#    Vector v2 = candidate - current_node
#    Angle between v1 and v2.
#    If the tour is sweeping around the hull, the angle should be consistent.
#    
#    Let's implement a **Directional Nearest Neighbor** heuristic.
#    
#    1. Calculate vector from destination to current: V_dir = pos[current] - pos[destination]
#    2. For each candidate c:
#       - dist_curr = dist(current, c)
#       - V_cand = pos[c] - pos[current]
#       - Dot product to measure alignment with V_dir.
#       - If the dot product is negative, we are moving "backwards" towards the start.
#       - We want to minimize distance, but penalize moving backwards too much?
#       
#    Actually, for a TSP, we want to go around. 
#    
#    Let's try the **2-Opt style lookahead** but simplified.
#    For each candidate c:
#    Cost = dist(current, c) + min_{u in unvisited \ {c}} dist(c, u)
#    This is the current program's "Prim-like" score. It scored 0.97.
#    
#    The inspiration 3 (score 1.07) used a "Model-and-optimize" approach with fitted params.
#    The knowledge cell says "Peripherality" improved score to 1.036.
#    
#    Let's implement the **Periphery-Preserving** strategy explicitly.
#    
#    1. Compute centroid of unvisited nodes.
#    2. For each candidate c:
#       a. local_cost = dist(current, c)
#       b. periphery_score = dist(c, centroid_unvisited)
#       c. We want to pick nodes that are NOT too central. 
#          However, if we pick the most peripheral node, we might jump across the cluster.
#          
#    Let's use a **Softmax** or **Rank-based** selection?
#    
#    Let's try a different structural family: **Genetic/Population-based**? No, stateless.
#    
#    Let's try **Simulated Annealing** logic? No, stateless.
#    
#    Let's go with **Angle Minimization relative to the Start Node**.
#    This is a distinct geometric framing.
#    
#    Hypothesis: By minimizing the angle change relative to the start node, we create a "fan-like" 
#    expansion which approximates the convex hull.
#    
#    Implementation:
#    1. Get coordinates of current, destination, and all unvisited nodes.
#    2. Calculate angle of vector (destination -> current) relative to x-axis.
#    3. For each candidate, calculate angle of vector (current -> candidate).
#    4. Calculate the absolute difference in angles (normalized to [0, pi]).
#    5. Score = dist(current, c) + alpha * angle_diff.
#    
#    This encourages the tour to continue in the same "sweep" direction.

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a Directional Sweep heuristic.
    
    Strategy:
    We approximate the global direction of the tour by using the vector from the 
    destination (start) to the current node. We then select the next node that 
    minimizes the distance to the current node, penalized by the deviation from 
    this global direction. This encourages a "sweep" around the point cloud, 
    which tends to follow the convex hull and avoid interior crossings.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_remaining = len(unvisited_nodes)
    
    # Get coordinates
    # distance_matrix is NxN. We don't have coordinates directly.
    # We can extract coordinates if we assume Euclidean and reconstruct? 
    # No, we only have distance_matrix.
    # We can use the distance matrix to compute angles? 
    # Law of Cosines: a^2 = b^2 + c^2 - 2bc cos(A)
    # Let A be destination, B be current, C be candidate.
    # d(A,B) = dist(dest, curr)
    # d(B,C) = dist(curr, cand)
    # d(A,C) = dist(dest, cand)
    # cos(angle at B) = (d(A,B)^2 + d(B,C)^2 - d(A,C)^2) / (2 * d(A,B) * d(B,C))
    
    # This allows us to compute the turning angle at the current node 
    # relative to the start node, without coordinates!
    
    d_dest_curr = distance_matrix[destination_node, current_node]
    
    if d_dest_curr < 1e-10:
        # Degenerate case: current is destination. Just pick nearest.
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Distances from destination to all candidates
    d_dest_cands = distance_matrix[destination_node, unvisited_nodes]
    
    # Distances from current to all candidates
    d_curr_cands = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate cosines of the angle at 'current' for the triangle (dest, curr, cand)
    # We want to minimize the deviation from a straight line? 
    # Or maximize the angle?
    # If we are sweeping around the hull, the angle should be roughly constant.
    # Initially, the angle is 0 (if we started at dest and moved to curr).
    # As we go around, the angle changes.
    
    # Actually, a simpler geometric proxy:
    # Minimize the "cross product" magnitude? No, we don't have 2D vectors.
    
    # Let's use the Law of Cosines to find the angle theta at 'current'.
    # cos(theta) = (d_dest_curr^2 + d_curr_cand^2 - d_dest_cand^2) / (2 * d_dest_curr * d_curr_cand)
    
    numerator = d_dest_curr**2 + d_curr_cands**2 - d_dest_cands**2
    denominator = 2 * d_dest_curr * d_curr_cands
    
    # Avoid division by zero
    denominator[denominator < 1e-10] = 1e-10
    
    cos_theta = numerator / denominator
    
    # Clip to [-1, 1] for arccos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # theta is the angle between vector (curr->dest) and vector (curr->cand).
    # We want to continue in the "forward" direction.
    # If we assume the tour is building a loop, the angle relative to the "incoming" vector 
    # (which is curr->prev, but we use curr->dest as a proxy for "global direction") 
    # should be minimized?
    
    # Actually, if we use curr->dest as the reference, and we want to sweep away from the start,
    # we might want to maximize the angle? Or minimize?
    
    # Let's look at the "Periphery" insight again.
    # Penalizing interior nodes.
    # Interior nodes are those where the angle is "sharp" relative to the hull?
    
    # Let's try a different structural change: **Lookahead-2 with Global Return Penalty**
    # Score(c) = dist(current, c) + min_{u != c} dist(c, u) + beta * dist(c, destination)
    
    # The term dist(c, destination) penalizes nodes that are far from the start, 
    # but we want to return to the start eventually. 
    # Actually, we want to AVOID nodes that are far from the start if we are early in the tour?
    # No, we want to visit all nodes.
    
    # Let's go with the **Angle Minimization** relative to the start.
    # If we minimize the angle (theta), we are moving "backwards" towards the start.
    # If we maximize the angle (theta close to pi), we are moving "forwards" away from the start.
    # For a TSP tour, we want to go around. 
    # Initially, we should move away. 
    # So we want to MAXIMIZE theta?
    # But if we maximize theta, we pick the farthest node in terms of angle, which might be far in distance.
    
    # Let's combine:
    # Score = dist(current, c) - alpha * theta
    # This encourages moving away from the start direction.
    
    theta = np.arccos(cos_theta)
    
    # Normalize theta to [0, 1]
    theta_norm = theta / np.pi
    
    # We want to minimize distance, but reward moving "forward" (away from start).
    # However, if we are late in the tour, we should move "back" towards the start.
    # We don't know the step count.
    
    # Let's assume a constant bias towards "forward" motion.
    w_angle = 0.5
    
    scores = d_curr_cands - w_angle * theta_norm
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
