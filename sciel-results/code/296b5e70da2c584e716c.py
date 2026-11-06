"""Combined approach: structural 'inverse-degree neighbor sum' from inspiration 1
refined with the 2-step neighborhood penalty from the current program."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees
    deg = A.sum(axis=1)
    
    # Sum of degrees of neighbors for each vertex
    neighbor_deg_sum = A @ deg
    
    # Mean degree of neighbors (avoid division by zero)
    safe_deg = np.maximum(deg, 1.0)
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Structural insight from Inspiration 1:
    # Instead of just penalizing degree, reward vertices whose neighbors
    # have LOW degree (high 1/deg). This captures the idea that blocking
    # low-degree neighbors is "cheap" because they are easy to exclude
    # or have few other conflicts.
    inv_deg = 1.0 / safe_deg
    neighbor_sum_inv_deg = A @ inv_deg
    
    # Refinement from Current Program:
    # Penalize vertices with many 2-step connections. If a vertex has many
    # paths of length 2, it sits in a dense part of the graph where picking
    # it might block a complex web of interactions.
    # A2 = A @ A
    # two_step_paths[v] = sum_u (A @ A)[v, u] = sum_u sum_k A[v,k]A[k,u]
    # This is essentially the sum of degrees of neighbors of v.
    # Note: neighbor_deg_sum[v] = sum_{u in N(v)} deg[u].
    # So two_step_paths == neighbor_deg_sum.
    # Wait, let's verify: (A @ A) @ 1 = A @ (A @ 1) = A @ deg.
    # So the total number of 2-step paths starting at v is exactly the sum of degrees of its neighbors.
    # The current program used "off_diag_two_step" which subtracted deg[v].
    # This represents 2-step paths that don't return to v immediately?
    # Actually, (A@A)[v,v] = deg[v]. So off-diag sum is neighbor_deg_sum - deg[v].
    # This metric is redundant with neighbor_deg_sum and deg.
    # Let's look for a different structural feature.
    
    # Let's stick to the strongest signals:
    # 1. -deg (low degree is good)
    # 2. + neighbor_sum_inv_deg (neighbors are low-degree, so blocking them is cheap)
    # 3. + mean_neighbor_deg (neighbors are high-degree, so they are "expensive" to keep, 
    #    but wait... if neighbors are high degree, they are hard to include in MIS anyway?
    #    Inspiration 1 used + mean_neighbor_deg. 
    #    Current program used + 0.8 * mean_neighbor_deg.
    #    Let's combine: We want to pick vertices that are low degree themselves,
    #    but whose neighbors are EITHER very low degree (easy to block) 
    #    OR very high degree (hard to include elsewhere).
    #    Actually, the standard heuristic is "pick low degree, prefer neighbors with high degree"
    #    because high-degree neighbors are likely to be picked by others or are hard to fit.
    #    But Inspiration 1 scored 1.0263 with: -deg + mean_neighbor_deg.
    #    Current program scored 1.0150 with: -deg + 0.8*mean_neighbor_deg + ...
    #    The difference is likely the extra terms or weights.
    #    Let's try a cleaner combination of the two best ideas:
    #    Idea A (Insp 1): -deg + mean_neighbor_deg
    #    Idea B (Current): -deg + 0.8*mean_neighbor_deg + 0.3*clustering - 0.2*two_step
    
    # The "clustering" term in current program:
    # neighbor_clustering = (A * A2).sum(axis=1)
    # This measures how many edges exist within the neighborhood of v.
    # If neighbors are densely connected (clique), picking v blocks only 1 of them effectively.
    # So high clustering is GOOD for the picker (less loss).
    # So we should ADD clustering.
    
    A2 = A @ A
    neighbor_clustering = (A * A2).sum(axis=1)
    mean_neighbor_connectivity = neighbor_clustering / safe_deg
    
    # Combine:
    # Base: -deg
    # Add: mean_neighbor_deg (Insp 1)
    # Add: mean_neighbor_connectivity (Current, refined)
    # Add: neighbor_sum_inv_deg? 
    # Let's check if neighbor_sum_inv_deg is redundant.
    # If deg is low, inv_deg is high. So neighbor_sum_inv_deg is high if neighbors are low degree.
    # mean_neighbor_deg is high if neighbors are high degree.
    # These are somewhat opposite.
    # Inspiration 1 found that -deg + mean_neighbor_deg works well.
    # Let's try: -deg + mean_neighbor_deg + mean_neighbor_connectivity
    
    score = -1.0 * deg + 1.0 * mean_neighbor_deg + 0.5 * mean_neighbor_connectivity
    
    # Check deadline
    if time.time() > deadline:
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
