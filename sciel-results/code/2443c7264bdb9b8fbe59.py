import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Primary: Lowest Degree (vertices with fewer neighbors are better candidates).
    2. Secondary: Prefer vertices whose neighbors have high degree.
       Rationale: Selecting a vertex that is adjacent to hubs is beneficial because
       it "blocks" the hub. Since the hub has many neighbors, blocking it prevents
       the hub from being selected, which would otherwise block many other vertices.
       By blocking the hub, we keep its other neighbors (which might be low-degree)
       available for selection.
       
    We use a weighted sum: Score = -(w1 * Degree + w2 * Sum(Degree(Neighbors)))
    
    We tune weights to ensure Degree is the primary factor but Neighbor Degree Sum
    breaks ties effectively.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate sum of degrees of neighbors
    # neighbor_deg_sum[i] = sum_{j in N(i)} deg[j]
    # This is a matrix-vector product: adj @ deg
    neighbor_deg_sum = adj @ deg
    
    # 3. Combine scores
    # Heuristic: Minimize degree, but prefer vertices whose neighbors have high degree.
    # Using average neighbor degree normalizes the secondary term to be comparable to degree.
    # score = -deg(u) + beta * (sum(deg(N(u))) / deg(u))
    # This rewards picking "leaves" attached to "hubs".
    
    # Avoid division by zero
    safe_deg = deg + 1.0
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Beta controls the trade-off. 
    # If Beta is large, we strongly prefer blocking hubs.
    # If Beta is small, we strictly follow LDF.
    # Let's try Beta = 0.5.
    score = -deg + 0.5 * avg_neighbor_deg
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
