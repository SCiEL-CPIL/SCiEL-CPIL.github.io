import numpy as np
import time

def priority(adjacency):
    """
    Assigns a priority score to each vertex for a greedy independent set decoder.
    The decoder selects vertices in descending order of score.
    
    This implementation uses a tuned heuristic approach. It defines a family of 
    scoring functions based on graph features (degree, neighbor degree sum, 
    local clustering) and selects the one that yields the largest independent set 
    size when simulated with the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Ensure adjacency is a numpy array
    A = np.asarray(adjacency, dtype=float)
    
    # Pre-compute features
    degrees = A.sum(axis=1)
    
    # Feature 1: Degree
    f1 = degrees
    
    # Feature 2: Sum of neighbor degrees
    f2 = A.dot(degrees)
    
    # Feature 3: Number of edges among neighbors (clustering numerator)
    # For each vertex v, count pairs (u, w) in N(v) such that A[u, w] == 1
    # This is equivalent to (A^2 * 1)_v? No.
    # (A^2)_{vv} = sum_{u} A_{vu} A_{uv} = sum_{u} A_{vu}^2 = d_v (if binary)
    # We want sum_{u \in N(v)} sum_{w \in N(u)} A_{uw} ... wait.
    # Number of edges in N(v) is 0.5 * sum_{u \in N(v)} (d_u - 1) ? No.
    # Let's compute it directly for small n.
    f3 = np.zeros(n)
    for v in range(n):
        # Get neighbors
        neighbors = np.where(A[v] > 0)[0]
        if len(neighbors) < 2:
            f3[v] = 0
        else:
            # Submatrix of neighbors
            sub_A = A[np.ix_(neighbors, neighbors)]
            f3[v] = np.sum(sub_A) / 2.0
            
    # Define candidate parameter sets (a, b, c) for score = -(a*f1 + b*f2 + c*f3)
    # We want to minimize the cost, so we negate for the priority score (higher is better)
    candidates = [
        (1.0, 0.0, 0.0),  # Degree only
        (0.0, 1.0, 0.0),  # Neighbor degree sum only
        (1.0, 1.0, 0.0),  # Current program's logic
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (1.0, 1.0, 1.0),  # Include clustering
        (1.0, 1.0, 0.5),
        (0.0, 1.0, 1.0),
        (1.0, 2.0, 0.0),
        (2.0, 1.0, 0.0),
    ]
    
    best_score_val = -1
    best_scores = None
    
    for a, b, c in candidates:
        # Compute scores: We want to select vertices with LOW cost.
        # So priority score = - cost.
        costs = a * f1 + b * f2 + c * f3
        scores = -costs
        
        # Simulate greedy MIS
        # Decoder: pick vertex with highest score, remove it and its neighbors.
        remaining = set(range(n))
        selected_count = 0
        
        # To speed up simulation, we can pre-sort? 
        # No, the set of remaining changes.
        # For n=200, a simple loop is fine.
        # We need to find max score in remaining.
        
        # Optimization: Use a list of remaining and find max.
        # Since n is small, O(n^2) simulation is fast.
        
        rem_list = list(remaining)
        while rem_list:
            # Find vertex with max score in rem_list
            # Using argmax on a masked array or simple loop
            # Simple loop is safer and fast for n=200
            max_idx = -1
            max_score = -np.inf
            for idx in rem_list:
                if scores[idx] > max_score:
                    max_score = scores[idx]
                    max_idx = idx
            
            if max_idx == -1:
                break
                
            selected_count += 1
            
            # Remove max_idx and its neighbors from rem_list
            # Neighbors of max_idx
            neighbors_to_remove = set(np.where(A[max_idx] > 0)[0])
            neighbors_to_remove.add(max_idx)
            
            # Filter rem_list
            rem_list = [v for v in rem_list if v not in neighbors_to_remove]
            
        if selected_count > best_score_val:
            best_score_val = selected_count
            best_scores = scores.copy()
            
        # Check time
        if time.time() > deadline:
            break
    
    if best_scores is None:
        # Fallback
        best_scores = -f1
        
    return best_scores
