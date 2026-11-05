import numpy as np
import time
import random

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Initialize scores using a strong heuristic (Weighted Local Degree).
    2. Perform internal local search (Simulated Annealing / Hill Climbing) 
       on the score vector to maximize the resulting independent set size.
       - Perturbation: Swap the scores of two random vertices.
       - Evaluation: Run the fixed greedy decoder.
       - Acceptance: Accept if the new set size is larger, or with some probability if smaller (to escape local optima).
    3. Return the best score vector found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=int)
    n = adj.shape[0]
    
    # Helper function to run the greedy decoder
    def run_greedy(scores):
        """
        Greedy decoder: picks vertices in descending score order, 
        skipping neighbors of already picked vertices.
        Returns the size of the independent set.
        """
        # Get indices sorted by score descending
        order = np.argsort(-scores)
        
        selected = np.zeros(n, dtype=bool)
        # We need to check neighbors. 
        # For speed, we can maintain a 'blocked' set.
        # Since n=200, simple loops are fine, but let's use vectorization where possible.
        
        # Pre-compute neighbor lists for faster lookup? 
        # Or just use the matrix.
        
        count = 0
        for v in order:
            if selected[v]:
                continue
            # Check if v has any neighbors that are already selected
            # neighbors = adj[v]
            # if np.any(selected[neighbors]):
            #     continue
            # 
            # Faster: check intersection
            if np.any(selected & adj[v]):
                continue
            
            # Select v
            selected[v] = True
            count += 1
            
            # Block neighbors (optional optimization, but selected check is enough)
            # selected[adj[v]] = True # This would be wrong, we only skip if already selected or neighbor of selected.
            # Actually, the standard greedy is:
            # if v is not adjacent to any previously selected vertex, select it.
            # So the check `np.any(selected & adj[v])` is correct.
            
        return count

    # 1. Initial Scores: Weighted Local Degree (WLD)
    # Score = -degree - alpha * (sum of neighbor degrees) / degree
    deg = adj.sum(axis=1).astype(float)
    neighbor_deg_sum = adj @ deg
    
    # Avoid division by zero
    safe_deg = deg + 1e-6
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # Heuristic: Minimize degree, and minimize neighbor degree sum (to avoid picking hubs)
    # Actually, picking a low degree vertex is good. Picking a vertex whose neighbors are high degree is GOOD 
    # because it blocks the hubs. So we want to MAXIMIZE neighbor degree sum?
    # Let's re-evaluate.
    # If I pick v, I block N(v).
    # If N(v) contains hubs, I block hubs. Hubs would block many vertices if they were picked.
    # So blocking hubs is GOOD.
    # So I want to pick v if N(v) has high total degree.
    # So Score should be: -degree(v) + beta * sum_deg_neighbors(v)
    
    # Let's try a combination.
    # Primary: Low degree.
    # Secondary: High neighbor degree sum.
    
    scores = -deg + 0.5 * neighbor_deg_sum
    
    # Normalize scores to [0, 1] for better perturbation handling? 
    # Not strictly necessary, but helps with annealing temperature.
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    else:
        scores = np.zeros(n)
        
    # Add small random noise to break symmetries
    scores += np.random.uniform(0, 0.01, n)
    
    best_scores = scores.copy()
    best_size = run_greedy(best_scores)
    
    # 2. Local Search
    current_scores = best_scores.copy()
    current_size = best_size
    
    # Simulated Annealing parameters
    T = 0.1  # Initial temperature
    cooling_rate = 0.999
    min_T = 0.001
    
    # Random seed for reproducibility in local search? 
    # The task implies deterministic or at least stable performance. 
    # Using a fixed seed for the internal search makes the result deterministic for a given graph.
    random.seed(42)
    np.random.seed(42)
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Perturbation: Swap two random vertices' scores
        i, j = random.sample(range(n), 2)
        
        # Save old scores
        old_i = current_scores[i]
        old_j = current_scores[j]
        
        # Swap
        current_scores[i] = old_j
        current_scores[j] = old_i
        
        # Evaluate
        new_size = run_greedy(current_scores)
        
        # Acceptance Criteria
        if new_size > current_size:
            # Better, accept
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_scores = current_scores.copy()
        else:
            # Worse or equal
            delta = new_size - current_size
            # Accept with probability exp(delta / T)
            # Since delta is negative (integer steps), this is exp(-|delta|/T)
            if random.random() < np.exp(delta / T):
                current_size = new_size
            else:
                # Reject, swap back
                current_scores[i] = old_i
                current_scores[j] = old_j
                
        # Cool down
        T = max(min_T, T * cooling_rate)
        
        # Periodic restart to escape deep local minima?
        # If no improvement for a while, re-initialize with noise?
        # For now, simple SA.

    return best_scores
