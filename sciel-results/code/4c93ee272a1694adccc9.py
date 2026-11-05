import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Instead of using a static formula based on initial degrees, we simulate the 
    optimal greedy selection process (Minimum Degree Greedy) directly.
    
    1. We maintain a "remaining" set of vertices.
    2. At each step, we find the vertex with the minimum degree within the remaining subgraph.
    3. We select this vertex, add it to our independent set, and remove it along with all 
       its neighbors from the remaining set.
    4. We repeat until no vertices remain.
    5. The priority score is assigned such that vertices selected earlier in this simulation 
       have higher priority. This ensures that when the external greedy decoder runs, 
       it will pick vertices in the same order as our internal simulation, 
       reproducing the high-quality independent set found by the min-degree heuristic.
    
    This is structurally different from static scoring because it accounts for the 
    dynamic change in the graph structure as vertices are removed. It essentially 
    "solves" the greedy selection problem and encodes the solution order into the scores.
    """
    start_time = time.time()
    
    # Convert to boolean array for efficient operations
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Initialize scores with a large negative number (will be overwritten)
    scores = np.full(n, -1e9)
    
    # Track which vertices are still in the "remaining" graph
    remaining = np.ones(n, dtype=bool)
    
    # Count of selected vertices
    selected_count = 0
    
    # We need to simulate the greedy selection.
    # For n=200, a loop of up to 200 iterations is fine.
    # In each iteration, we need to find the min-degree vertex in the induced subgraph of 'remaining'.
    
    # Optimization: Instead of recalculating degrees from scratch every time (O(N^2) per step -> O(N^3) total),
    # we can update degrees incrementally or just use a reasonably efficient check since N=200 is small.
    # O(N^3) = 8,000,000 operations is very fast in Python/NumPy.
    
    # However, finding the min degree in the remaining subgraph requires checking adj[u, v] for v in remaining.
    
    while remaining.any():
        # Check time limit
        if time.time() - start_time > 140:
            # Fallback: assign random scores or simple degree-based if we run out of time
            # But N=200 is so small this is unlikely.
            # Just break and return what we have (unselected get low score)
            break
            
        # Calculate degrees for all remaining vertices within the remaining subgraph
        # deg_remaining[i] = sum(adj[i, remaining])
        # This is a matrix-vector product if we treat remaining as a vector, 
        # but we only care about rows where remaining is True.
        
        # Efficient way:
        # degs = adj.dot(remaining) gives degree of each vertex to the remaining set.
        degs = adj.dot(remaining)
        
        # We only care about vertices that are still remaining.
        # Set degs of non-remaining vertices to infinity so they are not picked.
        degs[~remaining] = np.inf
        
        # Find the vertex with the minimum degree
        # If there are ties, any is fine. np.argmin picks the first one.
        u = np.argmin(degs)
        
        # If the min degree is inf, it means no vertices are remaining (should be caught by while)
        if not np.isfinite(degs[u]):
            break
            
        # Select u
        selected_count += 1
        scores[u] = n - selected_count  # Earlier selections get higher scores (n-1, n-2, ...)
        
        # Remove u and its neighbors from 'remaining'
        # Neighbors of u are where adj[u, :] is True.
        # u itself is also removed.
        # We can do: remaining[adj[u, :]] = False; remaining[u] = False
        # Or: remaining = remaining & ~adj[u, :] & ~(np.eye(n)[u])
        
        # Vectorized removal:
        # Create a mask of vertices to remove: u and N(u)
        to_remove = adj[u, :].copy()
        to_remove[u] = True
        
        remaining[to_remove] = False
        
    # For any vertices not assigned a score (due to timeout or logic break), 
    # assign a very low score so they are picked last (or not at all if decoder stops).
    # However, the decoder picks until no more can be picked. 
    # If we didn't assign a score, it's -1e9, which is lower than any assigned score.
    # This is correct: unassigned vertices were not part of the simulated independent set.
    
    return scores

