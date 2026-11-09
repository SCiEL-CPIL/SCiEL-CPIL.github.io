import numpy as np

def priority(adjacency):
    A = adjacency.astype(int)
    n = A.shape[0]
    
    # Compute degrees
    degrees = np.sum(A, axis=1)
    
    # Core Number Algorithm
    # We maintain a list of vertices and their current degrees
    # We process vertices in order of increasing current degree
    
    # Initialize core numbers to 0
    core_numbers = np.zeros(n, dtype=int)
    
    # Create a list of vertices sorted by initial degree
    # We need to be careful with ties and dynamic updates
    # A standard efficient implementation uses a bucket queue or similar, 
    # but for n=200, a simpler O(n^2) or O(n*m) approach is fine.
    
    # Let's use a simple iterative peeling
    # 1. Sort vertices by degree
    # 2. For each vertex in sorted order, if its degree is less than current max core, 
    #    remove it and decrement neighbors.
    
    # Actually, a simpler robust method for small n:
    # Repeat n times:
    #   Find vertex with minimum current degree
    #   Assign its core number to its current degree
    #   Remove it and update neighbors
    
    current_A = A.copy()
    current_degrees = degrees.copy()
    removed = np.zeros(n, dtype=bool)
    
    for _ in range(n):
        # Find index of vertex with minimum degree among not-removed
        # Mask removed vertices with inf
        masked_degrees = np.where(removed, np.inf, current_degrees)
        min_idx = np.argmin(masked_degrees)
        
        # If all removed, break (should not happen)
        if removed[min_idx]:
            break
            
        core_numbers[min_idx] = current_degrees[min_idx]
        removed[min_idx] = True
        
        # Update neighbors
        neighbors = np.where(current_A[min_idx] == 1)[0]
        for nb in neighbors:
            if not removed[nb]:
                current_degrees[nb] -= 1
                current_A[min_idx, nb] = 0
                current_A[nb, min_idx] = 0
                # Note: We don't strictly need to update current_A for degree counting 
                # if we track degrees separately, but it helps if we recompute.
                # Here we just decrement the stored degree.
                
    # Score: We want high scores for vertices likely to be in MIS.
    # Low core number = periphery = good for MIS.
    # Low degree = good for MIS.
    # Let's combine them.
    # score = -core_numbers - 0.1 * current_degrees_initial? 
    # The initial degrees are `degrees`.
    
    # Let's try: score = -core_numbers
    # Or score = -(core_numbers + degrees)
    
    # Let's use a weighted sum. Core number is a strong indicator of "density participation".
    scores = -(core_numbers.astype(float) + degrees.astype(float))
    
    return scores
