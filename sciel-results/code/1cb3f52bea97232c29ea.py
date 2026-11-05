import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    
    # Triangular mask for upper triangle
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    def random_graph():
        A = np.zeros((n, n), dtype=np.int8)
        edges = []
        edge_set = set()
        # Generate all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:m]:
            A[u, v] = 1
            A[v, u] = 1
            edges.append((u, v))
            edge_set.add((u, v))
        return A, edges, edge_set

    def get_best_edges(A, edge_set):
        # Return list of edges from the set
        return [list(e) for e in edge_set]

    # Initialize
    random.seed(42)
    A, edges, edge_set = random_graph()
    
    best_max_book = float('inf')
    best_A = A.copy()
    best_edge_set = set(edge_set)
    
    # Precompute indices for faster access?
    # We will use vectorized operations.
    
    iterations = 0
    no_improve_count = 0
    target_book = 7
    
    while time.time() < deadline:
        # Compute B = A @ A
        B = A @ A
        
        # Find max book among existing edges
        # books_matrix[i][j] is valid only if A[i][j] == 1 and i < j
        # We can create a masked matrix
        # To find max, we can just iterate over the upper triangle where A is 1
        
        # Vectorized max finding:
        # Set B to 0 where A is 0
        B_masked = np.where(triu_mask & (A == 1), B, 0)
        current_max_book = np.max(B_masked)
        
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            best_A = A.copy()
            best_edge_set = set(edge_set)
            no_improve_count = 0
            
            if best_max_book <= target_book:
                break
        else:
            no_improve_count += 1
        
        # If we are stuck, we might need to randomize more or restart
        # For now, just continue
        
        # Select edge to remove
        # Find all edges with book size == current_max_book
        # If none, pick random
        max_indices = np.argwhere(B_masked == current_max_book)
        if len(max_indices) > 0:
            # Pick one randomly
            idx = random.randint(0, len(max_indices) - 1)
            u, v = max_indices[idx]
        else:
            # Pick random existing edge
            u, v = random.choice(list(edge_set))
        
        # Select edge to add
        # Candidates: A[i][j] == 0 and i < j
        # We want to minimize the impact.
        # Score for adding (i,j) is B[i][j] (current common neighbors)
        # We pick from candidates with low B[i][j]
        
        # Create a score matrix
        # Set score to B where it's a candidate, else high value
        candidate_mask = triu_mask & (A == 0)
        scores = np.where(candidate_mask, B, 1000)
        
        # Find indices with lowest scores
        # We can flatten and find k smallest
        # But we need to map back to (i,j)
        
        # Simple approach:
        # Get all candidate indices
        cand_indices = np.argwhere(candidate_mask)
        if len(cand_indices) == 0:
            # Graph is complete, shouldn't happen with m=401 < 780
            break
            
        # Get scores for these candidates
        cand_scores = scores[cand_indices[:, 0], cand_indices[:, 1]]
        
        # Sort by score
        # To save time, if many candidates, just pick from a random subset of the best ones
        # Or sort all (300 items is fast)
        sorted_indices = np.argsort(cand_scores)
        
        # Pick from top 10 (lowest scores)
        k = min(10, len(sorted_indices))
        choice_idx = random.randint(0, k - 1)
        sorted_pos = sorted_indices[choice_idx]
        a, b = cand_indices[sorted_pos]
        
        # Apply move
        # Remove (u,v)
        A[u, v] = 0
        A[v, u] = 0
        edge_set.remove((u, v))
        
        # Add (a,b)
        A[a, b] = 1
        A[b, a] = 1
        edge_set.add((a, b))
        
        iterations += 1
        
        # Optional: If no improvement for a long time, restart with new random seed?
        # Or just continue.
        
    return get_best_edges(best_A, best_edge_set)
