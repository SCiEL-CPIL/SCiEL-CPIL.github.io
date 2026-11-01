"""Construct integer vectors in R^11 forming a kissing configuration."""
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin for evaluator kill at 150s
    
    # Generate roots of D_11: all vectors with two non-zero coordinates
    # of magnitude 1 and opposite or same signs, in all pairs of coordinates
    vectors = []
    
    # For each pair of coordinates (i, j) with i < j
    for i in range(D):
        for j in range(i + 1, D):
            # Four sign combinations: (+1, +1), (+1, -1), (-1, +1), (-1, -1)
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Convert to array
    vecs = np.array(vectors, dtype=np.int64)
    
    # Verify constraints
    # Max norm: each vector has norm sqrt(2)
    # Min pairwise distance: need to check
    
    # Compute pairwise squared distances efficiently using matrix operations
    # ||v - w||^2 = ||v||^2 + ||w||^2 - 2*v.w
    # All vectors have norm^2 = 2
    # So ||v - w||^2 = 4 - 2*v.w
    # We need ||v - w|| >= max_norm = sqrt(2)
    # So we need ||v - w||^2 >= 2
    # i.e., 4 - 2*v.w >= 2
    # i.e., v.w <= 1
    
    # The inner product of two distinct D_11 roots is in {-2, -1, 0, 1, 2}
    # It equals 2 only if they are the same vector
    # It equals -2 only if they are negatives
    # For distinct vectors, max inner product is 1 (when they share one coordinate with same sign)
    
    # Verify by checking max inner product among distinct vectors
    k = len(vecs)
    
    # Use chunked computation to avoid memory issues
    chunk_size = 1000
    max_ip = -1000
    
    for start in range(0, k, chunk_size):
        end = min(start + chunk_size, k)
        chunk = vecs[start:end]
        
        if time.time() > deadline:
            break
        
        # Compute inner products with all other vectors
        # We only need to check against vectors after this chunk to avoid redundancy
        # But for simplicity, let's just check the max off-diagonal
        
        # Inner product matrix for this chunk vs all
        ips = chunk @ vecs.T
        
        # Mask diagonal (self inner products)
        row_indices = np.arange(end - start)
        col_indices = np.arange(start, end)
        ips[row_indices, col_indices] = -1000
        
        current_max = np.max(ips)
        max_ip = max(max_ip, current_max)
    
    # For D_11 roots, the max inner product between distinct vectors should be 1
    # This means min distance squared = 4 - 2*1 = 2, and max norm squared = 2
    # So min distance = sqrt(2) >= max norm = sqrt(2). Valid.
    
    # If somehow the constraint is violated (shouldn't happen), filter
    # But D_11 is a well-known valid kissing configuration
    
    return vecs
# EVOLVE-BLOCK-END
