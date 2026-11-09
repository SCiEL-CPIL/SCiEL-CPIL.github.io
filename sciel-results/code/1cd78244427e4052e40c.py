import time
import sys
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    
    Strategy: Start with schoolbook (rank 72), then attempt to reduce rank by
    identifying linear dependencies in the tensor space and eliminating redundant
    terms. Use iterative rank-reduction with time budgeting.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with schoolbook
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Attempt rank reduction
    while time.time() < deadline - 10:
        result, rank = _try_reduce_rank(best_result, n, m, p, deadline)
        if rank < best_rank:
            best_result = result
            best_rank = rank
            # Try to reduce further
            continue
        else:
            break
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    """
    Construct the schoolbook (naive) matrix multiplication tensor decomposition.
    """
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(current_result, n, m, p, deadline):
    """
    Attempt to reduce the rank of the current decomposition by finding
    linear dependencies among the rank-1 tensor components.
    
    Returns (new_result, new_rank) if successful, else (None, current_rank).
    """
    U = np.array(current_result["U"], dtype=float)
    V = np.array(current_result["V"], dtype=float)
    W = np.array(current_result["W"], dtype=float)
    current_rank = U.shape[1]
    
    # The tensor T is represented as: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We can try to find if any column is a linear combination of others.
    
    # Build the full tensor and check for rank reduction
    # For each output index (i,j,k), T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try to find a column that can be expressed as a combination of others
    # This is equivalent to finding a linear dependency in the column space
    # of the "unfolding" of the tensor
    
    # A simpler approach: try removing each column and see if the remaining
    # columns can still span the same tensor (up to linear transformation)
    
    # For efficiency, we'll use a heuristic: look for columns that are
    # linearly dependent in a structured way
    
    # Method: Try to find two columns that differ only in one factor
    # If U[:,c1] and U[:,c2] differ in one row, and similarly for V and W,
    # we might be able to combine them
    
    # More systematic approach: use the fact that for matrix multiplication,
    # certain symmetries exist. For 3x4x6, we can try block-based reductions.
    
    # Try a specific reduction pattern: if we have terms for (i,j,k) and (i',j,k)
    # where i' = i+1 (mod n), we might combine them if the other factors allow.
    
    # Let's try a different approach: use randomized linear algebra to find
    # the minimal rank of the tensor
    
    # For now, implement a simple check: can we reduce by 1?
    # Try removing each column and checking if the tensor is preserved
    
    for col_to_remove in range(current_rank):
        if time.time() > deadline - 15:
            break
        
        # Create reduced matrices
        U_red = np.delete(U, col_to_remove, axis=1)
        V_red = np.delete(V, col_to_remove, axis=1)
        W_red = np.delete(W, col_to_remove, axis=1)
        
        # Check if the reduced decomposition still gives the same tensor
        # T_original[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # T_reduced[a,b,c] = sum_r U_red[a,r]*V_red[b,r]*W_red[c,r]
        
        # For efficiency, check on a sample of (a,b,c) triples
        # a ranges over n*m, b over m*p, c over p*n
        # But we only need to check the "valid" triples where a=i*m+j, b=j*p+k, c=k*n+i
        
        valid_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_triples.append((a, b, c))
        
        # Sample some triples to check
        sample_size = min(50, len(valid_triples))
        indices = np.random.choice(len(valid_triples), size=sample_size, replace=False)
        
        all_match = True
        for idx in indices:
            a, b, c = valid_triples[idx]
            # Original value
            orig_val = np.sum(U[a, :] * V[b, :] * W[c, :])
            # Reduced value
            red_val = np.sum(U_red[a, :] * V_red[b, :] * W_red[c, :])
            if abs(orig_val - red_val) > 1e-10:
                all_match = False
                break
        
        if all_match:
            # Verify on all triples
            full_match = True
            for a, b, c in valid_triples:
                orig_val = np.sum(U[a, :] * V[b, :] * W[c, :])
                red_val = np.sum(U_red[a, :] * V_red[b, :] * W_red[c, :])
                if abs(orig_val - red_val) > 1e-10:
                    full_match = False
                    break
            
            if full_match:
                # Success! Convert back to lists with integer/half-integer entries
                new_U = U_red.tolist()
                new_V = V_red.tolist()
                new_W = W_red.tolist()
                
                # Round to nearest half-integer
                new_U = [[round(x * 2) / 2 for x in row] for row in new_U]
                new_V = [[round(x * 2) / 2 for x in row] for row in new_V]
                new_W = [[round(x * 2) / 2 for x in row] for row in new_W]
                
                return {"U": new_U, "V": new_V, "W": new_W}, current_rank - 1
    
    # No single column removal worked. Try a more sophisticated approach:
    # Look for pairs of columns that can be combined
    
    # For now, return failure
    return None, current_rank

# EVOLVE-BLOCK-END
