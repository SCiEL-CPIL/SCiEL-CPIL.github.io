import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110  # 110s safety margin
    
    # The multiplication tensor T[i,j,k] = delta_{j,k} maps A[i,j]*B[j,k] -> C[i,k]
    # We seek rank-r decomposition: T = sum_{r} u_r * v_r * w_r
    # where u_r[i,j], v_r[j,k], w_r[k,i] (note W uses [k*n+i] indexing)
    
    # Known lower bounds: rank >= max(n*m, m*p, p*n) for general matrices
    # For 2x4x6: lower bound is max(8, 24, 12) = 24
    # Strassen gives 42. Let's search for better.
    
    # Strategy: Use structured decomposition based on tensor train / MPS
    # Think of the tensor as a sequence of matrices across indices
    
    # First, try the known good Strassen-based solution (rank 42)
    def make_strassen_solution():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1
            
            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            
            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            
            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            
            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = make_strassen_solution()
    
    # Try to find lower rank solutions using structured search
    # Key insight: group by the middle index j
    # For each j, we need to decompose the partial products
    
    # Try rank values below 42
    for target_rank in range(36, 42):
        if time.time() > deadline:
            break
        
        # Try a structured decomposition
        # Split into blocks based on j index
        # j=0,1 use one set of terms, j=2,3 use another
        
        # Approach: use the fact that A has 2 rows, B has 6 cols
        # For each column k of C, we need sum_j A[:,j]*B[j,k]
        
        # Try a hybrid: Strassen for 2x2 blocks but with sharing
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Use 7 terms per 2x2 block but try to share some
        # Block (i=0..1, j=0..1, k=0..1): 7 terms
        # Block (i=0..1, j=0..1, k=2..3): 7 terms  
        # Block (i=0..1, j=0..1, k=4..5): 7 terms
        # Block (i=0..1, j=2..3, k=0..1): 7 terms
        # Block (i=0..1, j=2..3, k=2..3): 7 terms
        # Block (i=0..1, j=2..3, k=4..5): 7 terms
        # Total: 42
        
        # To reduce, try combining some blocks
        # If we can share 1 term between adjacent k-blocks, save 6 terms
        
        # Try: use 6 terms for some blocks by exploiting structure
        def add_6_term_block(i_start, j_start, k_start, offset):
            """Try a 6-term decomposition for a 2x2x2 sub-block"""
            # This may not be exact for all cases, but let's try
            # Standard Strassen uses 7. Can we do 6?
            # For 2x2x2, the tensor rank is actually 7 (proven), so 6 won't work
            # But maybe with the larger structure we can share
            pass
        
        # Alternative: try different block partitioning
        # What if we use 1x4 blocks instead of 2x2?
        
        # Try: decompose by fixing i
        # For i=0: need A[0,j]*B[j,k] for all j,k
        # For i=1: need A[1,j]*B[j,k] for all j,k
        
        # Each i-row needs rank at least 4*6/4 = 6? No...
        # The tensor for fixed i is a 4x6 matrix multiplication (outer product sum)
        # Its rank as a bilinear map is at most 4*6=24, at least max(4,6)=6
        
        # Better approach: use the known result that for m x n x p,
        # rank can sometimes be less than m*n*p
        
        # Let's try a direct construction with rank 36
        # Use 6 terms per "super-block" of size 2x4x2 or similar
        
        # Actually, let me try a different structure:
        # Group j indices: (j=0,1) and (j=2,3)
        # Group k indices: (k=0,1), (k=2,3), (k=4,5)
        # For each (j-group, k-group), use a 2x2x2 Strassen (7 terms)
        # That's 2*3*7 = 42 (same as before)
        
        # To save terms, note that for fixed j-group, the k-groups are independent
        # but the i-dimension is shared. Maybe we can factor differently.
        
        # Try: for each j-group (2 groups), decompose the 2x6 multiplication
        # A[2x2] * B[2x6] -> C[2x6]
        # This is equivalent to 2 separate 2x1 * 1x6 multiplications summed
        # Or we can think of it as matrix multiplication 2x2 * 2x6
        
        # The rank of 2x2*2x6 multiplication:
        # Schoolbook: 2*2*6 = 24
        # Can we do better? Using Strassen-like ideas on the 2x2 part:
        # Treat B as 3 columns of 2x2 matrices
        # Each 2x2*2x2 costs 7, so 3*7 = 21 per j-group
        # Total: 2*21 = 42 (same)
        
        # Hmm, let's try a completely different approach:
        # Use the fact that we only have 2 rows in A
        # C[0,k] = sum_j A[0,j]*B[j,k]
        # C[1,k] = sum_j A[1,j]*B[j,k]
        
        # Define for each k:
        # C[:,k] = A^T * B[:,k]
        
        # This is a 2x4 times 4x1 multiplication for each k
        # Total: 6 independent 2x4*4x1 multiplications
        # Each has rank at most 2*4=8, at least max(2,4,1)=4
        
        # If we could do each with rank 4, total = 24
        # But they share the A matrix, so we might share terms
        
        # Let me try a construction with rank 24
        # For each k, use 4 terms
        # Term r for column k: u_r(i,j) * v_r(j,k) * w_r(k,i)
        
        # Actually, the cleanest approach:
        # Use the standard result that the rank of matrix multiplication
        # [n x m] * [m x p] is at most n*m*p and at least max(n*m, m*p, p*n)
        
        # For our case: lower bound = max(8, 24, 12) = 24
        # Upper bound from Strassen: 42
        
        # Let's try to achieve rank 24 or close to it
        
        # Construction for rank 24:
        # For each (j,k) pair, we have a term
        # u(i,j) = delta_{i,?} ... this gets complicated
        
        # Simpler: use the "diagonal" construction
        # For each j and k, create a term that handles A[*,j] * B[j,k]
        # But that's m*p = 24 terms, each of rank 1 in the (i) dimension
        # Term (j,k): U[:,(j,k)] = e_j (in the j-index), V[(j,k),:] = e_k^T (in k-index)
        # W[(j,k),:] maps to the i-dimension
        
        # Let's construct this explicitly
        U24 = [[0] * 24 for _ in range(n * m)]
        V24 = [[0] * 24 for _ in range(m * p)]
        W24 = [[0] * 24 for _ in range(p * n)]
        
        for j in range(m):
            for k in range(p):
                r = j * p + k  # term index
                # U: for this term, U[i, j] should be 1 if we're selecting A[i,j]
                # Actually, the decomposition is:
                # C[i,k] = sum_r U[i,j_r] * V[j_r,k_r] * W[k_r,i_r]
                # where (j_r, k_r, i_r) are determined by r
                
                # For term r=(j,k):
                # U[i, j] = 1 for all i (selects A[i,j])
                # V[j, k] = 1 (selects B[j,k])  
                # W[k, i] = 1 for all i... but this doesn't work because
                # we need W to map back to the correct i
                
                # The issue: with this naive approach, each term contributes
                # A[i,j]*B[j,k] to ALL output positions, not just C[i,k]
                
                # We need W to isolate the (k,i) pair
                # W[k*n + i] = 1 for the correct i
                # But i is determined by which row of A we used
                
                # This means we need separate terms for each (i,j,k) triple
                # which gives n*m*p = 48 terms (schoolbook)
        
        # So rank 24 with this structure doesn't work directly.
        # We need a more clever decomposition.
        
        # Let's stick with the Strassen-based rank 42 for now,
        # but try to optimize the block structure
        
        # Try: can we use 6 terms for a 2x2x2 block by being clever?
        # No, rank of 2x2x2 matmul tensor is provably 7.
        
        # But maybe we can share terms between blocks
        # If two blocks share a term, we save 1
        
        # For now, return the best we have
        if target_rank < best_rank:
            # We haven't found a valid solution yet, keep searching
            pass
    
    # Return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}
