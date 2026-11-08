import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Store best result found so far
    best_rank = schoolbook_rank
    best_solution = None
    
    # Generate schoolbook solution
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Strategy: Use block structure of 2x3x6 tensor
    # The 3x6 factor can be viewed as three 3x2 sub-blocks
    # Try to use rank-5 decomposition for 2x2x2 (Strassen-like) on sub-blocks
    
    # First, try some known structured decompositions
    # For 2x3x6, we can think of it as computing C = A*B where A is 2x3, B is 3x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: treat B as having 3 columns of 2x3 blocks, or 6 columns of 2x1
    # Actually, let's think differently.
    
    # The tensor has indices (i,j,k) with i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}
    # We want to decompose T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Try rank 12, 11, 10, ... down to some lower bound
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 18, 12) = 18? No, that's not right.
    # The tensor rank can be less than any of these.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x6, we can try to find good decompositions.
    
    # Let's try a few structured approaches:
    
    # Approach 1: Use Strassen-like for 2x2x2 on overlapping blocks
    # But first, let's try random search with faster verification for lower ranks
    
    # Try ranks from 35 down to 1
    # Start higher because schoolbook is 36, and minimal rank is likely > 15
    for rank in range(35, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Try random candidates with sparse structure
        max_candidates = 5000
        if time.time() > deadline - 20.0:
            max_candidates = 500
            
        for _ in range(max_candidates):
            if time.time() > deadline - 2.0:
                break
                
            # Generate sparse random matrices
            # Use mostly zeros with some non-zero entries
            density = 0.3  # 30% non-zero entries
            U = _generate_sparse_matrix(n * m, rank, density)
            V = _generate_sparse_matrix(m * p, rank, density)
            W = _generate_sparse_matrix(p * n, rank, density)
            
            # Verify
            if _verify_fast(n, m, p, U, V, W):
                best_rank = rank
                best_solution = {"U": U, "V": V, "W": W}
                return best_solution
                
    # If we haven't found anything better, return the valid schoolbook solution
    # which is already stored in best_solution
    return best_solution

def _generate_sparse_matrix(rows, cols, density):
    """Generate a sparse matrix with entries in {-1, 0, 1}"""
    M = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if random.random() < density:
                M[i][j] = random.choice([-1, 1])
    return M

def _verify_fast(n, m, p, U, V, W, num_tests=20):
    """
    Fast verification using NumPy vectorization.
    """
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, size=(n, m), dtype=np.int64)
        B = np.random.randint(-3, 4, size=(m, p), dtype=np.int64)
        
        C_direct = A @ B
        
        # Compute decomposed multiplication
        # LA[i, r] = sum_j U[i*m+j, r] * A[i, j]
        U_reshaped = U_arr.reshape(n, m, -1)
        V_reshaped = V_arr.reshape(m, p, -1)
        W_reshaped = W_arr.reshape(p, n, -1)
        
        LA = np.einsum('ij,ijr->ir', A, U_reshaped)
        LB = np.einsum('jk,jkr->kr', B, V_reshaped)
        
        C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
        
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
            
    return True

def _try_structured_fallback(n, m, p):
    """
    Construct a valid schoolbook-like decomposition (rank 36) to ensure a valid return.
    This fixes the IndexError from the previous version which had mismatched dimensions.
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
# EVOLVE-BLOCK-END
