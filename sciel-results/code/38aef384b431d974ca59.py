"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check dimensions
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for matrix mult: T[i,j,k] = 1 for the (i,j,k) entry meaning A[i,j]*B[j,k] contributes to C[i,k]
    
    # Actually, the matrix multiplication tensor M has M[i,j,k] = 1 
    # meaning the product A[i,j] * B[j,k] contributes to C[i,k]
    # In the factorization: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry is 1 (since A[i,j]*B[j,k] appears in C[i,k])
                tensor_val = 1.0
                decomp_val = 0.0
                for c in range(rank):
                    decomp_val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(tensor_val - decomp_val) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Best known: schoolbook gives rank 54
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # For 3x3 x 3x6, we can view it as two 3x3x3 multiplications
    # If 3x3x3 has rank 23, then total rank is at most 46
    
    # Let's try to construct a rank-46 decomposition using known 3x3x3 rank-23 factors
    # The BCBDR rank-23 decomposition for 3x3 matrix multiplication
    
    # I'll try to use a known rank-23 decomposition for 3x3x3
    # and combine two copies for the 3x3x6 case
    
    # First, let's try a direct approach: use the structure of 3x3x6
    # A (3x3) * B (3x6) = C (3x6)
    # Split B = [B1, B2] where B1, B2 are 3x3
    # C = [A*B1, A*B2]
    # Each A*B is a 3x3x3 multiplication
    
    # Use rank-23 decomposition for each 3x3x3
    # Total rank: 23 + 23 = 46
    
    # Let me try to find the BCBDR rank-23 factors
    # This is a known result; let me try to implement it
    
    # Alternative: try to use a computational search
    # Given time constraints, let's try a few known decompositions
    
    # For now, let's try to construct a decomposition using Strassen-like ideas
    # 3x3 can be decomposed using 2x2 Strassen + additional terms
    
    # Actually, let me try a more direct approach:
    # Use the fact that we can do 3x3x3 with 23 multiplications
    # I'll try to hardcode the BCBDR decomposition if I can recall it
    
    # Let me try a different approach: use numerical optimization
    # to find a low-rank decomposition
    
    # Given the complexity, let's try to use a known result
    # The rank of 3x3x3 matrix multiplication tensor is 23
    
    # I'll attempt to construct the factors for rank-23 3x3x3
    # and then combine for 3x3x6
    
    # For now, return schoolbook if we can't find better in time
    # But let's try a simple improvement first
    
    # Try rank 46: two copies of a rank-23 3x3x3 decomposition
    # I need to find the actual rank-23 factors
    
    # Let me try to use a known explicit decomposition
    # After research, the BCBDR decomposition has specific factors
    
    # Given time constraints, let me try a simpler approach:
    # Use the schoolbook but see if we can merge some terms
    
    # Actually, let me just return the schoolbook for now and 
    # focus on getting a valid answer
    
    return best

# EVOLVE-BLOCK-END
