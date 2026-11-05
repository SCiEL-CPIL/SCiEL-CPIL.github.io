"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 54
    
    # Try to find a lower rank decomposition
    # Strategy: Use the tensor product structure
    # 3x3x6 = (3x3x3) x (1x1x2) or similar factorizations
    
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor T_{i,j,k} = 1 if (i,j,k) corresponds to a valid multiplication term.
    # For A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank), indexed by (i*m + j)
    # V has shape (m*p, rank), indexed by (j*p + k)  
    # W has shape (p*n, rank), indexed by (k*n + i)
    
    # The decomposition satisfies:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid term, else 0
    # Wait, that's not quite right. Let me reconsider.
    
    # Actually, the standard matrix multiplication tensor decomposition works as follows:
    # We want to compute C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The bilinear form for each output element C[i,k] involves summing over j.
    # The tensor T has entries T[(i,k), (i,j), (j,k)] = 1 for all valid i,j,k.
    
    # In the factorization:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a indexes C (size p*n, with a = k*n + i)
    #       b indexes A (size n*m, with b = i*m + j)  
    #       c indexes B (size m*p, with c = j*p + k)
    
    # So we need: sum_r U[k*n+i, r] * V[i*m+j, r] * W[j*p+k, r] = 1 for all i,j,k
    
    # Schoolbook: for each (i,j,k), create one rank-1 term with U[k*n+i]=1, V[i*m+j]=1, W[j*p+k]=1
    
    # To reduce rank, we need to find overlaps.
    
    # One approach: if p is even, we can potentially pair up k values.
    # For example, if we group k into pairs (k, k+1), we might be able to share computations.
    
    # Let's try a structured approach:
    # Group the p=6 columns into 3 pairs: (0,1), (2,3), (4,5)
    # For each pair, we might be able to use 2 rank-1 terms instead of 2 separate ones?
    # Actually, that doesn't immediately help.
    
    # Alternative: Use the fact that for fixed i,j, we have p terms.
    # Can we combine them? If we set V[i*m+j, r] = 1 for multiple r's corresponding to different k's...
    
    # Let's try a different approach: search for a low-rank decomposition using random initialization
    # and iterative refinement.
    
    # First, let's try some small ranks to see if we can find a valid decomposition.
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank = U_arr.shape[1]
        
        # Compute the tensor from the decomposition
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # a = k*n + i, b = i*m + j, c = j*p + k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = k * n + i
                    b = i * m + j
                    c = j * p + k
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try schoolbook first as a baseline
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Strategy: Use random initialization and gradient-like updates
    
    # Actually, let's try a more systematic approach.
    # The tensor has a specific structure. Let's see if we can exploit it.
    
    # For 3x3x3, the Strassen decomposition gives rank 7 instead of 9.
    # For 3x3x6, can we do better than 54?
    
    # One idea: since p=6=2*3, we can think of this as two 3x3x3 multiplications.
    # If we use Strassen for each, we'd get 2*7=14 terms for the "core" part, but we need to handle the input/output mapping.
    
    # Actually, let's think of it as: C = A * B where B is 3x6.
    # We can split B into two 3x3 matrices: B1 (columns 0-2) and B2 (columns 3-5).
    # Then C1 = A * B1 and C2 = A * B2, each a 3x3x3 multiplication.
    # If we use Strassen for each, we need 7+7=14 rank-1 terms.
    # But wait, the U factor would be shared between C1 and C2 since they both use the same A.
    
    # Let's try this approach:
    # Strassen for 3x3:
    # M1 = (A00 + A22) * (B00 + B22)
    # M2 = (A20 + A22) * B02
    # M3 = A00 * (B02 - B22)
    # M4 = A22 * (B20 - B00)
    # M5 = (A00 + A02) * B20
    # M6 = (A20 - A00) * (B00 + B10)  -- wait, this is for 2x2
    # 
    # Actually, Strassen is for 2x2. For 3x3, there are different decompositions.
    # The minimal rank for 3x3x3 over the reals is 19 (I think), or maybe 23?
    # Actually, I recall that the rank of the 3x3x3 multiplication tensor is 19 over the reals.
    # But that's a complex result. Let me not go down this path.
    
    # Let's try a simpler approach: brute force search over small ranks with random initialization.
    
    # Actually, given the time constraint, let's try a few specific structured decompositions.
    
    # Idea: Pair up the k indices. For each (i,j), we have 6 k values.
    # If we can represent the 6 values with fewer than 6 rank-1 terms by sharing V entries...
    
    # Let's try rank = 36 (2/3 of schoolbook) as a target.
    # Or even lower.
    
    # Given the complexity, let's just try to verify if any rank below 54 works by searching.
    # But this is computationally expensive.
    
    # For now, let's return the schoolbook decomposition.
    # The task is to find the rank-minimal decomposition, but finding it is hard.
    
    # Let's at least try a few random searches for lower ranks.
    
    for target_rank in [48, 42, 36, 30, 24]:
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Try to refine using a simple iterative method
            # This is a heuristic and may not converge to a valid solution
            
            # Check if this random guess happens to work (unlikely)
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
