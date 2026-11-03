import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 6]
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds, with safety margin

def _schoolbook_rank48():
    n, m, p = DIMENSIONS
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank42():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    B_forms = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    C_coeffs = [
        [[1, 0, 0, 1, -1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0]],
        [[0, 1, 0, 1, 0, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    ]
    
    for b in range(6):
        if b < 3:
            j_offset_A = 0
            j_offset_B = 0
        else:
            j_offset_A = 2
            j_offset_B = 2
            
        k_offset = (b % 3) * 2
        
        for s in range(7):
            col = b * 7 + s
            
            for (i_loc, j_loc), coeff in A_forms[s].items():
                i_global = i_loc
                j_global = j_loc + j_offset_A
                U[i_global * m + j_global][col] += coeff
                
            for (j_loc, k_loc), coeff in B_forms[s].items():
                j_global = j_loc + j_offset_B
                k_global = k_loc + k_offset
                V[j_global * p + k_global][col] += coeff
                
            for i_loc in range(2):
                for k_loc in range(2):
                    coeff = C_coeffs[i_loc][k_loc][s]
                    if coeff != 0:
                        i_global = i_loc
                        k_global = k_loc + k_offset
                        W[k_global * n + i_global][col] += coeff
                        
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is correct."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # and compare with the true matmul tensor: T[i,j,k] = 1 if the index pattern matches
    # Actually, the tensor for matmul is: T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}
    # No - the matmul tensor is T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Wait, that's not right either.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # and the decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by checking a few random instances
    # For random A (n x m) and B (m x p), check A@B = sum_r (U_r^T A) * (B V_r) * W_r pattern
    
    # Better: just check the tensor identity directly
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank17_embedding():
    """
    Try to use the rank-17 4x4x4 decomposition embedded into 2x4x6.
    
    Strategy: The 2x4x6 matmul can be seen as a sub-tensor of 4x4x4.
    We pad A to 4x4 (rows 2,3 are zero), B to 4x4 (cols 4,5... wait B is 4x6, pad to 4x4? No.)
    
    Actually, let's think differently. The matmul tensor for 2x4x6 is:
    T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    We can embed this into a 4x4x4 tensor by:
    - Mapping i: 0->0, 1->1 (rows 2,3 of the 4x4 are unused/zero)
    - Mapping j: 0->0, 1->1, 2->2, 3->3 (all 4 columns used)
    - Mapping k: 0->0, 1->1, 2->2, 3->3, 4->?, 5->? (only 4 cols in 4x4)
    
    This doesn't work directly because p=6 > 4.
    
    Alternative: Use two overlapping 4x4x4 decompositions or a different embedding.
    
    Actually, a better approach: use the rank-17 decomposition of 4x4x4 and
    exploit the fact that our A has only 2 rows. We can write:
    
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x6) = [B1; B2] where B1, B2 are 2x6
    
    C = A1*B1 + A2*B2 (2x6)
    
    For each 2x2 * 2x6 product, we can further split:
    A1*B1: A1 is 2x2, B1 is 2x6 = [B1a | B1b] where B1a, B1b are 2x3
    This gets complicated.
    
    Let me try a different known result: the rank of 2x4x6 matmul.
    Known lower bounds: max(4, 4*6/2, ...) 
    The trivial upper bound is 48 (schoolbook).
    Strassen gives 42 as we have.
    
    Let me try to see if we can do better by using the structure more carefully.
    
    For 2x4x6: split into two 2x2x6 products:
    C = A_left * B_left + A_right * B_right
    where A_left, A_right are 2x2, B_left, B_right are 2x6.
    
    Each 2x2x6 product: we can split B into three 2x2 blocks:
    2x2x6 = 2x2x2 + 2x2x2 + 2x2x2 (three Strassen 2x2x2)
    That gives 7*3 = 21 per 2x2x6, total 42. Same as before.
    
    Can we share between the left and right? The A factors are different, B factors are different.
    Unless we find a common structure...
    
    Let me try: what if we use a 4x4x4 rank-17 decomposition and map it?
    
    Embedding: Consider A' (4x4) where A'[0:2, :] = A, A'[2:4, :] = 0
    Consider B' (4x4) where B'[:, 0:4] = B[:, 0:4], B'[:, 4:] = 0... but B is 4x6, not 4x4.
    
    This doesn't fit. Let me try another approach.
    
    What about using the fact that rank(2x4x6) <= rank(4x4x4) + something?
    
    Actually, let me just try to improve the current 42 by looking for shared terms.
    
    For the two halves:
    Left: A_L (2x2) * B_L (2x6)
    Right: A_R (2x2) * B_R (2x6)
    
    If we could express this with fewer than 14 Strassen terms total, that would help.
    But A_L != A_R and B_L != B_R in general, so no obvious sharing.
    
    Let me try a completely different block structure:
    Split A (2x4) into 2x2 blocks: A = [A11 A12; A21 A22]
    Split B (4x6) into 2x2 blocks: B = [B11 B12 B13; B21 B22 B23]
    
    C = A*B = [A11*B11 + A12*B21, A11*B12 + A12*B22, A11*B13 + A12*B23;
               A21*B11 + A22*B21, A21*B12 + A22*B22, A21*B13 + A22*B23]
    
    Each entry of C is a sum of two 2x2*2x2 products.
    Using Strassen for each pair... but that's what we already do (3 blocks of 2x2x2).
    
    I think 42 might be close to optimal for this structure. Let me just verify the current
    solution works and try minor optimizations.
    
    Actually, wait - let me re-examine. The known best for 2x4x6:
    - Schoolbook: 48
    - Strassen block: 42
    - Could there be a rank-40 or lower?
    
    Lower bound: The tensor has dimension 2*4*6 = 48. The rank must be at least
    max(n*m, m*p, p*n) = max(8, 24, 12) = 24.
    Also, rank >= m * max(n,p) / max(n,p) ... various bounds.
    
    Let me just return the rank-42 solution which is verified to work.
    """
    return None  # No improvement found yet

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE - 5.0:
        return _schoolbook_rank48()
    
    # Try the rank-42 Strassen solution
    result = _strassen_rank42()
    
    # Verify it
    if _verify_decomposition(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback
    return _schoolbook_rank48()
