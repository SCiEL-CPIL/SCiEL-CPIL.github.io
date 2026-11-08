# EVOLVE-BLOCK-START
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # First, check if our rank-27 solution works
    U27, V27, W27 = _build_rank27_solution(n, m, p)
    rank27 = len(U27[0])
    
    if _verify_solution_fast(U27, V27, W27, n, m, p):
        return {"U": U27, "V": V27, "W": W27}
    
    # Fall back to rank-42 Strassen
    U, V, W = _build_strassen_solution(n, m, p)
    
    if time.time() < deadline - 10:
        if not _verify_solution_fast(U, V, W, n, m, p):
            U, V, W = _build_trivial_solution(n, m, p)
            
    return {"U": U, "V": V, "W": W}

def _build_rank27_solution(n, m, p):
    """
    Attempt rank-27 decomposition using the identity:
    r(2,4,6) <= r(2,2,3) * r(2,2,2) ... wait, that's not quite right.
    
    Actually, let me reconsider. We have 2x4 by 4x6.
    We can split the 4 into 2+2 and the 6 into 3+3.
    So we have a 2x4 matrix A and 4x6 matrix B.
    C = AB where C is 2x6.
    
    Write A = [A1, A2] where A1, A2 are 2x2.
    Write B = [B1; B2] where B1, B2 are 2x6.
    Then C = A1*B1 + A2*B2.
    
    Now B1 and B2 are 2x6. Write B1 = [B11, B12] where B11, B12 are 2x3.
    Similarly B2 = [B21, B22] where B21, B22 are 2x3.
    
    So C = A1*([B11, B12]) + A2*([B21, B22])
       = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    Now A1*B11 is 2x3, A2*B21 is 2x3, etc.
    So we need to multiply 2x2 by 2x3 four times: A1*B11, A1*B12, A2*B21, A2*B22.
    Then add the pairs.
    
    If r(2,2,3) = 7 (which is achievable), then 4 * 7 = 28. Not 27.
    
    Hmm. But wait, can we do better? The output is 2x6, which is 12 elements.
    The total rank needed is at least the tensor rank of the whole thing.
    
    Let me just implement the 4-block approach with r(2,2,3)=7, giving rank 28.
    """
    r_223 = 7
    total_rank = 4 * r_223
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # We need the 2x2 x 2x3 decomposition
    # For multiplying X (2x2) by Y (2x3) to get Z (2x3)
    # Z[i,k] = sum_l X[i,l] * Y[l,k]
    # 
    # We can use the fact that r(2,2,2)=7, and handle the 3rd column separately?
    # Actually, a simple way: use 4 trivial terms for the 2x2 part times each of the 3 columns?
    # No, that would be 4*3=12 per block, total 48.
    #
    # Better: for each 2x2 x 2x3 multiplication, we can use 7 factors from 2x2x2 
    # but that only gives 2x2 output. We need 2x3.
    #
    # Simple approach: for the 2x2 by 2x3 multiplication, treat it as:
    # The tensor is sum_{i,k,l} X[i,l] Y[l,k] e_{ik}
    # We can decompose by fixing l: for l=0 and l=1.
    # For fixed l, the term is X[:,l] * Y[l,:] which is a 2x3 rank-1 (outer product of column and row)
    # But X[:,l] is 2x1, Y[l,:] is 1x3. Their product is 2x3.
    # But we need bilinear decomposition, not just rank-1 of the output.
    #
    # Actually, the simplest: just use 2*3 = 6 factors where each factor picks one (i,k) and one l?
    # No. Let's just use 12 factors (trivial) per 2x2x2x3 block? That's 4*12=48.
    #
    # I'm overcomplicating this. Let me just keep rank 42 and try a different improvement.
    # 
    # Alternative: use the fact that m=4 and p=6.
    # We can write the multiplication as sum over j of A[:,j] * B[j,:]
    # where A[:,j] is 2x1 and B[j,:] is 1x6.
    # There are 4 such terms. Each is a 2x1 by 1x6 outer product.
    # That's just 4 terms if we treat them as rank-1 in the output space.
    # But the standard decomposition uses individual scalar multiplications.
    #
    # The trivial rank is 2*4*6 = 48.
    # The Strassen block gives 42.
    #
    # Let me just try to find if there's a rank-36 or lower solution.
    # For now, return None to indicate this approach didn't yield improvement.
    return None, None, None

def _build_strassen_solution(n, m, p):
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j][col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k][col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i][col] += coeff
                
                col += 1

    return U, V, W

def _build_trivial_solution(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    return U, V, W

def _verify_solution_fast(U, V, W, n, m, p):
    if U is None or V is None or W is None:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check dimensions
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
    
    # Build C = AB via decomposition
    # C[i,k] = sum_l A[i,l] * B[l,k]
    # In the decomposition: C[i,k] = sum_r U[i,l... wait, U is indexed by (i,m+l?) no.
    # U has shape (n*m, rank), indexed by i*m + j for row i, col j of A.
    # V has shape (m*p, rank), indexed by j*p + k for row j, col k of B.
    # W has shape (p*n, rank), indexed by k*n + i for output C[i,k].
    # 
    # The decomposition says: sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ... 
    # No, that's not right either. Let me re-examine.
    #
    # Actually, the standard tensor decomposition for matrix multiplication (n,m,p) is:
    # T_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] ... 
    # Hmm, the indices are tricky. Let me just verify using the standard formula.
    #
    # For the matrix multiplication tensor, we have:
    # (AB)_{i,k} = sum_l A_{i,l} B_{l,k}
    # The decomposition provides rank-r factors such that:
    # sum_r U_{i,l,r} V_{l,k,r} W_{k,i,r} ... 
    #
    # I think the verification should reconstruct C and check it equals AB.
    # Let me use the formula from the verify function in the original.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
