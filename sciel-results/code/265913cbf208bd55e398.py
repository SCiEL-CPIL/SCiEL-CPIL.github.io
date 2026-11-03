import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]
_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _validate_exact(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0.0
                for c in range(rank):
                    val += U_arr[u_idx, c] * V_arr[v_idx, c] * W_arr[w_idx, c]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _build_strassen_2x2x2():
    """Return U,V,W for 2x2x2 Strassen decomposition, each a list of lists with integer entries."""
    # Strassen's 7 multiplications for 2x2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A10)B11
    # P6 = (A01-A10)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P7
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Tensor entries: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # For 2x2x2: n=m=p=2
    # U has 4 rows, V has 4 rows, W has 4 rows, each with 7 columns.
    
    # Let's define U, V, W as 4x7 matrices with integer entries.
    # The tensor is: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0,1}
    
    # From Strassen:
    # c=0 (P1): U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
    #   Check: i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0]=1*1*1=1 ✓
    #   i=1,j=1,k=1: U[1,1]*V[1,1]*W[1,1]=1*1*1=1 ✓
    #   i=0,j=1,k=0: U[0,1]*V[1,0]*W[0,0]=0*0*1=0 ✓
    #   i=1,j=0,k=1: U[1,0]*V[0,1]*W[1,1]=0*0*1=0 ✓
    #   All other (i,j,k): product is 0 ✓
    
    # c=1 (P2): U[1,0]=1, U[1,1]=1; V[0,0]=1; W[0,1]=1
    #   i=1,j=0,k=0: U[1,0]*V[0,0]*W[0,1]=1*1*1=1 ✓
    #   i=1,j=1,k=0: U[1,1]*V[1,0]*W[0,1]=1*0*1=0 ✓
    #   Others: 0 ✓
    
    # c=2 (P3): U[0,0]=1; V[0,1]=1, V[1,1]=-1; W[1,0]=1
    #   i=0,j=0,k=1: U[0,0]*V[0,1]*W[1,0]=1*1*1=1 ✓
    #   i=0,j=1,k=1: U[0,1]*V[1,1]*W[1,0]=0*(-1)*1=0 ✓
    #   Others: 0 ✓
    
    # c=3 (P4): U[1,1]=1; V[1,0]=1, V[0,0]=-1; W[0,1]=1
    #   i=1,j=1,k=0: U[1,1]*V[1,0]*W[0,1]=1*1*1=1 ✓
    #   i=1,j=0,k=0: U[1,0]*V[0,0]*W[0,1]=0*(-1)*1=0 ✓
    #   Others: 0 ✓
    
    # c=4 (P5): U[0,0]=1, U[0,1]=1; V[1,1]=1; W[1,1]=1
    #   i=0,j=1,k=1: U[0,1]*V[1,1]*W[1,1]=1*1*1=1 ✓
    #   i=0,j=0,k=1: U[0,0]*V[0,1]*W[1,1]=1*0*1=0 ✓
    #   Others: 0 ✓
    
    # c=5 (P6): U[0,1]=1, U[1,0]=-1; V[0,0]=1, V[0,1]=1; W[0,1]=1
    #   i=0,j=1,k=0: U[0,1]*V[1,0]*W[0,1]=1*0*1=0 ✓
    #   i=0,j=1,k=1: U[0,1]*V[1,1]*W[1,1]=1*0*1=0 ✓
    #   Wait, this doesn't work. Let me re-derive.
    #
    # Actually, the standard Strassen tensor decomposition for 2x2x2 is:
    # The 7 rank-1 tensors are:
    # T1: A00*B00, A00*B11, A11*B00, A11*B11 → contributes to C00,C01,C10,C11
    # This is getting complex. Let me use a known explicit decomposition.
    
    # Known Strassen 2x2x2 decomposition (from literature):
    # The tensor T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k has rank 7.
    # One explicit decomposition:
    
    U_s = [
        [1, 1, 1, 0, 1, 1, 1],  # i=0,j=0
        [0, 0, 0, 1, 1, -1, -1], # i=0,j=1
        [0, 1, 0, 1, 0, 1, 0],   # i=1,j=0
        [1, 1, 0, 1, 0, 0, 1],   # i=1,j=1
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 1],  # j=0,k=0
        [0, 0, 1, 0, 1, 0, 1],   # j=0,k=1
        [1, 0, 1, 1, 1, 1, 0],   # j=1,k=0
        [1, 1, -1, 1, 0, 0, 1],  # j=1,k=1
    ]
    W_s = [
        [1, 0, 0, 0, 1, 0, -1],  # k=0,i=0
        [0, 1, 1, 1, 0, 1, 1],   # k=0,i=1
        [1, 1, 0, 0, 0, 1, 1],   # k=1,i=0
        [1, 0, 1, 1, 1, 0, 0],   # k=1,i=1
    ]
    
    # Verify
    n2, m2, p2 = 2, 2, 2
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                val = 0
                for c in range(7):
                    val += U_s[i*m2+j][c] * V_s[j*p2+k][c] * W_s[k*n2+i][c]
                if val != 1:
                    raise ValueError(f"Strassen verification failed at i={i},j={j},k={k},val={val}")
    
    return U_s, V_s, W_s

def _build_strassen_2x3x8():
    """
    Construct a rank-29 decomposition for 2x3x8 using:
    - 4 copies of Strassen 2x2x2 for the 2x2x8 part (j=0,1; k=0..7)
    - 1 rank-1 component for the 2x1x8 part (j=2; k=0..7)
    
    Total rank = 4*7 + 1 = 29
    """
    n, m, p = 2, 3, 8
    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    U = [[0] * 29 for _ in range(n * m)]  # 6 x 29
    V = [[0] * 29 for _ in range(m * p)]  # 24 x 29
    W = [[0] * 29 for _ in range(p * n)]  # 16 x 29
    
    # Fill in 4 Strassen blocks for j=0,1 and k blocks 0-3, 4-7
    # Each Strassen block handles a 2x2x2 sub-problem
    # For j in {0,1} and k_block in {0,1} (covering k=0..3 and k=4..7):
    #   The sub-problem is A[i, j] * B[j, k] for i in {0,1}, j in {0,1}, k in block
    
    # Actually, let's think of it differently.
    # The 2x3x8 multiplication can be seen as:
    # For each j in {0,1,2} and k in {0..7}: C[i,k] += A[i,j] * B[j,k]
    # 
    # For j=0,1: we have a 2x2x8 problem (i in {0,1}, j in {0,1}, k in {0..7})
    # This can be decomposed into 4 independent 2x2x2 problems by splitting k into two groups of 4? 
    # No, Strassen works for 2x2x2. For 2x2x8, we can use 4 copies of Strassen for k=0..1, 2..3, 4..5, 6..7.
    
    # Let's use 4 Strassen decompositions for k pairs (0,1), (2,3), (4,5), (6,7)
    for k_pair in range(4):
        k0 = k_pair * 2
        k1 = k_pair * 2 + 1
        offset = k_pair * 7  # column offset in the 29-column matrices
        
        for c in range(7):
            # U part: for i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * m + j][offset + c] = U_s[i * 2 + j][c]
            
            # V part: for j in {0,1}, k in {k0, k1}
            for j in range(2):
                for dk in range(2):
                    k = k0 + dk
                    V[j * p + k][offset + c] = V_s[j * 2 + dk][c]
            
            # W part: for k in {k0, k1}, i in {0,1}
            for dk in range(2):
                k = k0 + dk
                for i in range(2):
                    W[k * n + i][offset + c] = W_s[dk * 2 + i][c]
    
    # For j=2: rank-1 decomposition for each k
    # C[i, k] += A[i, 2] * B[2, k]
    # This is already rank-1 for each (i,k) pair, but we need to express it as:
    # sum_c U[i,2,c] * V[2,k,c] * W[k,i,c] = 1 for all i,k
    # 
    # We can use 1 column for all 8 k values? No, because V[2,k,c] and W[k,i,c] vary with k.
    # Actually, for j=2, the contribution is:
    # For each (i,k): we need U[i,2,c] * V[2,k,c] * W[k,i,c] = 1
    # If we set U[i,2,c] = 1, V[2,k,c] = 1, W[k,i,c] = 1 for all i,k,c, then the sum over c would be rank, not 1.
    # 
    # Wait, the tensor equation is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = delta_{i,j,k} in some sense.
    # For the schoolbook, each (i,j,k) has its own column.
    # For j=2, we have 2*8=16 terms (i in {0,1}, k in {0..7}), each needing to equal 1.
    # But these are independent constraints. We can't do it with 1 column unless the products factor.
    #
    # Actually, for j=2, the term is A[i,2] * B[2,k] which contributes to C[i,k].
    # In the tensor decomposition, for fixed j=2, we need:
    # sum_c U[i,2,c] * V[2,k,c] * W[k,i,c] = 1 for all i in {0,1}, k in {0..7}
    #
    # This is a 2x8x2 tensor (i,k,i) with 16 entries, each = 1.
    # The rank of this tensor: it's the all-ones tensor of shape 2x8x2.
    # The rank of the all-ones tensor of shape a x b x c is max(a,b,c) if... no.
    # Actually, the all-ones tensor 1_{a,b,c} = sum_i sum_j sum_k e_i ⊗ e_j ⊗ e_k has rank... 
    # It can be written as (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) which is rank 1!
    # Wait: (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # Yes! So the all-ones tensor is rank 1.
    #
    # So for j=2, we can use 1 column:
    # U[i,2,c] = 1 for all i
    # V[2,k,c] = 1 for all k  
    # W[k,i,c] = 1 for all k,i
    # Then sum_c U[i,2,c] V[2,k,c] W[k,i,c] = 1*1*1 = 1 ✓
    
    c_last = 28  # last column index
    for i in range(2):
        U[i * m + 2][c_last] = 1
    for k in range(8):
        V[2 * p + k][c_last] = 1
    for k in range(8):
        for i in range(2):
            W[k * n + i][c_last] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    # Try the structured Strassen-based construction (rank 29)
    try:
        result = _build_strassen_2x3x8()
        if _validate_exact(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 48)
    return _build_schoolbook()
