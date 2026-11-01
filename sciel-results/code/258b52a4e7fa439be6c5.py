"""Matrix multiplication 2x2 x 2x16 via Strassen decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Return the Strassen decomposition of 2x2 matrix multiplication.
    Rank-7 with coefficients in {-1, 0, 1}.
    
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify_strassen_core(U_core, V_core, W_core):
    """Verify the 2x2 Strassen core decomposition is correct."""
    # The tensor T[i,j,k] = delta_{i,k} (for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k])
    # For 2x2: T is a 2x2x2 tensor where T[i,j,k] = 1 if i==k else 0... 
    # Actually for matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = delta_{i,k}... no.
    # T_{ijk} where C_{ik} = sum_j A_{ij} B_{jk}
    # So T[i,j,k] = 1 if we're computing C[i,k] using A[i,j]*B[j,k]
    # The rank-1 decomposition: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # means T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    
    # Build the true tensor
    T_true = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_true[i, j, k] = 1.0  # C[i,k] gets contribution from A[i,j]*B[j,k]
    
    # Wait, that's not right. Let me think again.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for 2x2x2 matmul)
    # No wait: T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # C[i,k] = sum_j A[i,j] * B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    # That means the tensor is all ones!
    
    # Actually no. The tensor for matrix multiplication [n,m,p] is:
    # T[i,j,k] = 1 if the output index (i,k) depends on input (i,j) and (j,k)
    # For C[i,k] = sum_j A[i,j] B[j,k], every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # has T[i,j,k] = 1.
    
    # So for 2x2x2, T is the all-ones 2x2x2 tensor.
    
    # Let's verify: sum_r U[i,r]*V[j,r]*W[k,r] should equal 1 for all i,j,k in {0,1}
    T_decomp = np.einsum('ir,jr,kr->ijk', U_core, V_core, W_core)
    T_true = np.ones((2, 2, 2))
    
    if not np.allclose(T_decomp, T_true, atol=1e-10):
        return False, T_decomp, T_true
    return True, T_decomp, T_true

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x16 matrix.
    
    Uses Strassen's rank-7 decomposition for the 2x2 core, applied
    independently to each of the 8 column-pairs of the 2x16 matrix.
    Total rank = 7 * 8 = 56.
    
    Indexing conventions:
    - U row index: i*m + j where (i,j) indexes A (i=0,1; j=0,1)
    - V row index: j*p + k where (j,k) indexes B (j=0,1; k=0..15)
    - W row index: k*n + i where (k,i) indexes C output (k=0..15; i=0,1)
      Per the trace convention W[k*n+i, r]
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    
    U_core, V_core, W_core = _strassen_2x2()
    
    # Verify the core decomposition
    valid, _, _ = _verify_strassen_core(U_core, V_core, W_core)
    if not valid:
        raise ValueError("Strassen core verification failed")
    
    num_pairs = p // 2  # 8 pairs
    rank = 7 * num_pairs  # 56
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for t in range(num_pairs):
        col_offset = t * 7
        
        for r in range(7):
            # U: same for all pairs
            U[:, col_offset + r] = U_core[:, r]
            
            # V: row index j*p + k. Pair t covers k=2t, 2t+1
            # V_core rows: 0=(j=0,local_k=0), 1=(j=0,local_k=1), 2=(j=1,local_k=0), 3=(j=1,local_k=1)
            V[0 * p + 2*t, col_offset + r] = V_core[0, r]
            V[0 * p + 2*t + 1, col_offset + r] = V_core[1, r]
            V[1 * p + 2*t, col_offset + r] = V_core[2, r]
            V[1 * p + 2*t + 1, col_offset + r] = V_core[3, r]
            
            # W: row index k*n + i. Pair t covers output cols k=2t, 2t+1
            # W_core rows: 0=(i=0,local_k=0), 1=(i=0,local_k=1), 2=(i=1,local_k=0), 3=(i=1,local_k=1)
            W[(2*t) * n + 0, col_offset + r] = W_core[0, r]
            W[(2*t + 1) * n + 0, col_offset + r] = W_core[1, r]
            W[(2*t) * n + 1, col_offset + r] = W_core[2, r]
            W[(2*t + 1) * n + 1, col_offset + r] = W_core[3, r]
    
    # Full verification: check the tensor decomposition
    # T[i,j,k] should equal 1 for all i in [0,2), j in [0,2), k in [0,16)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Spot check a few entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {val}, expected 1")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
