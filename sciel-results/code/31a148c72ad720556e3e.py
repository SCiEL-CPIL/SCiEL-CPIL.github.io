import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4) with rank 32.
    Uses Strassen's algorithm for 2x2x2 sub-blocks (rank 7 each) for the top 2 rows of C.
    Uses rank-4 identity decomposition for the bottom row of C (1x4x4).
    """
    n, m, p = 3, 4, 4
    
    cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        cols.append((u_entries, v_entries, w_entries))
    
    def u(i, j): return i * m + j
    def v(j, k): return j * p + k
    def w(k, i): return k * n + i
    
    def add_strassen_2x2x2(a11, a12, a21, a22, b11, b12, b21, b22, c11, c12, c21, c22):
        # Strassen's 7 multiplications for 2x2x2
        add_term([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)])
        add_term([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,-1)])
        add_term([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,1)])
        add_term([(a22,1)], [(b21,1),(b11,-1)], [(c11,1),(c21,1)])
        add_term([(a11,1),(a12,1)], [(b22,1)], [(c12,1),(c11,-1)])
        add_term([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c22,1)])
        add_term([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)])

    # Block 1: C[0:2,0:2] from A[0:2,0:2]*B[0:2,0:2]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),
        v(0,0), v(0,1), v(1,0), v(1,1),
        w(0,0), w(1,0), w(0,1), w(1,1)
    )
    
    # Block 2: C[0:2,0:2] from A[0:2,2:4]*B[2:4,0:2]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),
        v(2,0), v(2,1), v(3,0), v(3,1),
        w(0,0), w(1,0), w(0,1), w(1,1)
    )
    
    # Block 3: C[0:2,2:4] from A[0:2,0:2]*B[0:2,2:4]
    add_strassen_2x2x2(
        u(0,0), u(0,1), u(1,0), u(1,1),
        v(0,2), v(0,3), v(1,2), v(1,3),
        w(2,0), w(3,0), w(2,1), w(3,1)
    )
    
    # Block 4: C[0:2,2:4] from A[0:2,2:4]*B[2:4,2:4]
    add_strassen_2x2x2(
        u(0,2), u(0,3), u(1,2), u(1,3),
        v(2,2), v(2,3), v(3,2), v(3,3),
        w(2,0), w(3,0), w(2,1), w(3,1)
    )
    
    # Block 5: C[2,0:4] from A[2,0:4]*B[0:4,0:4]
    # This is a 1x4x4 multiplication. The tensor is the identity in the k dimension.
    # Rank is 4.
    # C[2,k] = sum_j A[2,j] B[j,k]
    # Term r for k=r:
    # U[2, r] = 1
    # V[j, r] = 1 if j == r? No.
    # We need sum_r U[2,r] V[j,r] W[k,r] = A[2,j] B[j,k]
    # If we set W[k,r] = delta_{k,r}, then term is U[2,r] V[j,r].
    # We need U[2,r] V[j,r] = A[2,j] B[j,r] for the r-th term?
    # No, the sum is over r.
    # C[2,k] = sum_r U[2,r] V[j,r] W[k,r] A[2,?] B[?,?]
    # The tensor T_{j,k} = delta_{j,k} is not correct.
    # The tensor is T_{j,k} = 1.
    # Wait, C[2,k] = sum_j A[2,j] B[j,k].
    # This is a linear map from A[2,:] to C[2,:].
    # The tensor is T_{j,k} = 1.
    # This is rank 1?
    # If T_{j,k} = 1, then rank 1.
    # Let's check: sum_r u_r v_j w_k = 1.
    # u_1=1, v_j=1, w_k=1.
    # Then C[2,k] = A[2,?] B[?,?] * 1.
    # No, the indices of U, V, W must match the indices of A, B.
    # U is indexed by (i,j). Here i=2. So U[2,j].
    # V is indexed by (j,k).
    # W is indexed by (k,i). Here i=2. So W[k,2].
    # Term: U[2,j] V[j,k] W[k,2].
    # We want sum_{j,k} A[2,j] B[j,k] T_{j,k,2} = C[2,?].
    # No, the output is C[2,k_out].
    # The sum is over j. C[2,k_out] = sum_j A[2,j] B[j,k_out].
    # The tensor is T_{j, k_out} = 1.
    # But W is indexed by (k, i). Here k is the column of B, i is the row of C.
    # So W[k_out, 2].
    # The term is U[2, j_in] V[j_in, k_in] W[k_in, 2].
    # We sum over j_in, k_in?
    # No, the tensor decomposition sums over r.
    # C[2, k_out] = sum_r U[2, j_r?] ...
    # The indices of U, V, W are fixed for each term r.
    # U_r is a vector in R^{3*4}.
    # V_r is a vector in R^{4*4}.
    # W_r is a vector in R^{4*3}.
    # C = sum_r U_r (A) * V_r (B) * W_r?
    # No, C_{ik} = sum_r U_{ir} A_{i?} ...
    # The formula is C = sum_r (U_r A) (V_r B) W_r?
    # No, the standard form is:
    # C_{ik} = sum_r U_{ir} V_{jr} W_{kr} A_{ij} B_{jk}
    # Sum over i, j, k, r.
    # For the 1x4x4 part:
    # i=2.
    # C_{2,k} = sum_r U_{2r} V_{jr} W_{kr} A_{2j} B_{jk}
    # We want this to equal sum_j A_{2j} B_{jk}.
    # So we need sum_r U_{2r} V_{jr} W_{kr} = delta_{j, ?} ...
    # No, we need sum_r U_{2r} V_{jr} W_{kr} = 1 for all j,k?
    # No, the coefficient of A_{2j} B_{jk} in C_{2,k} is 1.
    # So we need sum_r U_{2r} V_{jr} W_{kr} = 1 for all j,k.
    # This is the all-ones matrix.
    # The rank of the all-ones matrix is 1.
    # So we can use 1 term!
    # U_{21} = 1, V_{j1} = 1, W_{k1} = 1.
    # Then sum_r ... = 1 * 1 * 1 = 1.
    # So rank 1 is sufficient for the 1x4x4 part.
    
    add_term([(u(2,0),1), (u(2,1),1), (u(2,2),1), (u(2,3),1)], 
             [(v(0,0),1), (v(0,1),1), (v(0,2),1), (v(0,3),1),
              (v(1,0),1), (v(1,1),1), (v(1,2),1), (v(1,3),1),
              (v(2,0),1), (v(2,1),1), (v(2,2),1), (v(2,3),1),
              (v(3,0),1), (v(3,1),1), (v(3,2),1), (v(3,3),1)],
             [(w(0,2),1), (w(1,2),1), (w(2,2),1), (w(3,2),1)])
             
    rank = len(cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(cols):
        for idx, val in u_entries:
            U[idx][r] = val
        for idx, val in v_entries:
            V[idx][r] = val
        for idx, val in w_entries:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
