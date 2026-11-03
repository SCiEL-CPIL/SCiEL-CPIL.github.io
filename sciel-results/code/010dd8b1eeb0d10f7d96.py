import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 multiplication tensor decomposition
    # For matrices A (2x2), B (2x2), C = A*B
    # C[0,0] = M1 - M2 + M4 + M5
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M4 + M6
    # where:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # The tensor T[i,j,k] = 1 if C[i,k] involves A[i,j]*B[j,k]
    # In Strassen, we decompose this into 7 rank-1 terms
    
    # For each term t, we have U[i,j,t], V[j,k,t], W[i,k,t] such that
    # sum_t U[i,j,t]*V[j,k,t]*W[i,k,t] = delta(i,k) for the correct (i,k)
    
    # Let's define the Strassen decomposition explicitly
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # This contributes to C[0,0] with +1, C[1,1] with +1
    # U for term 1: U[0,0]=1, U[1,1]=1, others 0
    # V for term 1: V[0,0]=1, V[1,1]=1, others 0
    # W for term 1: W[0,0]=1, W[1,1]=1, others 0
    
    # But wait, the tensor is T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # For 2x2x2: n=2, m=2, p=2
    # a = i*2+j, b = j*2+k, c = k*2+i
    
    # Let me build the full decomposition for 2x4x5 using block structure
    
    # Split the 4 columns of A into two blocks of 2, and 5 rows of B into:
    # - Two blocks of 2 rows (for the first 4 columns of C)
    # - One block of 1 row (for the 5th column of C)
    
    # For the 2x2x2 blocks, use Strassen (rank 7)
    # For the 2x4x1 block, use schoolbook (rank 8)
    
    # Total rank: 4 blocks * 7 + 8 = 28 + 8 = 36
    
    rank = 36
    U = np.zeros((n*m, rank))  # 8 x 36
    V = np.zeros((m*p, rank))  # 20 x 36
    W = np.zeros((p*n, rank))  # 10 x 36
    
    # Strassen terms for 2x2x2
    # Each term has U (2x2), V (2x2), W (2x2)
    # Term definitions (U[i,j], V[j,k], W[i,k]):
    
    strassen_terms = []
    
    # Term 1: M1 = (A00+A11)*(B00+B11)
    # U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]]
    strassen_terms.append((
        [[1,0],[0,1]],  # U
        [[1,0],[0,1]],  # V
        [[1,0],[0,1]]   # W
    ))
    
    # Term 2: M2 = (A10+A11)*B00
    # U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[0,0],[1,0]]
    strassen_terms.append((
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[1,0]]
    ))
    
    # Term 3: M3 = A00*(B01-B11)
    # U: [[1,0],[0,0]], V: [[0,1],[0,-1]], W: [[0,1],[0,0]]
    strassen_terms.append((
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[0,1],[0,0]]
    ))
    
    # Term 4: M4 = A11*(B10-B00)
    # U: [[0,0],[0,1]], V: [[0,-1],[1,0]], W: [[0,0],[1,0]]
    strassen_terms.append((
        [[0,0],[0,1]],
        [[0,-1],[1,0]],
        [[0,0],[1,0]]
    ))
    
    # Term 5: M5 = (A00+A01)*B11
    # U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[0,1],[0,0]]
    strassen_terms.append((
        [[1,1],[0,0]],
        [[0,0],[0,1]],
        [[0,1],[0,0]]
    ))
    
    # Term 6: M6 = (A10-A00)*(B00+B01)
    # U: [[-1,0],[1,0]], V: [[1,1],[0,0]], W: [[0,0],[1,0]]
    strassen_terms.append((
        [[-1,0],[1,0]],
        [[1,1],[0,0]],
        [[0,0],[1,0]]
    ))
    
    # Term 7: M7 = (A01-A11)*(B10+B11)
    # U: [[0,1],[0,-1]], V: [[0,0],[1,1]], W: [[0,0],[0,1]]
    strassen_terms.append((
        [[0,1],[0,-1]],
        [[0,0],[1,1]],
        [[0,0],[0,1]]
    ))
    
    # Now fill in the blocks
    # Block (j0, k0) where j0 in {0,2}, k0 in {0,2}
    # j = j0 + j', k = k0 + k', j', k' in {0,1}
    
    rank_offset = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for t in range(7):
                U_t, V_t, W_t = strassen_terms[t]
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i*m + j, rank_offset] = U_t[i][j_local]
                for j_local in range(2):
                    for k_local in range(2):
                        j = j0 + j_local
                        k = k0 + k_local
                        V[j*p + k, rank_offset] = V_t[j_local][k_local]
                for k_local in range(2):
                    for i in range(2):
                        k = k0 + k_local
                        W[k*n + i, rank_offset] = W_t[k_local][i]
                rank_offset += 1
    
    # Now handle the 5th column of B (k=4)
    # This is 2x4x1 schoolbook: 8 terms
    for i in range(2):
        for j in range(4):
            U[i*m + j, rank_offset] = 1
            V[j*p + 4, rank_offset] = 1
            W[4*n + i, rank_offset] = 1
            rank_offset += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
