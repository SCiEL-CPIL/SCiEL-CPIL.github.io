import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 Strassen decomposition (Rank 49)
    # Strassen 2x2 factors
    # U2: 2x7, V2: 2x7, W2: 2x7
    # Indices 0,1 for 2x2 blocks
    # M1..M7
    U2 = np.zeros((2, 7), dtype=int)
    V2 = np.zeros((2, 7), dtype=int)
    W2 = np.zeros((2, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    # U: 00=1, 11=1. V: 00=1, 11=1. W: 00=1? No, W extracts C.
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # W2[r, c] is coeff of M_r in C_c? No.
    # T = sum u v w.
    # W is the factor for C.
    # W[i, k, r] -> W_flat[i*2+k, r].
    # W_flat[0, r] = coeff of M_r in C00
    # W_flat[1, r] = coeff of M_r in C01
    # W_flat[2, r] = coeff of M_r in C10
    # W_flat[3, r] = coeff of M_r in C11
    
    # M1 (r=0)
    U2[0,0] = 1; U2[1,1] = 1
    V2[0,0] = 1; V2[1,1] = 1
    W2[0,0] = 1  # C00
    W2[1,0] = 0  # C01
    W2[2,0] = 0  # C10
    W2[3,0] = 1  # C11
    
    # M2 (r=1)
    U2[1,1] = 1  # (A10+A11) -> A10, A11. Indices: A is 2x2.
    # Wait, Strassen: M2 = (A21+A22)B11. In 0-indexed: (A[1,0]+A[1,1])B[0,0].
    U2[1,0] = 1; U2[1,1] = 1
    V2[0,0] = 1
    W2[0,1] = 0  # C00
    W2[1,1] = 0  # C01
    W2[2,1] = 1  # C10
    W2[3,1] = -1 # C11
    
    # M3 (r=2)
    # M3 = A11(B12-B22). 0-indexed: A[0,0](B[0,1]-B[1,1]).
    U2[0,0] = 1
    V2[0,1] = 1; V2[1,1] = -1
    W2[0,2] = 0
    W2[1,2] = 1  # C01
    W2[2,2] = 0
    W2[3,2] = 1  # C11
    
    # M4 (r=3)
    # M4 = A22(B21-B11). 0-indexed: A[1,1](B[1,0]-B[0,0]).
    U2[1,1] = 1
    V2[1,0] = 1; V2[0,0] = -1
    W2[0,3] = -1 # C00
    W2[1,3] = 0
    W2[2,3] = 1  # C10
    W2[3,3] = 0
    
    # M5 (r=4)
    # M5 = (A11+A12)B22. 0-indexed: (A[0,0]+A[0,1])B[1,1].
    U2[0,0] = 1; U2[0,1] = 1
    V2[1,1] = 1
    W2[0,4] = -1 # C00
    W2[1,4] = 1  # C01
    W2[2,4] = 0
    W2[3,4] = 0
    
    # M6 (r=5)
    # M6 = (A21-A11)(B11+B12). 0-indexed: (A[1,0]-A[0,0])(B[0,0]+B[0,1]).
    U2[1,0] = 1; U2[0,0] = -1
    V2[0,0] = 1; V2[0,1] = 1
    W2[0,5] = 0
    W2[1,5] = 0
    W2[2,5] = 0
    W2[3,5] = -1 # C11
    
    # M7 (r=6)
    # M7 = (A12-A22)(B21+B22). 0-indexed: (A[0,1]-A[1,1])(B[1,0]+B[1,1]).
    U2[0,1] = 1; U2[1,1] = -1
    V2[1,0] = 1; V2[1,1] = 1
    W2[0,6] = 1  # C00
    W2[1,6] = 0
    W2[2,6] = 0
    W2[3,6] = 0
    
    # Now form 4x4x4 factors via Kronecker product
    # U4 = U2 ⊗ U2, V4 = V2 ⊗ V2, W4 = W2 ⊗ W2
    U4 = np.kron(U2, U2) # 4x49
    V4 = np.kron(V2, V2) # 4x49
    W4 = np.kron(W2, W2) # 4x49
    
    # Part 2: 4x4x1 Schoolbook decomposition (Rank 16)
    # U5: 16x16, V5: 20x16, W5: 20x16
    U5 = np.zeros((16, 16), dtype=int)
    V5 = np.zeros((20, 16), dtype=int)
    W5 = np.zeros((20, 16), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U5[i*4 + j, col] = 1
            V5[j*5 + 4, col] = 1  # Col 4 is the 5th column (index 4)
            W5[4*4 + i, col] = 1  # Col 4 is the 5th column of C
            col += 1
            
    # Combine
    # U: 16 x 65
    U = np.hstack([U4, U5])
    # V: 20 x 65
    # V4 is 4x49. We need to embed it into 20x49.
    # V4 rows correspond to B's rows 0-3, cols 0-3?
    # In Strassen 4x4, V4 is for B (4x4).
    # The indices of V4 are $j \in 0..3$.
    # In the full V (20 rows), these correspond to indices $j*5 + k$ where $k \in 0..3$.
    # Wait, V4 is a matrix of shape (4, 49). The row index $j$ in V4 corresponds to the row $j$ of the 4x4 B matrix.
    # In the full B matrix (4x5), row $j$ has columns 0..4.
    # The 4x4 Strassen decomposition only uses columns 0..3.
    # So V4[j, r] should go to V_full[j*5 + 0..3, r]?
    # No, V is a factor matrix. $V_{index, r}$.
    # The index in V is flat index of B element.
    # $B$ is $4 \times 5$. Flat index $b = j*5 + k$.
    # The 4x4 part uses $B_{j,k}$ for $k \in 0..3$.
    # So for each row $j$ of V4, we need to place the value into $V_{j*5+k, r}$?
    # No, that's not how it works.
    # The tensor decomposition is $T = \sum_r u_r \otimes v_r \otimes w_r$.
    # $v_r$ is a vector of size $m \times p = 20$.
    # In the 4x4 case, $v_r$ was a vector of size $4 \times 4 = 16$.
    # The entries of $v_r$ in the 4x4 case correspond to $B_{j,k}$ for $j,k \in 0..3$.
    # So $v_{r, j, k}$.
    # In the full problem, $v_{r, j, k}$ for $k \in 0..3$ is at flat index $j*5 + k$.
    # For $k=4$, it is 0.
    
    V_full = np.zeros((20, 65), dtype=int)
    for j in range(4):
        for k in range(4):
            V_full[j*5 + k, 0:49] = V4[j, 0:49] # Wait, V4 is 4x49. Row j is the factor for row j of B?
            # No, V4 is the factor matrix for B.
            # V4[j, r] is the coefficient of the r-th term for B element (j, ?).
            # In the 4x4 Strassen, V4 has 4 rows?
            # Strassen 2x2: V2 has 2 rows.
            # Strassen 4x4: V4 has 4 rows?
            # Yes, V4 = V2 ⊗ V2.
            # V2 is 2x7. V4 is 4x49.
            # The rows of V4 correspond to the 4 elements of the 4x1? No, B is 4x4.
            # B is 4x4. There are 16 elements.
            # But V4 has only 4 rows?
            # Ah, the factor matrix V has dimensions $m \times \text{rank}$.
            # For $4 \times 4 \times 4$, $m=4$. So V is $4 \times 49$.
            # This implies that the factor $V$ does not depend on the column index of B?
            # No, that's impossible. $V_{j,r}$ must depend on $j$ (row of B) and the term $r$.
            # But what about the column $k$ of B?
            # In the tensor $T_{ijk}$, $j$ is the index for B's row?
            # Standard: $C_{ik} = \sum_j A_{ij} B_{jk}$.
            # Tensor indices: $i$ (A row), $j$ (A col/B row), $k$ (B col).
            # Factorization: $T_{ijk} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
            # Wait, the task defines:
            # U: (n*m, rank) -> $n \times m$ A-matrix. Index $i*m+j$.
            # V: (m*p, rank) -> $m \times p$ B-matrix. Index $j*p+k$.
            # W: (p*n, rank) -> $p \times n$ C-matrix. Index $k*n+i$.
            
            # So $T_{ijk} = \sum_r U_{im+j, r} V_{jp+k, r} W_{kn+i, r}$.
            # For $4 \times 4 \times 4$ Strassen:
            # U4 is $4 \times 49$? No, U is $n \times m \times \text{rank} = 4 \times 4 \times 49 = 16 \times 49$.
            # My previous $U4 = \text{kron}(U2, U2)$ resulted in $4 \times 49$.
            # $U2$ is $2 \times 7$. $\text{kron}(U2, U2)$ is $4 \times 49$.
            # But U should be $16 \times 49$.
            # Why?
            # Because $U$ corresponds to A ($4 \times 4$).
            # $U2$ corresponds to A ($2 \times 2$).
            # $U2$ has 2 rows? No, A is $2 \times 2$, so 4 elements.
            # $U2$ should be $4 \times 7$.
            # My $U2$ was $2 \times 7$. This is wrong.
            
    # Let's re-verify dimensions.
    # U: (n*m, rank). For 2x2, n=2, m=2. U is 4x7.
    # V: (m*p, rank). For 2x2, m=2, p=2. V is 4x7.
    # W: (p*n, rank). For 2x2, p=2, n=2. W is 4x7.
    
    # So U2, V2, W2 should be 4x7.
    
    # Let's redefine U2, V2, W2 as 4x7.
    # U2 rows correspond to A elements (00, 01, 10, 11).
    # V2 rows correspond to B elements (00, 01, 10, 11).
    # W2 rows correspond to C elements (00, 01, 10, 11).
    
    # M1: (A00+A11)(B00+B11)
    # A00 (idx 0): 1. A11 (idx 3): 1.
    # B00 (idx 0): 1. B11 (idx 3): 1.
    # C00 (idx 0): 1. C11 (idx 3): 1.
    # C01 (idx 1): 0. C10 (idx 2): 0.
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # M2: (A10+A11)B00
    # A10 (idx 2): 1. A11 (idx 3): 1.
    # B00 (idx 0): 1.
    # C10 (idx 2): 1. C11 (idx 3): -1.
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1; W2[3,1] = -1
    
    # M3: A00(B01-B11)
    # A00 (idx 0): 1.
    # B01 (idx 1): 1. B11 (idx 3): -1.
    # C01 (idx 1): 1. C11 (idx 3): 1.
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[1,2] = 1; W2[3,2] = 1
    
    # M4: A11(B10-B00)
    # A11 (idx 3): 1.
    # B10 (idx 2): 1. B00 (idx 0): -1.
    # C00 (idx 0): -1. C10 (idx 2): 1.
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = -1; W2[2,3] = 1
    
    # M5: (A00+A01)B11
    # A00 (idx 0): 1. A01 (idx 1): 1.
    # B11 (idx 3): 1.
    # C00 (idx 0): -1. C01 (idx 1): 1.
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[1,4] = 1
    
    # M6: (A10-A00)(B00+B01)
    # A10 (idx 2): 1. A00 (idx 0): -1.
    # B00 (idx 0): 1. B01 (idx 1): 1.
    # C11 (idx 3): -1.
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = -1
    
    # M7: (A01-A11)(B10+B11)
    # A01 (idx 1): 1. A11 (idx 3): -1.
    # B10 (idx 2): 1. B11 (idx 3): 1.
    # C00 (idx 0): 1.
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1
    
    # Now U4, V4, W4 are 16x49.
    U4 = np.kron(U2, U2) # 16x49
    V4 = np.kron(V2, V2) # 16x49
    W4 = np.kron(W2, W2) # 16x49
    
    # Now assemble full U, V, W.
    U_full = np.hstack([U4, U5]) # 16x65
    V_full = np.zeros((20, 65), dtype=int)
    W_full = np.zeros((20, 65), dtype=int)
    
    # Copy V4 to V_full.
    # V4 is 16x49. Rows correspond to B elements (4x4).
    # B is 4x5. Elements are (j, k) for j in 0..3, k in 0..4.
    # Flat index $j*5 + k$.
    # V4 row index $r$ corresponds to $j*4+k$ (for $k \in 0..3$).
    # We need to map $r = j*4+k$ to $j*5+k$.
    for j in range(4):
        for k in range(4):
            src_idx = j*4 + k
            dst_idx = j*5 + k
            V_full[dst_idx, 0:49] = V4[src_idx, 0:49]
            
    # Copy W4 to W_full.
    # W4 is 16x49. Rows correspond to C elements (4x4).
    # C is 4x5. Elements are (k, i) for k in 0..4, i in 0..3.
    # Flat index $k*4 + i$.
    # W4 row index $r$ corresponds to $k*4+i$ (for $k \in 0..3$).
    # We need to map $r = k*4+i$ to $k*5+i$? No.
    # W index is $k*n+i = k*4+i$.
    # Wait, W shape is (p*n, rank) = (5*4, rank) = 20.
    # Index $k*4+i$.
    # For the 4x4 part, $k \in 0..3$.
    # So W4 row $k*4+i$ goes to W_full row $k*4+i$.
    # This is a direct copy for the first 16 rows.
    W_full[0:16, 0:49] = W4
    
    # Copy U5, V5, W5 to the last 16 columns.
    # U5 is 16x16.
    U_full[:, 49:65] = U5
    # V5 is 20x16.
    V_full[:, 49:65] = V5
    # W5 is 20x16.
    W_full[:, 49:65] = W5
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
