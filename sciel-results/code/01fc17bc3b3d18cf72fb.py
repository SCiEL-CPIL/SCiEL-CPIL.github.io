import numpy as np
import time

def check_decomposition_fast(U, V, W, n, m, p):
    """Verify the tensor decomposition identity using NumPy broadcasting."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Tensor T[a, c, k, i] should be 1 if a==i, c==k, b==d? 
    # Indices:
    # U index: a*m + b
    # V index: c*p + d
    # W index: k*n + i
    # Target: C[k, i] = sum_{b, c} A[a, b] B[c, d] ... wait.
    # Standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
    # Here A is 2x3 (n=2, m=3), B is 3x6 (m=3, p=6). C is 2x6 (n=2, p=6).
    # C[i, k] = sum_{j=0}^{2} A[i, j] B[j, k].
    # Tensor T_{i, j, k} = A_{i, j} B_{j, k}.
    # The decomposition is: sum_r U[i, r] V[j, r] W[k, r] = delta_{i, ?} ...
    # Actually, the tensor is T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So T_{i, j, k} = 1 for all i, j, k? No.
    # The tensor $T$ is defined by $T_{i, j, k} = 1$ if $j$ is the summation index connecting $i$ and $k$.
    # In the standard basis, $T_{i, j, k} = \delta_{i, i} \delta_{j, j} \delta_{k, k}$? No.
    # $T_{i, j, k} = 1$ if the product $A_{i, j} B_{j, k}$ appears in $C_{i, k}$.
    # It always does. So $T_{i, j, k} = 1$ for all valid indices?
    # No, the tensor is $T = \sum_{i, j, k} e_i \otimes e_j \otimes e_k$?
    # If $U, V, W$ decompose it, then $\sum_r U_{i, r} V_{j, r} W_{k, r} = T_{i, j, k}$.
    # For matrix multiplication, $T_{i, j, k} = 1$ if $j$ is the common index?
    # Let's check the convention in the prompt: "output factor follows the official trace convention W[k*n+i,r]".
    # And the check in the original code:
    # target = 1.0 if (a == i and c == k and b == d) else 0.0
    # s = np.sum(U_arr[a * m + b] * V_arr[c * p + d] * W_arr[k * n + i])
    # Here a, b are A indices (row, col). c, d are B indices (row, col).
    # k, i are C indices (row, col).
    # Condition: a == i (A row == C row), c == k (B row == C row? No, B col is d, C col is i? No).
    # Let's trace indices for $C = A B$.
    # $C_{row, col} = \sum_{mid} A_{row, mid} B_{mid, col}$.
    # Prompt: A is 2x3 (n=2, m=3). B is 3x6 (m=3, p=6). C is 2x6 (n=2, p=6).
    # A indices: $a \in \{0,1\}, b \in \{0,1,2\}$. Index $a*m+b$.
    # B indices: $c \in \{0,1,2\}, d \in \{0..5\}$. Index $c*p+d$.
    # C indices: $k \in \{0..5\}, i \in \{0,1\}$. Index $k*n+i$.
    # Equation: $C_{i, k} = \sum_{b} A_{i, b} B_{b, k}$.
    # Note: $i$ is row of A and C. $k$ is col of B and C. $b$ is col of A and row of B.
    # So the term $A_{i, b} B_{b, k}$ contributes to $C_{i, k}$.
    # The tensor component $T_{a, b, c, d, k, i}$?
    # The decomposition is $\sum_r U_{a, b, r} V_{c, d, r} W_{k, i, r} = \delta_{a, i} \delta_{b, c} \delta_{d, k}$.
    # Let's verify this delta condition.
    # If $a=i, b=c, d=k$, then $A_{i, b} B_{b, k}$ contributes to $C_{i, k}$.
    # The sum over $r$ should equal 1 in this case, and 0 otherwise.
    # So $T_{a, b, c, d, k, i} = \delta_{a, i} \delta_{b, c} \delta_{d, k}$.
    
    # We can construct the tensor T and compare with U @ V @ W?
    # T has shape (n, m, m, p, p, n).
    # U has shape (n*m, R). Reshape to (n, m, R).
    # V has shape (m*p, R). Reshape to (m, p, R).
    # W has shape (p*n, R). Reshape to (p, n, R).
    
    U_reshaped = U_arr.reshape(n, m, -1)
    V_reshaped = V_arr.reshape(m, p, -1)
    W_reshaped = W_arr.reshape(p, n, -1)
    
    # Compute S[a, b, c, d, k, i] = sum_r U[a, b, r] V[c, d, r] W[k, i, r]
    # This is a tensor contraction.
    # S = np.einsum('abr,cdr,ki r->abcdki', U_reshaped, V_reshaped, W_reshaped)
    
    S = np.einsum('abr,cdr,kir->abcdki', U_reshaped, V_reshaped, W_reshaped)
    
    # Construct Target T
    # T[a, b, c, d, k, i] = 1 if a==i and b==c and d==k else 0
    # We can create this using broadcasting.
    # Indices:
    # a: 0..n-1
    # b: 0..m-1
    # c: 0..m-1
    # d: 0..p-1
    # k: 0..p-1
    # i: 0..n-1
    
    # delta_a_i: shape (n, 1, 1, 1, 1, n)
    # delta_b_c: shape (1, m, m, 1, 1, 1)
    # delta_d_k: shape (1, 1, 1, p, p, 1)
    
    a_idx = np.arange(n)
    b_idx = np.arange(m)
    c_idx = np.arange(m)
    d_idx = np.arange(p)
    k_idx = np.arange(p)
    i_idx = np.arange(n)
    
    delta_a_i = (a_idx[:, None, None, None, None, None] == i_idx[None, None, None, None, None, :]).astype(float)
    delta_b_c = (b_idx[None, :, None, None, None, None] == c_idx[None, None, :, None, None, None]).astype(float)
    delta_d_k = (d_idx[None, None, None, :, None, None] == k_idx[None, None, None, None, :, None]).astype(float)
    
    T = delta_a_i * delta_b_c * delta_d_k
    
    return np.allclose(S, T, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 6

    # Construct Rank 34 decomposition
    # Two independent 2x3x3 blocks
    rank_half = 17
    rank_total = 34
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # 2x3x3 Decomposition (Rank 17)
    # A: 2x3 (indices 0,1,2, 3,4,5)
    # B: 3x3 (indices 0,1,2, 3,4,5, 6,7,8)
    # C: 2x3 (indices 0,1, 2,3, 4,5)
    
    U_half = np.zeros((6, rank_half))
    V_half = np.zeros((9, rank_half))
    W_half = np.zeros((6, rank_half))
    
    # Strassen on A(0,1) x B(0,1) -> C(0,1)
    # P1: (A00+A11)(B00+B11)
    U_half[0, 0] = 1; U_half[4, 0] = 1
    V_half[0, 0] = 1; V_half[4, 0] = 1
    W_half[0, 0] = 1; W_half[3, 0] = 1

    # P2: (A10+A11)B00
    U_half[3, 1] = 1; U_half[4, 1] = 1
    V_half[0, 1] = 1
    W_half[1, 1] = 1

    # P3: A00(B01-B11)
    U_half[0, 2] = 1
    V_half[1, 2] = 1; V_half[4, 2] = -1
    W_half[2, 2] = 1; W_half[3, 2] = 1

    # P4: A11(B10-B00)
    U_half[4, 3] = 1
    V_half[3, 3] = 1; V_half[0, 3] = -1
    W_half[0, 3] = 1; W_half[1, 3] = 1

    # P5: (A00+A01)B11
    U_half[0, 4] = 1; U_half[1, 4] = 1
    V_half[4, 4] = 1
    W_half[0, 4] = -1; W_half[2, 4] = 1

    # P6: (A10-A00)(B00+B01)
    U_half[3, 5] = 1; U_half[0, 5] = -1
    V_half[0, 5] = 1; V_half[1, 5] = 1
    W_half[3, 5] = 1

    # P7: (A01-A11)(B10+B11)
    U_half[1, 6] = 1; U_half[4, 6] = -1
    V_half[3, 6] = 1; V_half[4, 6] = 1
    W_half[0, 6] = 1; W_half[3, 6] = 1

    # Column 2 of B: A(0,1,2) x B(0,1,2) col 2 -> C col 2
    # Q1: A00*B02
    U_half[0, 7] = 1
    V_half[2, 7] = 1
    W_half[4, 7] = 1

    # Q2: A01*B12
    U_half[1, 8] = 1
    V_half[5, 8] = 1
    W_half[4, 8] = 1

    # Q3: A10*B02
    U_half[3, 9] = 1
    V_half[2, 9] = 1
    W_half[5, 9] = 1

    # Q4: A11*B12
    U_half[4, 10] = 1
    V_half[5, 10] = 1
    W_half[5, 10] = 1

    # A col 2 x B(0,1,2) -> C(0,1,2)
    # R1: A02*B20
    U_half[2, 11] = 1
    V_half[6, 11] = 1
    W_half[0, 11] = 1

    # R2: A02*B21
    U_half[2, 12] = 1
    V_half[7, 12] = 1
    W_half[2, 12] = 1

    # R3: A02*B22
    U_half[2, 13] = 1
    V_half[8, 13] = 1
    W_half[4, 13] = 1

    # R4: A12*B20
    U_half[5, 14] = 1
    V_half[6, 14] = 1
    W_half[1, 14] = 1

    # R5: A12*B21
    U_half[5, 15] = 1
    V_half[7, 15] = 1
    W_half[3, 15] = 1

    # R6: A12*B22
    U_half[5, 16] = 1
    V_half[8, 16] = 1
    W_half[5, 16] = 1

    # Combine two 2x3x3 blocks into 2x3x6
    # Block 1: B cols 0-2, C cols 0-2
    # Block 2: B cols 3-5, C cols 3-5
    
    U[:, :17] = U_half
    U[:, 17:] = U_half

    for j in range(3):
        for k in range(3):
            # Block 1
            V[j * 6 + k, :17] = V_half[j * 3 + k, :17]
            # Block 2
            V[j * 6 + 3 + k, 17:] = V_half[j * 3 + k, :17]

    for k in range(3):
        for i in range(2):
            # Block 1
            W[k * 2 + i, :17] = W_half[k * 2 + i, :17]
            # Block 2
            W[(k + 3) * 2 + i, 17:] = W_half[k * 2 + i, :17]

    # Verify
    if check_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Schoolbook if verification fails (should not happen)
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
