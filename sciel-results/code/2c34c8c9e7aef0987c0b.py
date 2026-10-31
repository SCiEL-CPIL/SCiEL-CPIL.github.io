import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Initialize factor matrices
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 decomposition coefficients
    # For 2x2 matrices A, B, C:
    # C = A * B
    # Indices for A: 00, 01, 10, 11 -> flat index i*2+j
    # Indices for B: 00, 01, 10, 11 -> flat index i*2+j
    # Indices for C: 00, 01, 10, 11 -> flat index i*2+j
    
    # We define the 7 multiplications M_0 ... M_6
    # Each M_s = L_s(A) * R_s(B)
    # C is a linear combination of M_s
    
    # U2[row, s] is coefficient of A[row] in L_s(A)
    # V2[row, s] is coefficient of B[row] in R_s(B)
    # W2[row, s] is coefficient of M_s in C[row]
    
    # Rows are 0:00, 1:01, 2:10, 3:11
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]

    # M0 = (A00 + A11) * (B00 + B11)
    # C00 += M0, C11 += M0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1

    # M1 = (A10 + A11) * B00
    # C10 += M1, C11 -= M1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1

    # M2 = A00 * (B01 - B11)
    # C01 += M2, C11 += M2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1

    # M3 = A11 * (B10 - B00)
    # C00 += M3, C10 += M3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1

    # M4 = (A00 + A01) * B11
    # C00 -= M4, C01 += M4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    # C11 += M5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    # C00 += M6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # Construct 4x4 decomposition using Kronecker product logic
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    # Let A = [[A00, A01], [A10, A11]] where Aij are 2x2 blocks.
    # The block multiplication uses Strassen (indices s=0..6).
    # Each block multiplication (e.g., A00*B00) uses Strassen (indices t=0..6).
    # Total rank = 7 * 7 = 49.
    
    # Index mapping for 4x4:
    # A[i, j] where i, j in 0..3
    # Block row I = i // 2, Block col J = j // 2
    # Inner row ii = i % 2, Inner col jj = j % 2
    # Flat index for A: i * 4 + j
    # Flat index for B: k * 4 + l
    # Flat index for C: j * 4 + i (Note: C is transposed in standard matmul tensor T[i,k,j] -> C[i,j]? 
    # Wait, the task says W[k*n+i, r]. 
    # Standard matmul: C[i, j] = sum_k A[i, k] B[k, j].
    # Tensor T[a, b, c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i.
    # So c = j*4 + i.
    # My W matrix rows correspond to c index.
    # So W[row_w, rank_idx] corresponds to C[j, i] contribution.
    # In my W2, row is C index. C2[ii, jj] -> flat ii*2+jj.
    # In 4x4, C block is C[I, J] which contains C2[ii, jj].
    # C flat index = j * 4 + i = (J*2+jj)*4 + (I*2+ii) = J*8 + jj*4 + I*2 + ii.
    
    for s in range(7):
        for t in range(7):
            rank_idx = s * 7 + t
            
            # Update U
            # U_flat[i*4+j, rank_idx] = U_block[I*2+J, s] * U_inner[ii*2+jj, t]
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    J = j // 2
                    ii = i % 2
                    jj = j % 2
                    # Block index for U2: I*2+J
                    # Inner index for U2: ii*2+jj
                    coeff_block = U2[I * 2 + J][s]
                    coeff_inner = U2[ii * 2 + jj][t]
                    coeff = coeff_block * coeff_inner
                    if coeff != 0:
                        U[i * 4 + j][rank_idx] = coeff

            # Update V
            # V_flat[k*4+l, rank_idx] = V_block[K*2+L, s] * V_inner[kk*2+ll, t]
            # Note: B[k, l]
            # Block row K, Block col L
            # Inner row kk, Inner col ll
            for k in range(4):
                for l in range(4):
                    K = k // 2
                    L = l // 2
                    kk = k % 2
                    ll = l % 2
                    coeff_block = V2[K * 2 + L][s]
                    coeff_inner = V2[kk * 2 + ll][t]
                    coeff = coeff_block * coeff_inner
                    if coeff != 0:
                        V[k * 4 + l][rank_idx] = coeff

            # Update W
            # W_flat[j*4+i, rank_idx] = W_block[J*2+I, s] * W_inner[jj*2+ii, t]
            # Note: C[j, i] corresponds to W row j*4+i
            # Block row J (col of C block), Block col I (row of C block)?
            # C = A * B. C[I, J] = A[I, :] B[:, J].
            # In Strassen block formula, C_block[I, J] depends on A_block[I, K] and B_block[K, J].
            # The W2 matrix for 2x2 Strassen maps M_s to C_block[I, J].
            # So W2[I*2+J, s] is the coeff for C_block[I, J].
            # Here, C_block is at block position (I, J) where I is row, J is col.
            # The flat index of C is j*4 + i.
            # j = J*2 + jj (col), i = I*2 + ii (row).
            # So we need W2[I*2+J, s] for the block part.
            # And W2[ii*2+jj, t] for the inner part.
            # Let's verify indices.
            # W2 row index is C_flat_2x2 = row*2 + col.
            # Block C index: row=I, col=J. So W2_block_row = I*2+J.
            # Inner C index: row=ii, col=jj. So W2_inner_row = ii*2+jj.
            
            for j in range(4):
                for i in range(4):
                    J = j // 2
                    I = i // 2
                    jj = j % 2
                    ii = i % 2
                    coeff_block = W2[I * 2 + J][s]
                    coeff_inner = W2[ii * 2 + jj][t]
                    coeff = coeff_block * coeff_inner
                    if coeff != 0:
                        W[j * 4 + i][rank_idx] = coeff

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
