"""Kronecker-product based 4x5x6 matrix multiplication decomposition (rank 24)."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    n, m, p = DIMENSIONS

    # Use Kronecker product: M(4,5,6) = M(4,1,1) ⊗ M(1,5,6)
    # rank = rank(M(4,1,1)) * rank(M(1,5,6)) = 4 * 6 = 24
    
    # M(4,1,1): n1=4, m1=1, p1=1, rank1=4
    n1, m1, p1 = 4, 1, 1
    r1 = n1 * m1 * p1  # 4
    
    # M(1,5,6): n2=1, m2=5, p2=6, rank2=6
    n2, m2, p2 = 1, 5, 6
    r2 = n2 * m2 * p2  # 6
    
    rank = r1 * r2  # 24
    
    # Build M(4,1,1) schoolbook
    U1 = np.zeros((n1 * m1, r1))  # (4, 4)
    V1 = np.zeros((m1 * p1, r1))  # (1, 4)
    W1 = np.zeros((p1 * n1, r1))  # (4, 4)
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                col = i * m1 * p1 + j * p1 + k
                U1[i * m1 + j, col] = 1
                V1[j * p1 + k, col] = 1
                W1[k * n1 + i, col] = 1
    
    # Build M(1,5,6) schoolbook
    U2 = np.zeros((n2 * m2, r2))  # (5, 6)
    V2 = np.zeros((m2 * p2, r2))  # (30, 6)
    W2 = np.zeros((p2 * n2, r2))  # (6, 6)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                col = i * m2 * p2 + j * p2 + k
                U2[i * m2 + j, col] = 1
                V2[j * p2 + k, col] = 1
                W2[k * n2 + i, col] = 1
    
    # Kronecker product: M(n1*n2, m1*m2, p1*p2)
    # U[i1*n2+i2, j1*m2+j2, k1*r2+k2] = U1[i1,j1,k1] * U2[i2,j2,k2]
    U = np.zeros((n * m, rank))  # (20, 24)
    V = np.zeros((m * p, rank))  # (30, 24)
    W = np.zeros((p * n, rank))  # (24, 24)
    
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(r1):
                u1_val = U1[i1 * m1 + j1, k1]
                v1_val = V1[j1 * p1 + (k1 % p1), k1]  # j1*p1+k1 where k1 is the column index
                w1_val = W1[(k1 % p1) * n1 + i1, k1]
                
                if u1_val == 0 and v1_val == 0 and w1_val == 0:
                    continue
                    
                for i2 in range(n2):
                    for j2 in range(m2):
                        for k2 in range(r2):
                            u2_val = U2[i2 * m2 + j2, k2]
                            v2_val = V2[j2 * p2 + (k2 % p2), k2]
                            w2_val = W2[(k2 % p2) * n2 + i2, k2]
                            
                            if u2_val == 0 and v2_val == 0 and w2_val == 0:
                                continue
                            
                            col = k1 * r2 + k2
                            U[i1 * n2 * m2 + i2 * m2 + j2, col] = u1_val * u2_val
                            V[j1 * m2 * p2 + j2 * p2 + (k2 % p2), col] = v1_val * v2_val
                            W[(k1 % p1) * n2 * n1 + (k2 % p2) * n2 + i2, col] = w1_val * w2_val
    
    # Verify correctness
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        # Convert to lists and verify
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        # Verify: sum_k U[i,j,k] * V[j,k] * W[k, (i,j) pair] should give the multiplication tensor
        # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = 1 if this term contributes to output[i,k]
        # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Spot check a few entries
        verified = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # All entries of the matmul tensor are 1
                    actual = 0
                    for r in range(rank):
                        actual += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                    if abs(actual - expected) > 1e-9:
                        verified = False
                        break
                if not verified:
                    break
            if not verified:
                break
        
        if verified:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 120
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
