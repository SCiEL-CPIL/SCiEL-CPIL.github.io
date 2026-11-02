import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 3 and 2
    # Split p=5 into 3 and 2
    # Blocks:
    # 1. A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3] (3x3x3)
    # 2. A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3] (3x2x3)
    # 3. A[0:3, 0:3] * B[0:3, 3:5] -> C[0:3, 3:5] (3x3x2)
    # 4. A[0:3, 3:5] * B[3:5, 3:5] -> C[0:3, 3:5] (3x2x2)
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_block(n_b, m_b, p_b, i_off, j_off, k_off, U_b, V_b, W_b):
        # U_b is (n_b*m_b, r_b)
        # V_b is (m_b*p_b, r_b)
        # W_b is (p_b*n_b, r_b)
        # Append to global U, V, W
        global rank, U, V, W
        r_b = U_b.shape[1]
        rank += r_b
        
        # Pad U
        U_new = np.zeros((n*m, r_b))
        for i in range(n_b):
            for j in range(m_b):
                U_new[(i+i_off)*m + (j+j_off), :] = U_b[i*m_b + j, :]
        U.append(U_new)
        
        # Pad V
        V_new = np.zeros((m*p, r_b))
        for j in range(m_b):
            for k in range(p_b):
                V_new[(j+j_off)*p + (k+k_off), :] = V_b[j*p_b + k, :]
        V.append(V_new)
        
        # Pad W
        W_new = np.zeros((p*n, r_b))
        for k in range(p_b):
            for i in range(n_b):
                W_new[(k+k_off)*n + (i+i_off), :] = W_b[k*n_b + i, :]
        W.append(W_new)

    # Block 1: 3x3x3 Schoolbook
    n1, m1, p1 = 3, 3, 3
    r1 = n1*m1*p1
    U1 = np.zeros((n1*m1, r1))
    V1 = np.zeros((m1*p1, r1))
    W1 = np.zeros((p1*n1, r1))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i*m1+j, col] = 1
                V1[j*p1+k, col] = 1
                W1[k*n1+i, col] = 1
                col += 1
    add_block(3, 3, 3, 0, 0, 0, U1, V1, W1)
    
    # Block 2: 3x2x3 Schoolbook
    n2, m2, p2 = 3, 2, 3
    r2 = n2*m2*p2
    U2 = np.zeros((n2*m2, r2))
    V2 = np.zeros((m2*p2, r2))
    W2 = np.zeros((p2*n2, r2))
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i*m2+j, col] = 1
                V2[j*p2+k, col] = 1
                W2[k*n2+i, col] = 1
                col += 1
    add_block(3, 2, 3, 0, 3, 0, U2, V2, W2)
    
    # Block 3: 3x3x2 Schoolbook
    n3, m3, p3 = 3, 3, 2
    r3 = n3*m3*p3
    U3 = np.zeros((n3*m3, r3))
    V3 = np.zeros((m3*p3, r3))
    W3 = np.zeros((p3*n3, r3))
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i*m3+j, col] = 1
                V3[j*p3+k, col] = 1
                W3[k*n3+i, col] = 1
                col += 1
    add_block(3, 3, 2, 0, 0, 3, U3, V3, W3)
    
    # Block 4: 3x2x2 Optimized (Rank 6)
    # 3 independent 1x2x2 multiplications
    # 1x2x2 rank 2
    n4, m4, p4 = 3, 2, 2
    r4 = 6
    U4 = np.zeros((n4*m4, r4))
    V4 = np.zeros((m4*p4, r4))
    W4 = np.zeros((p4*n4, r4))
    col = 0
    for i in range(n4):
        # Term 1: a_i1 * b_1k
        # U[i, 0] = 1, V[0, k] = 1, W[k, i] = 1
        U4[i*m4 + 0, col] = 1
        V4[0*p4 + 0, col] = 1
        V4[0*p4 + 1, col] = 1 # Wait, V is m x p. j=0, k=0,1.
        # For 1x2x2: U is 1x2, V is 2x2, W is 2x1.
        # C_1 = a_1 b_11 + a_2 b_21. C_2 = a_1 b_12 + a_2 b_22.
        # Term 1: a_1 * [b_11, b_12]. U=[1,0], V=[1,0; 0,1]?? No.
        # V must be 2x2.
        # If we want a_1 * b_11 for C_1 and a_1 * b_12 for C_2.
        # We need V such that U V W gives the product.
        # Let's use 2 terms per i.
        # Term 1: a_i1 * b_1k.
        # U[i, 0] = 1.
        # V[0, 0] = 1, V[0, 1] = 1? No, V is fixed.
        # We need to separate k.
        # So 2 terms for k=0 and k=1?
        # Term 1: a_i1 * b_10. U[i,0]=1, V[0,0]=1, W[0,i]=1.
        # Term 2: a_i1 * b_11. U[i,0]=1, V[0,1]=1, W[1,i]=1.
        # Term 3: a_i2 * b_20. U[i,1]=1, V[1,0]=1, W[0,i]=1.
        # Term 4: a_i2 * b_21. U[i,1]=1, V[1,1]=1, W[1,i]=1.
        # Total 4 terms per i. 3*4 = 12.
        # This is schoolbook.
        # Is rank 6 possible?
        # 1x2x2 rank is 2.
        # For 1x2x2:
        # C_1 = a_1 b_1 + a_2 c_1.
        # C_2 = a_1 b_2 + a_2 c_2.
        # Rank 2?
        # c=1: U=[1,0], V=[b_1, 0], W=[1, 1]^T? No.
        # We need to factor the matrix [b_1, b_2; c_1, c_2] * [a_1; a_2]?
        # No, it's a tensor.
        # Actually, 1x2x2 rank is 2.
        # Let's check.
        # U is 1x2. V is 2x2. W is 2x1.
        # C_k = sum_j U_j V_jk W_k.
        # C_1 = U_1 V_11 W_1 + U_2 V_21 W_1.
        # C_2 = U_1 V_12 W_2 + U_2 V_22 W_2.
        # If we set W_1=1, W_2=1.
        # C_1 = U_1 V_11 + U_2 V_21.
        # C_2 = U_1 V_12 + U_2 V_22.
        # We want C_1 = a_1 b_1 + a_2 c_1.
        # C_2 = a_1 b_2 + a_2 c_2.
        # Set U_1=a_1, U_2=a_2.
        # V_11=b_1, V_21=c_1.
        # V_12=b_2, V_22=c_2.
        # This works with rank 1?
        # No, V is 2x2. We used 4 entries.
        # But rank is the number of terms c.
        # Here c=1.
        # So rank 1 for 1x2x2?
        # Wait, U is 1x2. V is 2x2. W is 2x1.
        # U V is 1x2. (U V)_k = sum_j U_j V_jk.
        # Then (U V W)_k = (U V)_k W_k.
        # So C_k = (sum_j U_j V_jk) W_k.
        # We need C_1 = a_1 b_1 + a_2 c_1.
        # C_2 = a_1 b_2 + a_2 c_2.
        # If W_1=1, W_2=1.
        # sum_j U_j V_j1 = a_1 b_1 + a_2 c_1.
        # sum_j U_j V_j2 = a_1 b_2 + a_2 c_2.
        # This requires V to depend on a, which is not allowed.
        # So rank 1 is not enough.
        # Rank 2?
        # c=1: U=[1,0], V=[b_1, b_2], W=[1, 1]^T.
        # C_1 = 1*b_1*1 + 0 = b_1. (Missing a_2 c_1)
        # c=2: U=[0,1], V=[c_1, c_2], W=[1, 1]^T.
        # C_1 = 0 + 1*c_1*1 = c_1.
        # C_2 = 0 + 1*c_2*1 = c_2.
        # Total C_1 = b_1 + c_1. We need a_1 b_1 + a_2 c_1.
        # So we need U to carry a.
        # c=1: U=[a_1, 0], V=[1, 1], W=[1, 1]^T.
        # C_1 = a_1*1*1 + 0 = a_1. (Need a_1 b_1)
        # So V must carry b.
        # c=1: U=[1, 0], V=[b_1, b_2], W=[1, 1]^T.
        # C_1 = b_1. C_2 = b_2.
        # c=2: U=[0, 1], V=[c_1, c_2], W=[1, 1]^T.
        # C_1 = c_1. C_2 = c_2.
        # We need to scale by a_1 and a_2.
        # So U must be [a_1, 0] and [0, a_2].
        # c=1: U=[a_1, 0], V=[1, 1], W=[1, 1]^T.
        # C_1 = a_1. C_2 = a_1.
        # This gives a_1, not a_1 b_1.
        # So V must be [b_1, b_2].
        # c=1: U=[a_1, 0], V=[b_1, b_2], W=[1, 1]^T.
        # C_1 = a_1 b_1. C_2 = a_1 b_2.
        # c=2: U=[0, a_2], V=[c_1, c_2], W=[1, 1]^T.
        # C_1 = a_2 c_1. C_2 = a_2 c_2.
        # Total C_1 = a_1 b_1 + a_2 c_1.
        # Total C_2 = a_1 b_2 + a_2 c_2.
        # This works! Rank 2 for 1x2x2.
        
        # So 3x2x2 rank is 3 * 2 = 6.
        
        # Construct U4, V4, W4 for 3x2x2
        # i in 0..2
        # Term 1 for i: U[i,0]=a_i1 (but U is constant 1? No, U depends on input A).
        # Wait, U, V, W are constant matrices. The inputs A, B are multiplied by them.
        # So U[i, 0] must be 1, and the input A[i, 0] provides the value.
        # So U4[i*m4 + 0, col] = 1.
        # V4[0*p4 + 0, col] = 1. V4[0*p4 + 1, col] = 1.
        # W4[0*n4 + i, col] = 1. W4[1*n4 + i, col] = 1.
        # This gives A[i,0] * (B[0,0] + B[0,1])?
        # No, V is 2x2. V[j, k].
        # V[0, 0] and V[0, 1].
        # If V[0, 0]=1, V[0, 1]=1, then sum_j U[i,j] V[j,k] = A[i,0]*1 + A[i,1]*0 = A[i,0].
        # Then multiply by W[k, i].
        # If W[0, i]=1, W[1, i]=1.
        # C[k, i] = sum_c U[i, j, c] V[j, k, c] W[k, i, c].
        # For c=1:
        # U[i, 0, 1] = 1.
        # V[0, 0, 1] = 1, V[0, 1, 1] = 1.
        # W[0, i, 1] = 1, W[1, i, 1] = 1.
        # Term = A[i,0] * (1*1 + 0) * 1 = A[i,0] for k=0.
        # Term = A[i,0] * (1*1 + 0) * 1 = A[i,0] for k=1.
        # This gives A[i,0] for both C[i,0] and C[i,1].
        # We need A[i,0] B[0,0] for C[i,0] and A[i,0] B[0,1] for C[i,1].
        # So V[0, 0] should be B[0,0]? No, V is constant.
        # The input B is multiplied by V.
        # So V[j, k] selects B[j, k].
        # If V[0, 0]=1, V[0, 1]=0, then we get B[0, 0].
        # If V[0, 0]=0, V[0, 1]=1, then we get B[0, 1].
        # So we need 2 terms for the first row of B.
        # And 2 terms for the second row of B.
        # Total 4 terms per i.
        # So rank 4 per i. Total 12.
        # My previous deduction of rank 2 for 1x2x2 was flawed because V is constant.
        # In the 1x2x2 case, V is 2x2.
        # C_1 = a_1 b_1 + a_2 c_1.
        # C_2 = a_1 b_2 + a_2 c_2.
        # c=1: U=[1, 0], V=[1, 0], W=[1, 0]^T.
        # C_1 = 1*1*1 + 0 = 1. (a_1 b_1)
        # C_2 = 1*0*0 + 0 = 0.
        # c=2: U=[1, 0], V=[0, 1], W=[0, 1]^T.
        # C_1 = 0. C_2 = 1. (a_1 b_2)
        # c=3: U=[0, 1], V=[1, 0], W=[1, 0]^T.
        # C_1 = 1. (a_2 c_1)
        # c=4: U=[0, 1], V=[0, 1], W=[0, 1]^T.
        # C_2 = 1. (a_2 c_2)
        # So rank 4 for 1x2x2.
        # Then 3x2x2 rank is 12.
        # So no improvement for 3x2x2.
        
        # So total rank is 27 + 18 + 18 + 12 = 75.
        # Back to 75.
        
        # I will stick to the schoolbook decomposition for all blocks to be safe, which sums to 75.
        # But the block structure allows me to potentially swap in better decompositions later.
        # For now, I will implement the block structure with schoolbook for all.
        
        # Actually, I can just use the original schoolbook code.
        # But the prompt asks for a stronger algorithm.
        # I will return the schoolbook result for now, as I cannot guarantee a lower rank without a known decomposition.
        
        pass

    # Since I couldn't find a guaranteed lower rank, I will return the schoolbook decomposition.
    # But I will structure it to allow easy swapping.
    
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
