import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 145.0 # 5 seconds margin
    
    # Try Rank 161 (Strassen 2x2 * Winograd 3x3)
    try:
        result_161 = construct_161()
        if verify_decomposition(result_161):
            return result_161
    except Exception as e:
        pass
        
    # Fallback to Rank 189 (Block Strassen)
    result_189 = construct_189()
    if verify_decomposition(result_189):
        return result_189
        
    # Fallback to Schoolbook Rank 216
    result_216 = construct_216()
    return result_216

def construct_216():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct_189():
    # Block Strassen: 3x3 blocks of 2x2 matrices
    # Rank = 3^3 * 7 = 189
    n, m, p = DIMENSIONS
    # We treat 6x6 as 3x3 blocks of 2x2
    # U, V, W for 2x2 Strassen
    U2, V2, W2 = get_strassen_2x2()
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map indices
    # i = i_block * 2 + i_local
    # j = j_block * 2 + j_local
    # k = k_block * 2 + k_local
    
    col = 0
    for ib in range(3):
        for jb in range(3):
            for kb in range(3):
                # This corresponds to the (ib, jb, kb) term in the 3x3 block multiplication
                # Wait, Block Strassen uses 7 products for the 2x2 block structure?
                # No, Block Strassen usually means using Strassen's algorithm on the blocks.
                # If we have 3x3 blocks, we use standard 27 multiplications?
                # No, the "Block-Strassen" that gave 189 likely used Strassen's algorithm for the 2x2 super-blocks?
                # 6x6 can be viewed as 2x2 blocks of 3x3 matrices.
                # Then we use Strassen for the 2x2 structure (7 products of 3x3).
                # Each 3x3 product is done with schoolbook (27 mults).
                # Total 7 * 27 = 189.
                
                # Let's implement this: 2x2 blocks of 3x3.
                pass
                
    # Correct construction for 189:
    # View 6x6 as 2x2 blocks of 3x3.
    # Use Strassen for the 2x2 block multiplication.
    # Each of the 7 intermediate products is a 3x3 matrix multiplication (schoolbook, rank 27).
    
    U2, V2, W2 = get_strassen_2x2()
    rank = 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r2 in range(7):
        for k3 in range(3): # local k in 3x3
            for i3 in range(3): # local i in 3x3
                # This is getting messy. Let's just use the tensor product logic.
                pass
                
    # Simpler 189: Tensor product of Strassen(2x2) and Schoolbook(3x3)
    return tensor_product_decomposition(get_strassen_2x2(), get_schoolbook_3x3())

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][3]=1 # C00, C11
    
    # M2 = (A10+A11)B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][0]=1; W[3][3]=-1 # C10, C11
    
    # M3 = A00(B01-B11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[0][1]=1; W[3][3]=1 # C01, C11
    
    # M4 = A11(B10-B00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][0]=1; W[1][1]=1 # C00, C10
    
    # M5 = (A00+A01)B11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][0]=-1; W[0][1]=1 # C00, C01
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][3]=1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][0]=1 # C00
    
    return U, V, W

def get_schoolbook_3x3():
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W

def tensor_product_decomp(U1, V1, W1, U2, V2, W2):
    # U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
    # U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
    # Result: (n1*n2*m1*m2, r1*r2)
    # n = n1*n2, m = m1*m2, p = p1*p2
    n1, m1, p1 = U1[0][0], V1[0][0], W1[0][0] # Dimensions not directly available, infer from shapes
    # Actually, we know the dimensions from the context.
    # For 2x2 * 3x3:
    # n1=2, m1=2, p1=2, r1=7
    # n2=3, m2=3, p2=3, r2=27 (or 23)
    
    n1, m1 = 2, 2
    n2, m2 = 3, 3
    n = n1 * n2
    m = m1 * m2
    p = n # square
    
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Index mapping:
    # i = i1 * n2 + i2
    # j = j1 * m2 + j2
    # k = k1 * p2 + k2
    
    # U[i, j, (r1, r2)] = U1[i1, j1, r1] * U2[i2, j2, r2]
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            for i1 in range(n1):
                for j1 in range(m1):
                    val1 = U1[i1 * m1 + j1][r1_idx]
                    if val1 == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            val2 = U2[i2 * m2 + j2][r2_idx]
                            if val2 == 0: continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j][r_idx] += val1 * val2
                            
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            for j1 in range(m1):
                for k1 in range(p1):
                    val1 = V1[j1 * p1 + k1][r1_idx]
                    if val1 == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            val2 = V2[j2 * p2 + k2][r2_idx]
                            if val2 == 0: continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k][r_idx] += val1 * val2
                            
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            for k1 in range(p1):
                for i1 in range(n1):
                    val1 = W1[k1 * n1 + i1][r1_idx]
                    if val1 == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            val2 = W2[k2 * n2 + i2][r2_idx]
                            if val2 == 0: continue
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            W[k * n + i][r_idx] += val1 * val2
                            
    return U, V, W

def construct_161():
    U2, V2, W2 = get_strassen_2x2()
    # Need Winograd 3x3 rank 23
    U3, V3, W3 = get_winograd_3x3()
    U, V, W = tensor_product_decomp(U2, V2, W2, U3, V3, W3)
    return {"U": U, "V": V, "W": W}

def get_winograd_3x3():
    # Placeholder for Winograd coefficients
    # If not implemented, return schoolbook
    return get_schoolbook_3x3()

def verify_decomposition(decomp):
    U = np.array(decomp['U'])
    V = np.array(decomp['V'])
    W = np.array(decomp['W'])
    # Compute tensor
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Note: U is (nm, rank), V is (mp, rank), W is (pn, rank)
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    n, m, p = 6, 6, 6
    rank = U.shape[1]
    
    # Efficient check:
    # Flatten U, V, W
    # We want to check if the decomposition is exact.
    # We can check a few random points or all points.
    # 6*6*6 = 216 points.
    # For each point (i, j, k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if (i,j,k) corresponds to a valid multiplication term?
    # No, the tensor for matrix multiplication has 1s at (i, j, k) for all i, j, k.
    # Wait, the multiplication tensor T has T[i, j, k] = 1 for all i, j, k?
    # Yes, C_ik = sum_j A_ij B_jk.
    # So the tensor is all ones.
    
    # Check if sum_r U[i, r] V[j, r] W[k, r] == 1 for all i, j, k.
    # Vectorized:
    # U_flat: (n*m, rank)
    # V_flat: (m*p, rank)
    # W_flat: (p*n, rank)
    
    # Create index arrays
    i_idx = np.arange(n)
    j_idx = np.arange(m)
    k_idx = np.arange(p)
    
    # This is O(n*m*p*rank). 216 * 161 ~ 35k ops. Fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j]
                v_val = V[j*p+k]
                w_val = W[k*n+i]
                dot = np.dot(u_val, v_val * w_val) # sum_r U * V * W
                if abs(dot - 1.0) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-END
